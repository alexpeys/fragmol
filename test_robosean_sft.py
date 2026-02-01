#!/usr/bin/env python
"""Test script for robosean SFT model - unconditional generation with classifier scoring."""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from transformers import PreTrainedTokenizerFast

from utils.models import EmbeddingConditionalLlamaDecoder, create_custom_llama_config
from synthesis.helpers import get_reactions, execute_path
from synthesis.dataloader import property_tokens
from s3torchconnector import S3Checkpoint


class ReactantLibrary:
    """Fast nearest-neighbor lookup for reactants using fingerprints."""

    def __init__(self, parquet_paths: list[str]):
        """Load and concatenate multiple reactant libraries."""
        all_smiles = []
        all_fps = []

        for path in parquet_paths:
            print(f"Loading reactant library from {path}...")
            df = pd.read_parquet(path)
            all_smiles.extend(df['smiles'].tolist())

            for fp_arr in tqdm(df['fingerprint'].tolist(), desc="Loading fingerprints"):
                bv = DataStructs.ExplicitBitVect(1024)
                on_bits = np.where(fp_arr)[0].tolist()
                bv.SetBitsFromList(on_bits)
                all_fps.append(bv)

        self.smiles = all_smiles
        self.fps = all_fps
        print(f"Total reactants loaded: {len(self.smiles):,}")

    def find_closest(self, smiles: str) -> str:
        """Find closest reactant to given SMILES by Tanimoto similarity."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.smiles[0]
        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, self.fps)
        return self.smiles[int(np.argmax(sims))]


def parse_args():
    parser = argparse.ArgumentParser(description="Test RoboSean SFT - Unconditional Generation")

    # Output
    parser.add_argument("--out_file", type=str, required=True,
                        help="Output parquet path")
    parser.add_argument("--force_reactants", type=str, nargs='+', default=[],
                        help="Paths to reactant library parquets (space-separated)")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (local or s3://)")
    parser.add_argument("--classifier_path", type=str, required=True,
                        help="S3 path to classifier folder")
    parser.add_argument("--classifier_batch_size", type=int, default=512)

    # Architecture
    parser.add_argument("--emb_dim", type=int, default=1024)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=200)

    # Generation
    parser.add_argument("--num_mols", type=int, default=1024,
                        help="Number of molecules to generate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_model(args, tokenizer, device):
    """Load model from checkpoint."""
    vocab_size = max(tokenizer.get_vocab().values()) + 1

    config = create_custom_llama_config(
        vocab_size=vocab_size,
        hidden_size=args.emb_dim,
        intermediate_size=int(args.emb_dim * args.intermediate_size_multiplier),
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        pad_token_id=tokenizer.convert_tokens_to_ids('[PAD]'),
        bos_token_id=tokenizer.convert_tokens_to_ids('[BOS]'),
        eos_token_id=tokenizer.convert_tokens_to_ids('[EOS]'),
        cls_token_id=tokenizer.convert_tokens_to_ids('[BOS]'),
        max_position_embeddings=args.max_len + 50,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        rope_base=10_000,
        use_rope=True,
    )

    model = EmbeddingConditionalLlamaDecoder(config)

    print(f"Loading checkpoint: {args.checkpoint}")
    if args.checkpoint.startswith('s3://'):
        with S3Checkpoint("us-west-2").reader(args.checkpoint) as reader:
            checkpoint = torch.load(reader, map_location='cpu')
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
        checkpoint = {k.replace('_orig_mod.', '', 1): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    del checkpoint

    model = model.to(device)
    model.eval()
    return model


def count_add_tags(s: str) -> int:
    return s.count('<ADD>')


def get_reactant_at_index(path: str, idx: int) -> tuple[int, int, str] | None:
    """Get the start, end positions and SMILES of the idx-th reactant in path."""
    add_positions = []
    pos = 0
    while True:
        pos = path.find('<ADD>', pos)
        if pos == -1:
            break
        add_positions.append(pos)
        pos += 5

    if idx >= len(add_positions):
        return None

    start = add_positions[idx] + 5
    remaining = path[start:]

    if '<' in remaining:
        end = start + remaining.index('<')
    else:
        end = len(path)

    return start, end, path[start:end]


def batched_generate_with_swapping(
    model,
    tokenizer,
    input_prefix: str,
    batch_size: int,
    reactant_lib: ReactantLibrary | None,
    max_len: int,
    temperature: float,
    top_p: float,
    device,
):
    """
    Batched generation with inline reactant swapping.
    When a reactant is completed, swap it and re-encode that sequence.
    """
    eos_id = tokenizer.convert_tokens_to_ids('[EOS]')
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')

    initial_tokens = tokenizer.encode(input_prefix)
    sequences = [list(initial_tokens) for _ in range(batch_size)]

    finished = [False] * batch_size
    num_swapped = [0] * batch_size
    swaps = [[] for _ in range(batch_size)]

    prefix_len = len(input_prefix)

    for step in range(max_len):
        if all(finished):
            break

        max_seq_len = max(len(seq) for seq in sequences)
        padded = []
        attention_masks = []

        for seq in sequences:
            pad_len = max_seq_len - len(seq)
            padded.append([pad_id] * pad_len + seq)
            attention_masks.append([0] * pad_len + [1] * len(seq))

        input_tensor = torch.tensor(padded, device=device)
        attention_mask = torch.tensor(attention_masks, device=device)

        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_tensor, attention_mask=attention_mask, labels=None)

        logits = outputs['logits'][:, -1, :]

        if temperature > 0:
            logits = logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for i in range(batch_size):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i, indices_to_remove] = float('-inf')

        if temperature == 0:
            next_tokens = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        for i in range(batch_size):
            if finished[i]:
                continue

            next_tok = next_tokens[i].item()
            sequences[i].append(next_tok)

            if next_tok == eos_id:
                finished[i] = True
                if reactant_lib is not None:
                    decoded = tokenizer.decode(sequences[i]).replace(' ', '')
                    if decoded.startswith(input_prefix.replace(' ', '')):
                        path_part = decoded[len(input_prefix.replace(' ', '')):]
                    else:
                        path_part = decoded
                    path_part = path_part.replace('[EOS]', '')

                    n_reactants = count_add_tags(path_part)
                    while num_swapped[i] < n_reactants:
                        result = get_reactant_at_index(path_part, num_swapped[i])
                        if result is None:
                            break
                        start, end, original_smi = result
                        if original_smi and not original_smi.startswith('<'):
                            swapped_smi = reactant_lib.find_closest(original_smi)
                            swaps[i].append((original_smi, swapped_smi))
                            path_part = path_part[:start] + swapped_smi + path_part[end:]
                        num_swapped[i] += 1

                    full_str = input_prefix + path_part + '[EOS]'
                    sequences[i] = tokenizer.encode(full_str)
                continue

            if reactant_lib is not None:
                decoded = tokenizer.decode(sequences[i]).replace(' ', '')
                if decoded.startswith(input_prefix.replace(' ', '')):
                    path_part = decoded[len(input_prefix.replace(' ', '')):]
                else:
                    path_part = decoded

                n_reactants = count_add_tags(path_part)

                if path_part.endswith('<') and n_reactants > num_swapped[i]:
                    while num_swapped[i] < n_reactants:
                        result = get_reactant_at_index(path_part, num_swapped[i])
                        if result is None:
                            break
                        start, end, original_smi = result
                        if original_smi and not original_smi.startswith('<'):
                            swapped_smi = reactant_lib.find_closest(original_smi)
                            swaps[i].append((original_smi, swapped_smi))
                            path_part = path_part[:start] + swapped_smi + path_part[end:]
                        num_swapped[i] += 1

                    full_str = input_prefix + path_part
                    sequences[i] = tokenizer.encode(full_str)

    results = []
    for i in range(batch_size):
        decoded = tokenizer.decode(sequences[i]).replace(' ', '')
        results.append((decoded, swaps[i]))

    return results


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("synthesis/synthesis_tokenizer")

    # Load model
    model = load_model(args, tokenizer, device)
    print("Model loaded!")

    # Load reactions
    reactions = get_reactions()

    # Load reactant library if specified
    reactant_lib = None
    if args.force_reactants:
        reactant_lib = ReactantLibrary(args.force_reactants)

    # Generate in batches with inline swapping
    gen_batch_size = 128
    all_results = []  # List of (decoded_path, swaps)

    print(f"\nGenerating {args.num_mols} synthesis paths...")
    for batch_start in range(0, args.num_mols, gen_batch_size):
        batch_end = min(batch_start + gen_batch_size, args.num_mols)
        batch_size = batch_end - batch_start

        batch_results = batched_generate_with_swapping(
            model=model,
            tokenizer=tokenizer,
            input_prefix='[BOS]',
            batch_size=batch_size,
            reactant_lib=reactant_lib,
            max_len=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        all_results.extend(batch_results)
        print(f"  Generated {len(all_results)}/{args.num_mols}...")

    # Parse and execute synthesis paths
    results = []
    seen_smiles = set()

    print(f"\nExecuting synthesis paths...")
    for decoded, swaps in tqdm(all_results, desc="Executing"):
        try:
            # Clean
            clean_path = decoded
            for tok in ['[BOS]', '[EOS]', '[PAD]'] + property_tokens:
                clean_path = clean_path.replace(tok, '')
            clean_path = clean_path.strip()

            if not clean_path.startswith('<ADD>'):
                continue

            # Execute
            result_smiles, _ = execute_path(clean_path, reactions)
            if result_smiles is None:
                continue

            mol = Chem.MolFromSmiles(result_smiles)
            if mol is None:
                continue

            canonical = Chem.MolToSmiles(mol, canonical=True)

            # Dedupe
            if canonical in seen_smiles:
                continue
            seen_smiles.add(canonical)

            results.append({'synthesis_path': clean_path, 'smiles': canonical})

        except Exception:
            continue

    print(f"\nGenerated {len(results)} unique valid molecules")

    if not results:
        print("No valid molecules generated!")
        return

    # Run classifier
    print(f"\nRunning classifier on {len(results)} molecules...")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        tmp_input = f.name
        pd.DataFrame({'smiles': [r['smiles'] for r in results]}).to_csv(f, index=False)

    tmp_output = tmp_input.replace('.csv', '_with_scores.csv')

    try:
        cmd = [
            'python', '/home/ubuntu/lolmol_inference_server/run_classifier.py',
            '--classifier_path', args.classifier_path,
            '--eval_data_path', tmp_input,
            '--batch_size', str(args.classifier_batch_size),
        ]
        subprocess.run(cmd, check=True, cwd='/home/ubuntu/lolmol_inference_server')

        scores_df = pd.read_csv(tmp_output)
        for i, row in scores_df.iterrows():
            results[i]['logit_mean'] = row['logit_mean']
            results[i]['logit_std'] = row['logit_std']

    finally:
        if os.path.exists(tmp_input):
            os.remove(tmp_input)
        if os.path.exists(tmp_output):
            os.remove(tmp_output)

    # Save results
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_file) if os.path.dirname(args.out_file) else '.', exist_ok=True)
    df.to_parquet(args.out_file, index=False)

    print(f"\nSaved {len(df)} molecules to {args.out_file}")
    print(f"Classifier scores: mean={df['logit_mean'].mean():.4f}, p90={df['logit_mean'].quantile(0.9):.4f}, p99={df['logit_mean'].quantile(0.99):.4f}")


if __name__ == "__main__":
    main()

