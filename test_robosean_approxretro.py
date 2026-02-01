#!/usr/bin/env python
"""Test script for robosean model - molecule-conditional (approx retro) evaluation with batched generation."""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Test RoboSean - Molecule Conditioning (ApproxRetro)")

    # Data
    parser.add_argument("--source_molecules", type=str, required=True,
                        help="Path to parquet with source SMILES")
    parser.add_argument("--out_file", type=str, required=True,
                        help="Output parquet path")
    parser.add_argument("--force_reactants", type=str, nargs='+', default=[],
                        help="Paths to reactant library parquets (space-separated)")

    # Model
    parser.add_argument("--checkpoint", type=str,
                        default="s3://shvaibackups/robosean/500m_robosean_molprefix2/75001.pt")
    parser.add_argument("--emb_dim", type=int, default=1024)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=150)

    # Generation
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_mols", type=int, default=100,
                        help="Limit number of molecules to process (for testing)")

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
        max_position_embeddings=args.max_len + 200,  # Extra room for mol prefix
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


def remove_cls_prefix(decoded: str) -> str:
    """Remove [CLS]smiles[CLS] prefix from decoded string."""
    first_cls = decoded.find('[CLS]')
    if first_cls >= 0:
        second_cls = decoded.find('[CLS]', first_cls + 5)
        if second_cls >= 0:
            return decoded[:first_cls] + decoded[second_cls + 5:]
    return decoded


def batched_generate_with_swapping(
    model,
    tokenizer,
    input_prefix: str,  # The prefix string ([CLS]smiles[CLS][BOS])
    batch_size: int,
    reactant_lib,
    max_len: int,
    temperature: float,
    top_p: float,
    device,
):
    """Batched generation with inline reactant swapping."""
    eos_id = tokenizer.convert_tokens_to_ids('[EOS]')
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')

    initial_tokens = tokenizer.encode(input_prefix)
    sequences = [list(initial_tokens) for _ in range(batch_size)]

    finished = [False] * batch_size
    num_swapped = [0] * batch_size
    swaps = [[] for _ in range(batch_size)]

    # For path extraction - need to handle [CLS]...[CLS] prefix
    clean_prefix = input_prefix.replace(' ', '')

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
                    # Remove [CLS]...[CLS] and extract path
                    decoded_clean = remove_cls_prefix(decoded)
                    path_part = decoded_clean.replace('[BOS]', '').replace('[EOS]', '')

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
                decoded_clean = remove_cls_prefix(decoded)
                path_part = decoded_clean.replace('[BOS]', '')

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



def calculate_tanimoto(target_smiles: str, generated_smiles: str) -> float:
    """Calculate Tanimoto similarity between target and generated SMILES."""
    try:
        target_mol = Chem.MolFromSmiles(target_smiles)
        gen_mol = Chem.MolFromSmiles(generated_smiles)
        if target_mol is None or gen_mol is None:
            return 0.0
        target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, radius=3, nBits=1024)
        gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, radius=3, nBits=1024)
        return DataStructs.TanimotoSimilarity(target_fp, gen_fp)
    except Exception:
        return 0.0


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("synthesis/synthesis_tokenizer")

    # Load model
    model = load_model(args, tokenizer, device)
    print("Model loaded!")

    # Load reactions
    reactions = get_reactions()

    # Load reactant library
    reactant_lib = None
    if args.force_reactants:
        reactant_lib = ReactantLibrary(args.force_reactants)

    # Load source molecules
    print(f"Loading source molecules from {args.source_molecules}...")
    df = pd.read_parquet(args.source_molecules)
    source_smiles = df['smiles'].tolist()
    if args.num_mols is not None:
        source_smiles = source_smiles[:args.num_mols]
    print(f"Loaded {len(source_smiles)} molecules")

    # Results accumulator
    all_results = []

    # Process each molecule
    for mol_idx, smi in enumerate(tqdm(source_smiles, desc="Generating")):
        # Validate source SMILES
        source_mol = Chem.MolFromSmiles(smi)
        if source_mol is None:
            for shot in range(args.shots):
                all_results.append({
                    'source_smiles_idx': mol_idx,
                    'source_smiles': smi,
                    'synthesis_path': None,
                    'generated_smiles': None,
                    'valid': False,
                    'tanimoto_sim': 0.0,
                    'shot_number': shot,
                })
            continue

        # Build input prefix: [CLS]{smiles}[CLS][BOS]
        input_prefix = f'[CLS]{smi}[CLS][BOS]'

        # Generate all shots in batch
        batch_results = batched_generate_with_swapping(
            model=model,
            tokenizer=tokenizer,
            input_prefix=input_prefix,
            batch_size=args.shots,
            reactant_lib=reactant_lib,
            max_len=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        # Process each shot result
        valid_count = 0
        best_tanimoto = 0.0

        for shot, (decoded, swaps) in enumerate(batch_results):
            # Debug: print first few decoded outputs
            if shot < 3 and mol_idx == 0:
                print(f"  [DEBUG shot {shot}] raw decoded: {decoded[:200]}...")

            # Clean decoded string - remove [CLS]...[CLS] prefix
            clean = remove_cls_prefix(decoded)
            for tok in ['[BOS]', '[EOS]', '[PAD]'] + property_tokens:
                clean = clean.replace(tok, '')
            clean = clean.strip()

            if shot < 3 and mol_idx == 0:
                print(f"  [DEBUG shot {shot}] clean: {clean[:200]}...")

            # Try to execute path
            generated_smiles = None
            valid = False
            tanimoto = 0.0

            if clean.startswith('<ADD>'):
                try:
                    result, _ = execute_path(clean, reactions)
                    if result is not None:
                        mol = Chem.MolFromSmiles(result)
                        if mol is not None:
                            generated_smiles = Chem.MolToSmiles(mol, canonical=True)
                            valid = True
                            tanimoto = calculate_tanimoto(smi, generated_smiles)
                            valid_count += 1
                            best_tanimoto = max(best_tanimoto, tanimoto)
                except Exception as e:
                    if shot < 3 and mol_idx == 0:
                        print(f"  [DEBUG shot {shot}] execute_path error: {e}")
            else:
                if shot < 3 and mol_idx == 0:
                    print(f"  [DEBUG shot {shot}] doesn't start with <ADD>")

            all_results.append({
                'source_smiles_idx': mol_idx,
                'source_smiles': smi,
                'synthesis_path': clean if clean.startswith('<ADD>') else None,
                'generated_smiles': generated_smiles,
                'valid': valid,
                'tanimoto_sim': tanimoto,
                'shot_number': shot,
            })

        # Print progress
        print(f"\n[Mol {mol_idx}] SMILES: {smi[:50]}...")
        print(f"[Mol {mol_idx}] Valid: {valid_count}/{args.shots} ({100*valid_count/args.shots:.1f}%)")
        print(f"[Mol {mol_idx}] Best Tanimoto: {best_tanimoto:.4f}")

    # Save results
    os.makedirs(os.path.dirname(args.out_file) if os.path.dirname(args.out_file) else '.', exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(args.out_file, index=False)
    print(f"\nSaved {len(results_df)} rows to {args.out_file}")

    # Print summary stats
    valid_df = results_df[results_df['valid']]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(results_df)}")
    print(f"Valid generations: {len(valid_df)} ({100*len(valid_df)/len(results_df):.1f}%)")
    if len(valid_df) > 0:
        print(f"Mean Tanimoto (valid only): {valid_df['tanimoto_sim'].mean():.4f}")
        print(f"P50 Tanimoto (valid only): {valid_df['tanimoto_sim'].quantile(0.5):.4f}")
        print(f"P90 Tanimoto (valid only): {valid_df['tanimoto_sim'].quantile(0.9):.4f}")

        # Best-of-N stats
        best_per_mol = results_df.groupby('source_smiles_idx').apply(
            lambda x: x[x['valid']]['tanimoto_sim'].max() if x['valid'].any() else 0.0
        )
        print(f"\nBest-of-{args.shots} stats:")
        print(f"  Mean Tanimoto: {best_per_mol.mean():.4f}")
        print(f"  P50 Tanimoto: {best_per_mol.quantile(0.5):.4f}")
        print(f"  P90 Tanimoto: {best_per_mol.quantile(0.9):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

