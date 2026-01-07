#!/usr/bin/env python
"""Test script for robosean model - evaluates property conditioning accuracy."""
import argparse
import random
import re
from collections import defaultdict

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from utils.models import EmbeddingConditionalLlamaDecoder, create_custom_llama_config
from synthesis.helpers import get_reactions, execute_path
from synthesis.dataloader import property_tokens, calculate_properties
from s3torchconnector import S3Checkpoint


# Property categories for reporting
PROPERTY_CATEGORIES = {
    'LOG1P': ['[logp_neg]', '[logp_0_1.5]', '[logp_1.5_3]', '[logp_3_5]', '[logp_5_plus]'],
    'FLEX': ['[flex_rigid]', '[flex_moderate]', '[flex_flexible]'],
    'FLAT': ['[flat_planar]', '[flat_mixed]', '[flat_3d]'],
    'SYNTH': ['[synth_easy]', '[synth_medium]'],
    'MEMBRANE': ['[membrane_bbb]', '[membrane_oral]'],
    'DRUGLIKE': ['[highly_druglike]'],
}


class ReactantLibrary:
    """Fast nearest-neighbor lookup for reactants using fingerprints."""

    def __init__(self, parquet_path):
        print(f"Loading reactant library from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        self.smiles = df['smiles'].tolist()

        # Convert fingerprints to RDKit BitVects for fast Tanimoto
        self.fps = []
        for fp_arr in tqdm(df['fingerprint'].tolist(), desc="Loading fingerprints"):
            bv = DataStructs.ExplicitBitVect(1024)
            on_bits = np.where(fp_arr)[0].tolist()
            bv.SetBitsFromList(on_bits)
            self.fps.append(bv)

        print(f"Loaded {len(self.smiles):,} reactants")

    def find_closest(self, smiles):
        """Find closest reactant to given SMILES by Tanimoto similarity."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.smiles[0]  # fallback to first

        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

        # Bulk Tanimoto
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, self.fps)
        best_idx = int(np.argmax(sims))
        return self.smiles[best_idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Test RoboSean model")
    
    # Checkpoint
    parser.add_argument("--checkpoint", type=str, 
                        default="s3://shvaibackups/robosean/500m_robosean/60001.pt")
    
    # Model architecture (same as train_robosean)
    parser.add_argument("--emb_dim", type=int, default=1024)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=150)
    
    # Test params
    parser.add_argument("--num_mols", type=int, default=100)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_reactants", type=str, default="synthesis/10k_reactants.parquet",
                        help="Path to reactant library parquet. Set to '' to disable.")
    
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
        max_position_embeddings=args.max_len,
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
    
    # Handle compiled model state dict
    if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
        checkpoint = {k.replace('_orig_mod.', '', 1): v for k, v in checkpoint.items()}
    
    model.load_state_dict(checkpoint)
    del checkpoint
    
    model = model.to(device)
    model.eval()
    
    return model


def generate_with_reactant_swapping(
    model,
    tokenizer,
    input_ids: list[int],
    reactant_lib: ReactantLibrary,
    max_len: int,
    temperature: float,
    top_p: float,
    device,
):
    """
    Generate a synthesis path, swapping each reactant with library match as it's generated.

    When the model finishes generating a reactant (after <ADD>smiles<), we:
    1. Stop
    2. Swap the smiles with closest library match
    3. Continue generation with swapped molecule in context
    """
    import torch.nn.functional as F

    eos_id = tokenizer.convert_tokens_to_ids('[EOS]')

    # Decode the input prefix (property tokens) to know where path starts
    input_prefix = tokenizer.decode(input_ids).replace(' ', '')

    # Track generated tokens
    generated = list(input_ids)
    swaps = []  # Track (original, swapped) pairs
    num_reactants_swapped = 0  # Track how many reactants we've processed

    def count_reactants(path):
        """Count number of <ADD> tags in path."""
        return path.count('<ADD>')

    def swap_last_reactant(path_part, ends_with_bracket=True):
        """Swap the last reactant in path_part. Returns (new_path, original, swapped) or None."""
        if '<ADD>' not in path_part:
            return None

        last_add_pos = path_part.rfind('<ADD>')
        after_add = path_part[last_add_pos + 5:]

        if ends_with_bracket:
            # Path ends with '<', so molecule is between <ADD> and final <
            between = after_add[:-1]  # Remove final <
        else:
            # Path doesn't end with <, molecule is up to first < (if any) or end
            if '<' in after_add:
                between = after_add[:after_add.index('<')]
            else:
                between = after_add

        if not between or between.startswith('<'):
            return None

        original_smi = between
        swapped_smi = reactant_lib.find_closest(original_smi)

        path_prefix = path_part[:last_add_pos]
        if ends_with_bracket:
            new_path = path_prefix + '<ADD>' + swapped_smi + '<'
        else:
            # Preserve anything after the molecule (like <RXN>...)
            suffix = after_add[len(between):]
            new_path = path_prefix + '<ADD>' + swapped_smi + suffix

        return new_path, original_smi, swapped_smi

    for _ in range(max_len):
        # Prepare input
        input_tensor = torch.tensor([generated], device=device)

        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_tensor, labels=None)

        logits = outputs['logits'][:, -1, :]  # Last token logits

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[0][sorted_indices_to_remove[0]]
            logits[0, indices_to_remove] = float('-inf')

        # Sample
        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1).item()
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

        # Decode current sequence
        decoded = tokenizer.decode(generated).replace(' ', '')

        # Get just the path part (after input prefix)
        if decoded.startswith(input_prefix):
            path_part = decoded[len(input_prefix):]
        else:
            path_part = decoded

        # Check for EOS - swap ALL remaining unswapped reactants
        if next_token == eos_id:
            path_part = path_part.replace('[EOS]', '')

            current_reactants = count_reactants(path_part)
            add_positions = [i for i in range(len(path_part)) if path_part[i:].startswith('<ADD>')]

            # Swap all unswapped reactants
            while num_reactants_swapped < len(add_positions):
                reactant_start = add_positions[num_reactants_swapped] + 5
                remaining = path_part[reactant_start:]

                # Find end: next '<' or end of string
                if '<' in remaining:
                    reactant_end = reactant_start + remaining.index('<')
                else:
                    reactant_end = len(path_part)

                original_smi = path_part[reactant_start:reactant_end]

                if original_smi and not original_smi.startswith('<'):
                    swapped_smi = reactant_lib.find_closest(original_smi)
                    swaps.append((original_smi, swapped_smi))

                    # Rebuild path
                    path_part = path_part[:reactant_start] + swapped_smi + path_part[reactant_end:]
                    # Recalculate add_positions since path changed
                    add_positions = [i for i in range(len(path_part)) if path_part[i:].startswith('<ADD>')]

                num_reactants_swapped += 1

            new_decoded = input_prefix + path_part + '[EOS]'
            generated = tokenizer.encode(new_decoded, add_special_tokens=False)
            break

        # Check if we just generated a '<' that starts a new tag after a reactant
        if path_part.endswith('<'):
            current_reactants = count_reactants(path_part)

            # Swap ALL unswapped reactants (there may be multiple if model generated fast)
            while current_reactants > num_reactants_swapped:
                # Recalculate positions each time since path_part changes after each swap
                add_positions = [i for i in range(len(path_part)) if path_part[i:].startswith('<ADD>')]

                if num_reactants_swapped >= len(add_positions):
                    break

                # Get the reactant at index num_reactants_swapped
                reactant_start = add_positions[num_reactants_swapped] + 5  # after '<ADD>'

                # Find end: next '<' after this position
                remaining = path_part[reactant_start:]
                if '<' in remaining:
                    reactant_end = reactant_start + remaining.index('<')
                    original_smi = path_part[reactant_start:reactant_end]
                else:
                    break  # No closing <, wait for more tokens

                if original_smi and not original_smi.startswith('<'):
                    swapped_smi = reactant_lib.find_closest(original_smi)
                    swaps.append((original_smi, swapped_smi))

                    # Rebuild path with this reactant swapped
                    path_part = path_part[:reactant_start] + swapped_smi + path_part[reactant_end:]
                    # Update count since path changed
                    current_reactants = count_reactants(path_part)

                num_reactants_swapped += 1

            # Re-tokenize full sequence with all swaps
            new_decoded = input_prefix + path_part
            generated = tokenizer.encode(new_decoded, add_special_tokens=False)

    return generated, swaps


def generate_with_shots(model, tokenizer, reactions, prop_tokens_list, args, device, reactant_lib=None):
    """Generate molecules with multiple shots and return best valid one per input."""
    bos_id = tokenizer.convert_tokens_to_ids('[BOS]')
    eos_id = tokenizer.convert_tokens_to_ids('[EOS]')

    num_inputs = len(prop_tokens_list)
    use_inline_swapping = reactant_lib is not None

    print(f"Generating {num_inputs} inputs x {args.shots} shots each...")
    if use_inline_swapping:
        print("Using inline reactant swapping (stop, swap, continue)")
    results = [None] * num_inputs

    for idx, prop_tokens in enumerate(tqdm(prop_tokens_list, desc="Inputs")):
        target_props = set(prop_tokens)
        char_ids = [tokenizer.convert_tokens_to_ids(t) for t in prop_tokens]
        tokens = char_ids + [bos_id]

        # Find best valid molecule from shots
        best_smiles = None
        best_path = None
        best_swaps = None
        best_score = -1

        for shot in range(args.shots):
            try:
                if use_inline_swapping:
                    # Use inline swapping generation
                    seq, swaps = generate_with_reactant_swapping(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=tokens,
                        reactant_lib=reactant_lib,
                        max_len=args.max_len,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        device=device,
                    )
                else:
                    # Use regular batched generation for this single shot
                    batch_inputs = [torch.tensor(tokens, device=device)]
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        batch_generated = model.generate(
                            input_ids_list=batch_inputs,
                            conditioning_embeddings=None,
                            max_generated_tokens=args.max_len,
                            stop_token_id=eos_id,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                    seq = batch_generated[0]
                    swaps = []

                seq_list = seq if isinstance(seq, list) else seq.tolist()
                decoded = tokenizer.decode(seq_list).replace(' ', '')

                # Clean
                for tok in ['[BOS]', '[EOS]', '[PAD]'] + property_tokens:
                    decoded = decoded.replace(tok, '')
                decoded = decoded.strip()

                if not decoded.startswith('<ADD>'):
                    continue

                result, _ = execute_path(decoded, reactions)
                if result is None:
                    continue

                mol = Chem.MolFromSmiles(result)
                if mol is None:
                    continue

                canonical = Chem.MolToSmiles(mol, canonical=True)

                # Score by property match
                try:
                    actual_props = set(calculate_properties(canonical)['quantized_properties'])
                    score = len(target_props & actual_props)
                except Exception:
                    score = 0

                if score > best_score:
                    best_score = score
                    best_smiles = canonical
                    best_path = decoded
                    best_swaps = swaps

            except Exception:
                continue

        if best_smiles is not None:
            # Calculate actual props for display
            try:
                actual_props = set(calculate_properties(best_smiles)['quantized_properties'])
                matched = len(target_props & actual_props)
                accuracy = matched / len(target_props) * 100 if target_props else 0
            except Exception:
                matched = 0
                accuracy = 0

            # Check what % of components are in reactant library
            in_library_pct = 0
            if reactant_lib is not None:
                # Extract all reactants from path
                parts = re.split(r'(<ADD>|<RXN>)', best_path)
                parts = [p for p in parts if p]
                reactants = []
                for i, p in enumerate(parts):
                    if p == '<ADD>' and i + 1 < len(parts):
                        reactants.append(parts[i + 1])

                # Check each reactant
                in_library = 0
                for smi in reactants:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                        sims = DataStructs.BulkTanimotoSimilarity(fp, reactant_lib.fps)
                        if max(sims) == 1.0:
                            in_library += 1
                in_library_pct = in_library / len(reactants) * 100 if reactants else 0

            results[idx] = (best_smiles, best_path, best_swaps)
            print(f"\n[Input {idx}] Target props: {prop_tokens}")
            print(f"[Input {idx}] Best path: {best_path}")
            if best_swaps:
                print(f"[Input {idx}] Swaps made:")
                for orig, swapped in best_swaps:
                    print(f"    {orig} -> {swapped}")
            print(f"[Input {idx}] Result SMILES: {best_smiles}")
            try:
                result_props = calculate_properties(best_smiles)['quantized_properties']
            except Exception:
                result_props = []
            print(f"[Input {idx}] Result SMILES Properties: {result_props}")
            print(f"[Input {idx}] Property Accuracy: {matched}/{len(target_props)} ({accuracy:.0f}%)")
            if reactant_lib is not None:
                print(f"[Input {idx}] Components in Reactant List: {in_library_pct:.0f}%")

    return results


def main():
    args = parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(f'cuda:{args.cuda}')
    print(f"Using device: {device}")
    
    # Load tokenizer
    from transformers import PreTrainedTokenizerFast
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

    # Load ZINC molecules and get property distributions
    print("Loading ZINC molecules...")
    zinc_path = "s3://shvaibackups/unibio_data/zinc22/shuffled/0000.parquet"
    df = pd.read_parquet(zinc_path, columns=['smiles'])
    zinc_smiles = df['smiles'].sample(n=args.num_mols, random_state=args.seed).tolist()

    # Calculate properties for each molecule
    print("Calculating target properties...")
    target_properties = []
    for smi in tqdm(zinc_smiles, desc="Getting properties"):
        try:
            props = calculate_properties(smi)
            target_properties.append(props['quantized_properties'])
        except Exception:
            target_properties.append([])

    # Generate molecules
    results = generate_with_shots(model, tokenizer, reactions, target_properties, args, device, reactant_lib)

    # Calculate accuracies
    category_stats = {cat: {'correct': 0, 'total': 0} for cat in PROPERTY_CATEGORIES}
    overall_correct = 0
    overall_total = 0
    valid_count = 0

    print("\nCalculating accuracies...")
    for idx, (target_props, result) in enumerate(tqdm(zip(target_properties, results),
                                                       total=len(results), desc="Checking")):
        if result is None:
            # No valid generation - count all target props as misses
            for cat, cat_props in PROPERTY_CATEGORIES.items():
                for prop in target_props:
                    if prop in cat_props:
                        category_stats[cat]['total'] += 1
                        overall_total += 1
            continue

        valid_count += 1
        smiles = result[0]

        try:
            actual_props = calculate_properties(smiles)['quantized_properties']
        except Exception:
            actual_props = []

        # Check each target property
        for prop in target_props:
            # Find which category this prop belongs to
            for cat, cat_props in PROPERTY_CATEGORIES.items():
                if prop in cat_props:
                    category_stats[cat]['total'] += 1
                    overall_total += 1
                    if prop in actual_props:
                        category_stats[cat]['correct'] += 1
                        overall_correct += 1
                    break

    # Print results
    validity_rate = valid_count / args.num_mols * 100

    print("\n" + "=" * 60)
    print(f"RESULTS (shots={args.shots})")
    print("=" * 60)
    print(f"\nVALIDITY RATE: {valid_count}/{args.num_mols} ({validity_rate:.1f}%)\n")

    for cat in ['LOG1P', 'FLEX', 'FLAT', 'SYNTH', 'MEMBRANE', 'DRUGLIKE']:
        stats = category_stats[cat]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"{cat} ACC (@{args.shots}): {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        else:
            print(f"{cat} ACC (@{args.shots}): N/A (no samples)")

    print()
    if overall_total > 0:
        overall_acc = overall_correct / overall_total * 100
        print(f"OVERALL ACC (@{args.shots}): {overall_correct}/{overall_total} ({overall_acc:.1f}%)")
    else:
        print(f"OVERALL ACC (@{args.shots}): N/A")

    print("=" * 60)


if __name__ == "__main__":
    main()

