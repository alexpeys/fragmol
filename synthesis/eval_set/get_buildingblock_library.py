#!/usr/bin/env python
"""Generate a reactant library from ZINC molecules by running synthesis paths."""
import argparse
import re
import random
import os
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

import sys
from pathlib import Path
_repo_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, _repo_root)
from synthesis.helpers import get_reactions, generate_paths, execute_path
sys.path.remove(_repo_root)

# Global reactions (loaded once per worker)
_reactions = None


def init_worker():
    """Initialize worker process with reactions."""
    global _reactions
    _reactions = get_reactions()


def process_molecule(args_tuple):
    """Process a single molecule and return reactants from valid paths."""
    smiles, num_paths, max_component_size, seed = args_tuple
    global _reactions

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], 0, 0

    # Generate paths
    paths = generate_paths(
        smiles, _reactions,
        max_component_size=max_component_size,
        num_paths=num_paths,
        seed=seed
    )

    if not paths:
        return [], 0, 0

    paths_generated = len(paths)
    paths_valid = 0
    reactants = []

    # Check each path
    for path in paths:
        result, _ = execute_path(path, _reactions)

        # Path must produce SOME molecule (doesn't have to be the target)
        if result is None:
            continue

        paths_valid += 1

        # Extract all <ADD> reactants (NOT products!)
        adds = re.findall(r'<ADD>([^<]+)', path)

        for reactant in adds:
            # Canonicalize
            rmol = Chem.MolFromSmiles(reactant)
            if rmol:
                canonical = Chem.MolToSmiles(rmol, canonical=True)
                reactants.append(canonical)

    return reactants, paths_generated, paths_valid


def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(fp, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Generate reactant library from molecules")
    parser.add_argument("--input_path", type=str, default="s3://shvaibackups/unibio_data/zinc22/shuffled/0000.parquet")
    parser.add_argument("--output_path", type=str, default="zinc22_0000_blocks.parquet")
    parser.add_argument("--min_occurrences", type=float, default=25)
    parser.add_argument("--num_mols", type=float, default=1e7,
                        help="Number of molecules to process (default: 1e7)")
    parser.add_argument("--num_paths", type=int, default=5,
                        help="Number of paths to generate per molecule (default: 5)")
    parser.add_argument("--max_component_size", type=int, default=12,
                        help="Max heavy atoms for components (default: 12)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count)")
    args = parser.parse_args()

    num_mols = int(args.num_mols)
    num_workers = args.workers or cpu_count()
    random.seed(args.seed)

    # Load ZINC data
    print("Loading ZINC data from S3...")
    df = pd.read_parquet(args.input_path, columns=['smiles'])
    print(f"Loaded {len(df):,} molecules from file")

    # Sample if needed
    if num_mols < len(df):
        df = df.sample(n=num_mols, random_state=args.seed)
        print(f"Sampled {num_mols:,} molecules")
    else:
        print(f"Using all {len(df):,} molecules")

    # Prepare work items with unique seeds
    smiles_list = df['smiles'].tolist()
    work_items = [
        (smi, args.num_paths, args.max_component_size, random.randint(0, 2**31))
        for smi in smiles_list
    ]

    # Track reactants
    reactant_counts = Counter()
    total_paths_generated = 0
    total_paths_valid = 0

    # Process in parallel
    print(f"\nProcessing molecules with {num_workers} workers...")
    with Pool(num_workers, initializer=init_worker) as pool:
        for reactants, paths_gen, paths_val in tqdm(
            pool.imap_unordered(process_molecule, work_items, chunksize=100),
            total=len(work_items),
            desc="Molecules"
        ):
            reactant_counts.update(reactants)
            total_paths_generated += paths_gen
            total_paths_valid += paths_val

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Molecules processed: {len(df):,}")
    print(f"  Paths generated: {total_paths_generated:,}")
    print(f"  Valid paths: {total_paths_valid:,}")
    print(f"  Unique reactants: {len(reactant_counts):,}")
    print(f"  Total reactant occurrences: {sum(reactant_counts.values()):,}")
    print(f"{'='*60}")

    # Save to parquet (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "sample_reactants.parquet")
    result_df = pd.DataFrame([
        {'smiles': smi, 'num_occurrences': count}
        for smi, count in reactant_counts.items()
    ])
    
    tqdm.pandas(desc="Fingerprints")
    
    result_df = result_df.sort_values('num_occurrences', ascending=False)
    result_df = result_df[result_df['num_occurrences'] >= args.min_occurrences]
    result_df['fingerprint'] = result_df['smiles'].progress_apply(get_fingerprint)

    result_df.to_parquet(args.output_path, index=False)
    print(f"\nSaved {len(result_df):,} reactants to {args.output_path}")

if __name__ == "__main__":
    main()

