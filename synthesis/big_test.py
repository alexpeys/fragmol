"""Run generate_paths on 1000 molecules and report stats."""
import re
import pandas as pd
from rdkit import Chem
from helpers import get_reactions, generate_paths, execute_path, get_heavy_atom_count
from collections import defaultdict

N_SAMPLES = 1000
MAX_COMPONENT_SIZE = 12
NUM_PATHS = 20


def main():
    print("Loading data from S3...")
    df = pd.read_parquet("s3://shvaibackups/unibio_data/zinc22/shuffled/0000.parquet",
                         columns=['smiles']).head(N_SAMPLES)
    print(f"Loaded {len(df)} molecules")

    print("Loading reactions...")
    reactions = get_reactions()
    print(f"Loaded {len(reactions['retro'])} retro, {len(reactions['syn'])} syn reactions")

    # Stats
    total_mols = 0
    mols_with_paths = 0
    mols_with_working_paths = 0
    total_paths = 0
    working_paths = 0
    path_lengths = []
    max_sizes = []
    rxn_usage = defaultdict(int)

    for i, row in df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        total_mols += 1
        paths = generate_paths(smiles, reactions, max_component_size=MAX_COMPONENT_SIZE,
                               num_paths=NUM_PATHS, seed=i)

        if paths:
            mols_with_paths += 1
            has_working = False

            for path in paths:
                total_paths += 1
                result, annotated = execute_path(path, reactions)

                # Count reactions used
                rxns = re.findall(r'<RXN>([^<]+)', path)
                for rxn in rxns:
                    rxn_usage[rxn] += 1

                if result:
                    working_paths += 1
                    has_working = True
                    path_lengths.append(len(rxns))

                    adds = re.findall(r'<ADD>([^<]+)', path)
                    sizes = [get_heavy_atom_count(a) for a in adds]
                    max_sizes.append(max(sizes) if sizes else 0)

            if has_working:
                mols_with_working_paths += 1

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{N_SAMPLES}...")

    # Report
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nMolecules:")
    print(f"  Total tested:         {total_mols}")
    print(f"  With any path:        {mols_with_paths} ({100*mols_with_paths/total_mols:.1f}%)")
    print(f"  With working path:    {mols_with_working_paths} ({100*mols_with_working_paths/total_mols:.1f}%)")

    print(f"\nPaths:")
    print(f"  Total generated:      {total_paths}")
    print(f"  Working (execute):    {working_paths} ({100*working_paths/total_paths:.1f}%)")

    if path_lengths:
        print(f"\nPath lengths (working paths only):")
        print(f"  Min:    {min(path_lengths)}")
        print(f"  Max:    {max(path_lengths)}")
        print(f"  Mean:   {sum(path_lengths)/len(path_lengths):.2f}")
        length_dist = defaultdict(int)
        for l in path_lengths:
            length_dist[l] += 1
        print(f"  Distribution: {dict(sorted(length_dist.items()))}")

    if max_sizes:
        print(f"\nMax component sizes (working paths only):")
        print(f"  Min:    {min(max_sizes)}")
        print(f"  Max:    {max(max_sizes)}")
        print(f"  Mean:   {sum(max_sizes)/len(max_sizes):.2f}")
        size_buckets = {'≤10': 0, '11-12': 0, '13-15': 0, '>15': 0}
        for s in max_sizes:
            if s <= 10:
                size_buckets['≤10'] += 1
            elif s <= 12:
                size_buckets['11-12'] += 1
            elif s <= 15:
                size_buckets['13-15'] += 1
            else:
                size_buckets['>15'] += 1
        print(f"  Distribution: {size_buckets}")

    print(f"\nTop 10 reactions used:")
    for rxn, count in sorted(rxn_usage.items(), key=lambda x: -x[1])[:10]:
        print(f"  {rxn}: {count}")


if __name__ == "__main__":
    main()

