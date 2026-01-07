"""Interactive test script for synthesis path generation."""
import re
import random
import pandas as pd
from rdkit import Chem
from helpers import get_reactions, generate_paths, execute_path, get_heavy_atom_count

MAX_COMPONENT_SIZE = 12


def main():
    print("Loading data from S3...")
    df = pd.read_parquet("s3://shvaibackups/unibio_data/zinc22/shuffled/0000.parquet")
    all_smiles = df['smiles'].tolist()
    print(f"Loaded {len(all_smiles):,} molecules")

    print("Loading reactions...")
    reactions = get_reactions()
    print(f"Loaded {len(reactions['retro'])} retro, {len(reactions['syn'])} syn reactions")

    print("\n" + "="*70)
    print("Interactive Synthesis Path Tester")
    print("Press ENTER to test another molecule, 'q' to quit")
    print("="*70 + "\n")

    while True:
        smiles = random.choice(all_smiles)
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True)
        n_heavy = mol.GetNumHeavyAtoms()

        print(f"\n{'='*70}")
        print(f"TARGET: {canonical} ({n_heavy} heavy atoms)")
        print(f"{'='*70}\n")

        paths = generate_paths(smiles, reactions, max_component_size=MAX_COMPONENT_SIZE, num_paths=10)

        if not paths:
            print("  No paths generated.")
        else:
            print(f"Generated {len(paths)} unique paths:\n")

            for i, path in enumerate(paths):
                result, annotated = execute_path(path, reactions)
                adds = re.findall(r'<ADD>([^<]+)', path)
                sizes = [get_heavy_atom_count(a) for a in adds]
                n_steps = path.count('<RXN>')

                status = "✓" if result else "✗"
                print(f"Path {i+1}: {status} ({n_steps} steps, max_size={max(sizes)})")
                print(f"  Input:  {path}")
                if annotated:
                    print(f"  Output: {annotated}")
                print()

        user_input = input("Press ENTER for next molecule (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break

    print("Done!")


if __name__ == "__main__":
    main()
