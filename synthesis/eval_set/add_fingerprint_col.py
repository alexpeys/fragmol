#!/usr/bin/env python
"""Add fingerprint column to a parquet file and drop invalid molecules."""
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')


def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(fp, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Add fingerprint column to parquet")
    parser.add_argument("parquet_path", type=str, help="Path to parquet file")
    args = parser.parse_args()

    print(f"Loading {args.parquet_path}...")
    df = pd.read_parquet(args.parquet_path)
    print(f"Loaded {len(df)} rows")

    print("Computing fingerprints...")
    tqdm.pandas(desc="Fingerprints")
    df['fingerprint'] = df['smiles'].progress_apply(get_fingerprint)

    # Drop invalid molecules
    before = len(df)
    df = df.dropna(subset=['fingerprint'])
    after = len(df)
    print(f"Dropped {before - after} invalid molecules, {after} remaining")

    # Save in place
    print(f"Saving to {args.parquet_path}...")
    df.to_parquet(args.parquet_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()

