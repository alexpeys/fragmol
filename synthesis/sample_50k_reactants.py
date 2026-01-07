#!/usr/bin/env python
"""Sample 50k reactants and compute fingerprints."""
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load reactants
df = pd.read_parquet(os.path.join(script_dir, 'sample_reactants.parquet'))
print(f"Loaded {len(df):,} reactants")

# Filter by occurrence
df = df[df['num_occurrences'] > 50]
print(f"After filtering (>50 occurrences): {len(df):,}")

# Sample 50k
df = df.sample(n=50_000, random_state=42)
print(f"Sampled 50,000 reactants")

# Compute 1024-bit Morgan fingerprints
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(fp, dtype=np.uint8)

tqdm.pandas()
df['fingerprint'] = df['smiles'].progress_apply(get_fingerprint)
df = df.dropna(subset=['fingerprint'])
print(f"After dropping invalid: {len(df):,}")

# Save
output_path = os.path.join(script_dir, '50k_reactants.parquet')
df.to_parquet(output_path)
print(f"Saved to {output_path}")

