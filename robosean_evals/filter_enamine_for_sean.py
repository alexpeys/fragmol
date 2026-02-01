"""
Filter enamine_properties.parquet for high property_score molecules
and flag those that need protecting groups (reactions with multiple products).
"""
import sys
sys.path.insert(0, '/home/ubuntu/fragmol')

import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from synthesis.helpers import get_reactions

def check_needs_protecting_group(synthesis_path: str, reactions: dict) -> bool:
    """
    Check if any reaction in the path produces multiple products,
    indicating a need for protecting groups.

    Returns True if any reaction has multiple possible sites.
    """
    # Get the syn reactions dict (already compiled ChemicalReaction objects)
    syn_reactions = reactions.get('syn', {})

    # Parse the path to get reactions and their inputs
    # Format: <ADD>smi<ADD>smi<RXN>name-variant<ADD>smi<RXN>name-variant...
    tokens = re.split(r'(<ADD>|<RXN>)', synthesis_path)
    tokens = [t for t in tokens if t]

    pending_smiles = []
    i = 0
    while i < len(tokens):
        if tokens[i] == '<ADD>':
            if i + 1 < len(tokens):
                pending_smiles.append(tokens[i + 1])
            i += 2
        elif tokens[i] == '<RXN>':
            if i + 1 < len(tokens):
                rxn_tag = tokens[i + 1]
                # rxn_tag is like "amide_coupling-1" - use directly

                if rxn_tag in syn_reactions and len(pending_smiles) >= 1:
                    rxn = syn_reactions[rxn_tag]

                    # Convert pending SMILES to mols
                    mols = []
                    for smi in pending_smiles:
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            mols.append(mol)

                    if mols:
                        # Try running the reaction
                        try:
                            n_reactants = rxn.GetNumReactantTemplates()
                            if len(mols) >= n_reactants:
                                if n_reactants == 1:
                                    products = rxn.RunReactants((mols[0],))
                                elif n_reactants == 2:
                                    products = rxn.RunReactants((mols[0], mols[1]))
                                    if not products and len(mols) >= 2:
                                        products = rxn.RunReactants((mols[1], mols[0]))
                                else:
                                    products = []

                                # If more than one product set, multiple sites match
                                if len(products) > 1:
                                    return True
                        except:
                            pass

                # After reaction, clear pending (simplified - actual execution is more complex)
                pending_smiles = []
            i += 2
        else:
            i += 1

    return False


def main():
    print("Loading data...")
    df = pd.read_parquet('robosean_evals/enamine_properties.parquet')
    print(f"Total rows: {len(df)}")
    
    # Filter for property_score > 0.9
    df_high = df[df['property_score'] > 0.9].copy()
    print(f"Rows with property_score > 0.9: {len(df_high)}")
    
    # Load reactions
    reactions = get_reactions()
    
    # Check each path for protecting group needs
    print("Checking for protecting group requirements...")
    needs_pg = []
    for idx, row in df_high.iterrows():
        path = row['synthesis_path']
        if pd.isna(path) or not path:
            needs_pg.append(0)
            continue
        needs_pg.append(1 if check_needs_protecting_group(path, reactions) else 0)
    
    df_high['needs_protecting_group'] = needs_pg
    
    # Create output dataframe with recipe, product, needs_protecting_group
    # Dedupe on product (generated_smiles)
    df_out = df_high[['synthesis_path', 'generated_smiles', 'needs_protecting_group']].copy()
    df_out = df_out.rename(columns={'synthesis_path': 'recipe', 'generated_smiles': 'product'})
    
    # Drop rows with None/NaN product
    df_out = df_out.dropna(subset=['product'])
    
    # Dedupe on product
    df_out = df_out.drop_duplicates(subset=['product'], keep='first')
    
    print(f"\nOutput stats:")
    print(f"  Total unique products: {len(df_out)}")
    print(f"  Needs protecting group: {df_out['needs_protecting_group'].sum()}")
    print(f"  No protecting group needed: {(df_out['needs_protecting_group'] == 0).sum()}")
    
    # Save
    output_path = 'robosean_evals/enamine_examples_for_sean.parquet'
    df_out.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

