#!/usr/bin/env python
"""Animate a synthesis path for a given molecule."""
import argparse
import re
import sys
import time
from rdkit import Chem

from helpers import get_reactions, generate_paths, execute_path


def parse_annotated_path(annotated):
    """Parse annotated path into list of steps.
    
    Each step is: {'reactants': [smi, ...], 'reaction': name, 'product': smi}
    """
    steps = []
    tokens = re.split(r'(<ADD>|<RXN>|<PRODUCT>)', annotated)
    tokens = [t for t in tokens if t]
    
    current_reactants = []
    current_rxn = None
    
    i = 0
    while i < len(tokens):
        if tokens[i] == '<ADD>' and i + 1 < len(tokens):
            current_reactants.append(tokens[i + 1])
            i += 2
        elif tokens[i] == '<RXN>' and i + 1 < len(tokens):
            current_rxn = tokens[i + 1]
            i += 2
        elif tokens[i] == '<PRODUCT>' and i + 1 < len(tokens):
            product = tokens[i + 1]
            steps.append({
                'reactants': current_reactants,
                'reaction': current_rxn,
                'product': product
            })
            # Product becomes first reactant for next step
            current_reactants = [product]
            current_rxn = None
            i += 2
        else:
            i += 1
    
    return steps


def animate_synthesis(steps, target_smiles, delay=1.5):
    """Print animated synthesis steps."""
    print("\n" + "="*70)
    print(f"🎯 TARGET: {target_smiles}")
    print("="*70 + "\n")
    time.sleep(delay)
    
    for i, step in enumerate(steps):
        reactants = step['reactants']
        rxn = step['reaction']
        product = step['product']
        
        # Build the equation
        if i == 0:
            lhs = " + ".join(reactants)
        else:
            # First reactant is the intermediate from previous step
            lhs = f"[intermediate] + " + " + ".join(reactants[1:]) if len(reactants) > 1 else "[intermediate]"
        
        print(f"Step {i+1}: {rxn}")
        print("-" * 50)
        
        # Animate the reactants appearing
        for j, r in enumerate(reactants):
            if j > 0:
                print("        +")
            print(f"        {r}")
            time.sleep(delay * 0.3)
        
        print(f"        ─── {rxn} ───▶")
        time.sleep(delay * 0.5)
        
        print(f"        {product}")
        print()
        time.sleep(delay)
    
    print("="*70)
    print(f"✅ FINAL PRODUCT: {steps[-1]['product']}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Animate synthesis path for a molecule')
    parser.add_argument('--smiles', required=True, help='Target molecule SMILES')
    parser.add_argument('--delay', type=float, default=1.0, help='Animation delay in seconds')
    parser.add_argument('--max-size', type=int, default=12, help='Max component size')
    parser.add_argument('--num-paths', type=int, default=10, help='Number of paths to try')
    args = parser.parse_args()
    
    # Validate input
    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        print(f"Error: Invalid SMILES: {args.smiles}", file=sys.stderr)
        sys.exit(1)
    
    target = Chem.MolToSmiles(mol, canonical=True)
    print(f"Loading reactions...")
    reactions = get_reactions()
    
    print(f"Generating paths for: {target}")
    paths = generate_paths(args.smiles, reactions, 
                          max_component_size=args.max_size,
                          num_paths=args.num_paths)
    
    if not paths:
        print("No paths generated. Molecule may be too small or no reactions apply.")
        sys.exit(1)
    
    # Find a path that produces the target
    print(f"Generated {len(paths)} paths, finding one that works...")
    
    for path in paths:
        result, annotated = execute_path(path, reactions)
        if result:
            result_mol = Chem.MolFromSmiles(result)
            if result_mol:
                result_canonical = Chem.MolToSmiles(result_mol, canonical=True)
                if result_canonical == target:
                    steps = parse_annotated_path(annotated)
                    animate_synthesis(steps, target, delay=args.delay)
                    return
    
    # If no exact match, just use the first working path
    for path in paths:
        result, annotated = execute_path(path, reactions)
        if result and annotated:
            print(f"(No exact match found, showing closest path)")
            steps = parse_annotated_path(annotated)
            animate_synthesis(steps, target, delay=args.delay)
            return
    
    print("No working paths found.")
    sys.exit(1)


if __name__ == "__main__":
    main()

