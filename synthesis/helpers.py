"""
Helper functions for molecular synthesis path generation and execution.
"""
import re
import json
import random
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

DEFAULT_REACTIONS_PATH = Path(__file__).parent / "reactions.json"


def get_reactions(path: str = None) -> dict:
    """Load and parse reactions from JSON. Returns dict with 'retro' and 'syn' keys."""
    if path is None:
        path = DEFAULT_REACTIONS_PATH

    with open(path, 'r') as f:
        content = f.read()

    # Fix Python booleans/None to JSON
    content = content.replace(': True', ': true')
    content = content.replace(': False', ': false')
    content = content.replace(': None', ': null')
    content = re.sub(r',(\s*[}\]])', r'\1', content)

    raw = json.loads(content)

    # Parse retro reactions
    retro = []
    for name, data in raw.items():
        smarts = data.get("retro_smarts") if isinstance(data, dict) else data
        if smarts:
            try:
                rxn = AllChem.ReactionFromSmarts(smarts)
                if rxn:
                    retro.append({'name': name, 'rxn': rxn})
            except:
                pass

    # Parse syn reactions
    syn = {}
    for name, data in raw.items():
        smarts = data.get("syn_smarts") if isinstance(data, dict) else None
        if smarts:
            try:
                rxn = AllChem.ReactionFromSmarts(smarts)
                if rxn:
                    syn[name] = rxn
            except:
                pass

    return {'retro': retro, 'syn': syn}


def get_heavy_atom_count(smi: str) -> int:
    """Get number of heavy atoms (non-hydrogen) in a SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return mol.GetNumHeavyAtoms()
    return 0


def _build_path_string(steps: list[dict]) -> str:
    """Build path string from list of retro steps.

    Format: <ADD>smi<ADD>smi<RXN>name<ADD>smi<RXN>name...
    """
    if not steps:
        return ""

    path_parts = []
    reversed_steps = list(reversed(steps))
    last_result = None

    for step in reversed_steps:
        # Add precursors, skip first if it matches last result (intermediate from prev step)
        for j, precursor in enumerate(step['products']):
            if j == 0 and last_result and precursor == last_result:
                continue
            path_parts.append(f"<ADD>{precursor}")

        path_parts.append(f"<RXN>{step['reaction']}")
        last_result = step['substrate']

    return "".join(path_parts)


def generate_paths(
    smiles: str,
    reactions: dict,
    max_component_size: int = 12,
    num_paths: int = 5,
    seed: int | None = None,
    max_iterations: int = 20
) -> list[str]:
    """
    Generate retrosynthetic paths for a target molecule.

    Args:
        smiles: SMILES string of target molecule
        reactions: dict from get_reactions() with 'retro' and 'syn' keys
        max_component_size: keep decomposing until fragments are <= this size
        num_paths: number of paths to generate
        seed: random seed for reproducibility
        max_iterations: safety limit on decomposition iterations

    Returns:
        list of recipe strings in format: <ADD>smi<ADD>smi<RXN>name...
    """
    if seed is not None:
        random.seed(seed)

    target_mol = Chem.MolFromSmiles(smiles)
    if not target_mol:
        return []
    target_canonical = Chem.MolToSmiles(target_mol, canonical=True)

    retro_rxns = reactions['retro']
    if not retro_rxns:
        return []

    def _do_one_retro_step(current_smi: str, require_size_reduction: bool = False,
                           seen_smiles: set = None) -> dict | None:
        """Try to apply one retro reaction, return step dict or None."""
        current_mol = Chem.MolFromSmiles(current_smi)
        if not current_mol:
            return None

        current_size = get_heavy_atom_count(current_smi)
        random.shuffle(retro_rxns)

        for rxn_data in retro_rxns:
            try:
                products_list = rxn_data['rxn'].RunReactants((current_mol,))
                if not products_list:
                    continue

                for products in products_list:
                    result_smiles = []
                    valid = True

                    for m in products:
                        try:
                            Chem.SanitizeMol(m)
                            smi = Chem.MolToSmiles(m, canonical=True)
                            result_smiles.append(smi)
                        except:
                            valid = False
                            break

                    if valid and result_smiles:
                        result_smiles.sort(key=get_heavy_atom_count, reverse=True)
                        largest = result_smiles[0]
                        largest_size = get_heavy_atom_count(largest)

                        if require_size_reduction and largest_size >= current_size:
                            continue
                        if seen_smiles and largest in seen_smiles:
                            continue

                        return {
                            'reaction': rxn_data['name'],
                            'substrate': current_smi,
                            'products': result_smiles
                        }
            except Exception:
                pass
        return None

    paths = []

    for _ in range(num_paths):
        current_smi = target_canonical
        steps = []
        iterations = 0
        seen_smiles = {target_canonical}

        while iterations < max_iterations:
            iterations += 1

            if get_heavy_atom_count(current_smi) <= max_component_size:
                break

            step = _do_one_retro_step(current_smi, require_size_reduction=True,
                                      seen_smiles=seen_smiles)
            if not step:
                break

            largest_product = step['products'][0]
            seen_smiles.add(largest_product)
            steps.append(step)
            current_smi = largest_product

        if steps:
            paths.append(_build_path_string(steps))

    return list(dict.fromkeys(paths))


# Common implicit reagents that reactions might need
IMPLICIT_REAGENTS = [
    # Nucleophiles / protic
    "O",           # Water
    "CO",          # Methanol
    "CCO",         # Ethanol
    "CC(C)O",      # Isopropanol
    "CC(C)(C)O",   # tert-Butanol
    "N",           # Ammonia
    "CN",          # Methylamine
    "CCN",         # Ethylamine
    "CNC",         # Dimethylamine
    # Acids / bases
    "Cl",          # HCl
    "Br",          # HBr
    "[OH-]",       # Hydroxide
    "CC(=O)O",     # Acetic acid
    "C(=O)O",      # Formic acid
    # Solvents (sometimes reactive)
    "CC#N",        # Acetonitrile
    "C1CCOC1",     # THF
    "ClCCl",       # DCM
    "CCOCC",       # Diethyl ether
    "CC(C)=O",     # Acetone
    "CS(C)=O",     # DMSO
    "CN(C)C=O",    # DMF
    "[H][H]",      # H2
]


def execute_path(recipe: str, reactions: dict) -> tuple[str | None, str | None]:
    """
    Execute a synthesis recipe and return result + annotated recipe.

    Args:
        recipe: Path string in format <ADD>smi<ADD>smi<RXN>name...
        reactions: dict from get_reactions() with 'retro' and 'syn' keys

    Returns:
        tuple of (result_smiles, annotated_recipe) or (None, None) if fails.
        Annotated recipe has <PRODUCT>smi after each <RXN>:
        <ADD>x<ADD>y<RXN>z<PRODUCT>a<ADD>b<RXN>c<PRODUCT>d...
    """
    syn_rxns = reactions['syn']

    def try_reaction(rxn, mols):
        """Try running a reaction, return product mol or None."""
        try:
            if len(mols) == 1:
                products = rxn.RunReactants((mols[0],))
            elif len(mols) == 2:
                products = rxn.RunReactants((mols[0], mols[1]))
                if not products:
                    products = rxn.RunReactants((mols[1], mols[0]))
            else:
                return None

            if products and products[0]:
                new_mol = products[0][0]
                Chem.SanitizeMol(new_mol)
                return new_mol
        except:
            pass
        return None

    # Parse tokens
    tokens = re.split(r'(<ADD>|<RXN>)', recipe)
    tokens = [t for t in tokens if t]

    pending_mols = []
    current_result = None
    output_parts = []

    i = 0
    while i < len(tokens):
        if tokens[i] == '<ADD>':
            if i + 1 < len(tokens):
                smi = tokens[i + 1]
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    pending_mols.append(mol)
                output_parts.append(f"<ADD>{smi}")
                i += 2
            else:
                i += 1

        elif tokens[i] == '<RXN>':
            if i + 1 < len(tokens):
                rxn_name = tokens[i + 1]
                output_parts.append(f"<RXN>{rxn_name}")

                if rxn_name not in syn_rxns:
                    return None, None

                rxn = syn_rxns[rxn_name]
                n_reactants = rxn.GetNumReactantTemplates()

                # Try implicit reagents if needed
                if len(pending_mols) < n_reactants:
                    for reagent_smi in IMPLICIT_REAGENTS:
                        if len(pending_mols) >= n_reactants:
                            break
                        reagent_mol = Chem.MolFromSmiles(reagent_smi)
                        if reagent_mol:
                            test_mols = pending_mols + [reagent_mol]
                            if try_reaction(rxn, test_mols[:n_reactants]):
                                pending_mols.append(reagent_mol)

                if len(pending_mols) < n_reactants:
                    return None, None

                result = try_reaction(rxn, pending_mols[:n_reactants])
                if not result:
                    return None, None

                result_smi = Chem.MolToSmiles(result, canonical=True)
                output_parts.append(f"<PRODUCT>{result_smi}")

                pending_mols = pending_mols[n_reactants:]
                current_result = result
                pending_mols.insert(0, current_result)

                i += 2
            else:
                i += 1
        else:
            i += 1

    if current_result:
        return Chem.MolToSmiles(current_result, canonical=True), "".join(output_parts)
    return None, None
