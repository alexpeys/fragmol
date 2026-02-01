"""
Helper functions for molecular synthesis path generation and execution.
"""
import re
import json
import random
from io import BytesIO
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


def parse_annotated_path(annotated: str) -> list[dict]:
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


def _load_raw_reactions(path: str = None) -> dict:
    """Load raw reactions JSON (for display_smarts etc)."""
    if path is None:
        path = DEFAULT_REACTIONS_PATH
    with open(path, 'r') as f:
        content = f.read()
    content = content.replace(': True', ': true')
    content = content.replace(': False', ': false')
    content = content.replace(': None', ': null')
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    return json.loads(content)


def create_reaction_diagram(
    synthesis_path: str,
    reactions: dict | None = None,
    mol_size: tuple[int, int] = (150, 150),
    mols_per_row: int | None = None,
) -> "PIL.Image.Image":
    """
    Create a reaction diagram image from a synthesis path string.

    Args:
        synthesis_path: Path string in format <ADD>smi<ADD>smi<RXN>name...
        reactions: dict from get_reactions(). If None, loads default reactions.
        mol_size: Size of each molecule image (width, height).
        mols_per_row: Number of molecules per row. None = all in one row.

    Returns:
        PIL Image with the reaction diagram.
        Format: ADD ADD rxn PRODUCT ADD rxn PRODUCT ...

    Raises:
        ValueError: If the path cannot be parsed or executed.
    """
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    if reactions is None:
        reactions = get_reactions()

    # Load raw reactions for display_smarts
    raw_reactions = _load_raw_reactions()

    # Execute the path to get annotated version with products
    result, annotated = execute_path(synthesis_path, reactions)

    if result is None or annotated is None:
        raise ValueError(f"Failed to execute synthesis path: {synthesis_path}")

    # Parse into steps
    steps = parse_annotated_path(annotated)

    if not steps:
        raise ValueError(f"No valid steps parsed from path: {synthesis_path}")

    # Build one flat list of mols and legends
    all_mols = []
    legends = []
    prev_product_smi = None

    for i, step in enumerate(steps):
        rxn_name = step.get('reaction', '?')
        product_mol = Chem.MolFromSmiles(step['product'])

        if product_mol is None:
            continue

        # Add reactants (skip first if it's the previous product)
        for j, r_smi in enumerate(step['reactants']):
            if j == 0 and i > 0 and r_smi == prev_product_smi:
                continue  # Skip - already shown as previous PRODUCT
            mol = Chem.MolFromSmiles(r_smi)
            if mol:
                all_mols.append(mol)
                legends.append("ADD")

        # Add reaction template from display_smarts
        rxn_data = raw_reactions.get(rxn_name, {})
        display_smarts = rxn_data.get('ld_data', {}).get('display_smarts', '')

        if display_smarts and '>>' in display_smarts:
            rxn_parts = display_smarts.split('>>')
            reactant_smarts = rxn_parts[0].split('.')
            for smarts in reactant_smarts[:1]:
                try:
                    tmpl_mol = Chem.MolFromSmarts(smarts)
                    if tmpl_mol:
                        all_mols.append(tmpl_mol)
                        legends.append(rxn_name)
                        break
                except:
                    pass

        # Add product
        all_mols.append(product_mol)
        legends.append("PRODUCT")
        prev_product_smi = step['product']

    if not all_mols:
        raise ValueError(f"Could not create diagram from path: {synthesis_path}")

    # Default: all in one row
    if mols_per_row is None:
        mols_per_row = len(all_mols)

    # Draw each molecule individually and crop whitespace
    from PIL import Image, ImageDraw, ImageOps

    draw_opts = rdMolDraw2D.MolDrawOptions()
    draw_opts.padding = 0.01
    draw_opts.bondLineWidth = 1.5

    mol_images = []
    for mol in all_mols:
        img = Draw.MolToImage(mol, size=mol_size, options=draw_opts)
        # Crop whitespace
        bg = Image.new(img.mode, img.size, 'white')
        diff = ImageOps.invert(img.convert('L'))
        bbox = diff.getbbox()
        if bbox:
            # Add small margin
            margin = 5
            bbox = (max(0, bbox[0]-margin), max(0, bbox[1]-margin),
                    min(img.width, bbox[2]+margin), min(img.height, bbox[3]+margin))
            img = img.crop(bbox)
        mol_images.append(img)

    # Find max height for uniform row height
    max_h = max(img.height for img in mol_images)
    legend_height = 28
    spacing = 17

    # Calculate total width
    total_width = sum(img.width for img in mol_images) + spacing * (len(mol_images) - 1)
    total_height = max_h + legend_height

    # Composite
    combined = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(combined)

    x = 0
    for img, legend in zip(mol_images, legends):
        # Center vertically
        y = (max_h - img.height) // 2
        combined.paste(img, (x, y))
        # Draw legend centered below with larger font
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), legend, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = x + (img.width - text_w) // 2
        draw.text((text_x, max_h + 2), legend, fill='black', font=font)
        x += img.width + spacing

    return combined


def sample_and_visualize(
    parquet_path: str = None,
    max_component_size: int = 12,
    num_paths: int = 5,
) -> "PIL.Image.Image":
    """
    Load zinc1000 examples, pick one at random, generate a retrosynthesis path, and display it.

    Args:
        parquet_path: Path to parquet file with SMILES. Defaults to zinc22_1000_evals.parquet.
        max_component_size: Max fragment size for retrosynthesis.
        num_paths: Number of paths to try generating.

    Returns:
        PIL Image of the reaction diagram.
    """
    import pandas as pd

    if parquet_path is None:
        parquet_path = "/home/ubuntu/fragmol/synthesis/eval_set/zinc22_1000_evals.parquet"

    df = pd.read_parquet(parquet_path)

    # Pick a random SMILES
    row = df.sample(n=1).iloc[0]
    smiles = row.get('smiles') or row.get('SMILES') or row.iloc[0]

    print(f"Target: {smiles}")

    reactions = get_reactions()

    # Generate paths
    paths = generate_paths(
        smiles, reactions,
        max_component_size=max_component_size,
        num_paths=num_paths,
        seed=random.randint(0, 2**31)
    )

    if not paths:
        raise ValueError(f"No paths generated for {smiles}")

    # Try each path until one works
    for path in paths:
        result, annotated = execute_path(path, reactions)
        if result is not None:
            print(f"Path: {path}")
            return create_reaction_diagram(path, reactions)

    raise ValueError(f"No valid paths found for {smiles}")


def plot_3d(
    smiles: str,
    optimize: bool = True,
    figsize: tuple[int, int] = (8, 8),
    bg_color: str = 'black',
    elev: float = 20,
    azim: float = 45,
) -> "plt.Figure":
    """
    Create a 3D ball-and-stick plot of a molecule.

    Args:
        smiles: SMILES string
        optimize: If True, run MMFF force field optimization
        figsize: Figure size (width, height)
        bg_color: Background color ('black' or 'white')
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from rdkit.Chem import AllChem
    import numpy as np

    # Atom colors (optimized for black background)
    ATOM_COLORS = {
        'C': '#CCCCCC',  # Light gray
        'N': '#8FB4FF',  # Light blue
        'O': '#FF6B6B',  # Bright red
        'S': '#FFEB3B',  # Bright yellow
        'P': '#FFA726',  # Bright orange
        'F': '#B2FF59',  # Bright green
        'Cl': '#69F0AE', # Bright teal
        'Br': '#FF8A65', # Bright coral
        'I': '#CE93D8',  # Bright purple
        'H': '#FFFFFF',  # White
    }

    ATOM_SIZES = {
        'H': 300, 'C': 600, 'N': 600, 'O': 600,
        'S': 750, 'P': 750, 'F': 540, 'Cl': 660, 'Br': 720, 'I': 780,
    }

    # Generate 3D structure
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            pass

    # Get coordinates
    conf = mol.GetConformer()
    coords = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])

    # Center coordinates
    coords -= coords.mean(axis=0)

    # Create figure
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)

    # Set limits with margin
    margin = 0.5
    ax.set_xlim([coords[:, 0].min() - margin, coords[:, 0].max() + margin])
    ax.set_ylim([coords[:, 1].min() - margin, coords[:, 1].max() + margin])
    ax.set_zlim([coords[:, 2].min() - margin, coords[:, 2].max() + margin])

    # Light gray 3D cube lines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    grid_color = '#444444' if bg_color == 'black' else '#cccccc'
    ax.xaxis.pane.set_edgecolor(grid_color)
    ax.yaxis.pane.set_edgecolor(grid_color)
    ax.zaxis.pane.set_edgecolor(grid_color)
    ax.xaxis._axinfo['grid']['color'] = grid_color
    ax.yaxis._axinfo['grid']['color'] = grid_color
    ax.zaxis._axinfo['grid']['color'] = grid_color

    bond_color = 'white' if bg_color == 'black' else '#333333'
    edge_color = 'white' if bg_color == 'black' else 'black'

    # Draw bonds
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ax.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                [coords[i, 2], coords[j, 2]],
                color=bond_color, linewidth=4, alpha=0.7, zorder=1)

    # Draw atoms as shaded spheres
    ATOM_RADII = {
        'H': 0.25, 'C': 0.4, 'N': 0.38, 'O': 0.35,
        'S': 0.5, 'P': 0.5, 'F': 0.3, 'Cl': 0.45, 'Br': 0.5, 'I': 0.55,
    }

    # Create sphere mesh
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        color = ATOM_COLORS.get(symbol, '#FF1493')
        radius = ATOM_RADII.get(symbol, 0.4)

        # Scale and translate sphere
        x = coords[i, 0] + radius * sphere_x
        y = coords[i, 1] + radius * sphere_y
        z = coords[i, 2] + radius * sphere_z

        # Convert hex color to RGB
        hex_color = color.lstrip('#')
        rgb = tuple(int(hex_color[j:j+2], 16) / 255 for j in (0, 2, 4))

        # Create shaded colors array (simple lighting from top-right)
        light_dir = np.array([0.5, 0.5, 1.0])
        light_dir /= np.linalg.norm(light_dir)

        normals_x = sphere_x
        normals_y = sphere_y
        normals_z = sphere_z

        shading = (normals_x * light_dir[0] + normals_y * light_dir[1] + normals_z * light_dir[2])
        shading = np.clip(shading, 0.3, 1.0)  # Ambient + diffuse

        colors = np.zeros((*shading.shape, 4))
        colors[..., 0] = rgb[0] * shading
        colors[..., 1] = rgb[1] * shading
        colors[..., 2] = rgb[2] * shading
        colors[..., 3] = 1.0

        ax.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=True, shade=False)

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    return fig
