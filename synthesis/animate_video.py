#!/usr/bin/env python
"""Generate MP4 animation of a synthesis path."""
import argparse
import re
import sys
import tempfile
import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont

from helpers import get_reactions, generate_paths, execute_path


def parse_annotated_path(annotated):
    """Parse annotated path into list of steps."""
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
            current_reactants = [product]
            current_rxn = None
            i += 2
        else:
            i += 1
    return steps


def get_font(size, bold=False):
    """Get font with fallback."""
    try:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except:
        return ImageFont.load_default()


def mol_to_image(smiles, size=(200, 200)):
    """Convert SMILES to PIL Image with transparent background."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Image.new('RGBA', size, (255, 255, 255, 0))
    img = Draw.MolToImage(mol, size=size, fitImage=True)
    return img.convert('RGBA')


def create_step_frame(step_num, total_steps, reactants, reaction, product, width=1280, height=720):
    """Create frame with vertical reactants -> reaction -> product layout."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    n_reactants = len(reactants)

    # Layout: LEFT side has reactants stacked, RIGHT side has product
    # Middle has arrow + reaction name

    left_x = width // 4
    right_x = 3 * width // 4
    arrow_x = width // 2

    # Size molecules to fit vertically (reactants) and use similar size for product
    margin_y = 60
    plus_height = 30
    usable_height = height - 2 * margin_y

    # Reactants need: n * mol_height + (n-1) * plus_height
    mol_height = min((usable_height - (n_reactants - 1) * plus_height) // max(n_reactants, 1), 280)
    mol_size = (mol_height, mol_height)

    # Calculate total reactants column height
    reactants_height = n_reactants * mol_height + (n_reactants - 1) * plus_height
    start_y = (height - reactants_height) // 2

    # Draw reactants in a column
    y = start_y
    plus_font = get_font(28, bold=True)

    for i, smi in enumerate(reactants):
        mol_img = mol_to_image(smi, mol_size)
        img.paste(mol_img, (left_x - mol_size[0]//2, y), mol_img)
        y += mol_height

        if i < n_reactants - 1:
            draw.text((left_x, y + plus_height//2), "+", fill='#333', font=plus_font, anchor='mm')
            y += plus_height

    # Arrow and reaction name in the middle
    arrow_font = get_font(36, bold=True)
    rxn_font = get_font(22)
    center_y = height // 2

    draw.text((arrow_x, center_y - 15), "―――→", fill='#333', font=arrow_font, anchor='mm')
    draw.text((arrow_x, center_y + 25), reaction, fill='#0066cc', font=rxn_font, anchor='mm')

    # Product on the right, vertically centered
    product_size = (min(mol_height + 60, 340), min(mol_height + 60, 340))
    product_img = mol_to_image(product, product_size)
    img.paste(product_img, (right_x - product_size[0]//2, center_y - product_size[1]//2), product_img)

    # Step label at top
    step_font = get_font(22, bold=True)
    draw.text((width//2, 15), f"Step {step_num}/{total_steps}", fill='#888', font=step_font, anchor='mt')

    return img


def create_final_frame(product_smiles, width=1280, height=720):
    """Create final frame with just the molecule."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    title_font = get_font(32, bold=True)
    draw.text((width//2, 40), "Final Molecule", fill='#333', font=title_font, anchor='mt')

    mol_size = min(width - 100, height - 120)
    mol_img = mol_to_image(product_smiles, (mol_size, mol_size))
    img.paste(mol_img, ((width - mol_size)//2, (height - mol_size)//2 + 25), mol_img)

    return img


def generate_video(steps, target_smiles, output_path, fps=1, duration_per_step=2):
    """Generate MP4 video from synthesis steps."""
    frames = []
    frames_per_step = int(fps * duration_per_step)
    total_steps = len(steps)

    # Step frames (no title frame)
    for i, step in enumerate(steps):
        frame = create_step_frame(i + 1, total_steps, step['reactants'], step['reaction'], step['product'])
        for _ in range(frames_per_step):
            frames.append(frame)

    # Final frame
    final = create_final_frame(steps[-1]['product'])
    for _ in range(frames_per_step):
        frames.append(final)

    # Save frames and convert to video with ffmpeg
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(frames):
            frame.save(f"{tmpdir}/frame_{i:04d}.png")

        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', f'{tmpdir}/frame_%04d.png',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    print(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate MP4 synthesis animation')
    parser.add_argument('--smiles', required=True, help='Target molecule SMILES')
    parser.add_argument('--output', '-o', default='synthesis.mp4', help='Output MP4 path')
    parser.add_argument('--max-size', type=int, default=12, help='Max component size')
    parser.add_argument('--num-paths', type=int, default=10, help='Number of paths to try')
    parser.add_argument('--duration', type=float, default=2, help='Seconds per step')
    args = parser.parse_args()

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
        print("No paths generated.")
        sys.exit(1)

    print(f"Generated {len(paths)} paths, finding one that works...")

    # Find best path
    annotated_path = None
    for path in paths:
        result, annotated = execute_path(path, reactions)
        if result and annotated:
            annotated_path = annotated
            result_mol = Chem.MolFromSmiles(result)
            if result_mol and Chem.MolToSmiles(result_mol, canonical=True) == target:
                break  # Found exact match

    if not annotated_path:
        print("No working paths found.")
        sys.exit(1)

    steps = parse_annotated_path(annotated_path)
    print(f"Creating video with {len(steps)} steps...")
    generate_video(steps, target, args.output, duration_per_step=args.duration)


if __name__ == "__main__":
    main()

