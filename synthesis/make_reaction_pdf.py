#!/usr/bin/env python
"""Generate a PDF catalog of all reactions with structures."""
import json
import re
from io import BytesIO
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def parse_display_smarts(display_smarts):
    """Parse display_smarts into reactants and products."""
    if '>>' not in display_smarts:
        return None, None
    
    # Remove atom map labels like |$...$|
    display_smarts = re.sub(r'\s*\|[^|]*\|', '', display_smarts)
    
    reactants_str, products_str = display_smarts.split('>>')
    reactants = reactants_str.split('.')
    products = products_str.split('.')
    
    return reactants, products


def smarts_to_mol(smarts):
    """Convert SMARTS to a displayable mol."""
    # Try as SMARTS first
    mol = Chem.MolFromSmarts(smarts)
    if mol:
        # Try to make it more displayable
        try:
            # Generate 2D coords
            AllChem.Compute2DCoords(mol)
        except:
            pass
    return mol


def draw_reaction(reactants, products, width=600, height=150):
    """Draw reaction as mol1 + mol2 -> product."""
    mols = []
    legends = []
    
    # Add reactants
    for i, r in enumerate(reactants):
        mol = smarts_to_mol(r)
        if mol:
            mols.append(mol)
            legends.append(f"R{i+1}")
            if i < len(reactants) - 1:
                mols.append(None)  # placeholder for +
                legends.append("+")
    
    # Add arrow placeholder
    mols.append(None)
    legends.append("→")
    
    # Add products
    for i, p in enumerate(products):
        mol = smarts_to_mol(p)
        if mol:
            mols.append(mol)
            legends.append(f"P{i+1}")
    
    # Filter out None placeholders and draw
    real_mols = [m for m in mols if m is not None]
    if not real_mols:
        return None
    
    # Draw molecules in a grid
    img = Draw.MolsToGridImage(real_mols, molsPerRow=len(real_mols), 
                                subImgSize=(150, 150), returnPNG=True)
    return img


def draw_reaction_with_arrow(reactants, products, width=700, height=200):
    """Draw reaction with arrow between reactants and products."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    
    # Draw reactants
    r_mols = [smarts_to_mol(r) for r in reactants]
    r_mols = [m for m in r_mols if m]
    
    # Draw products  
    p_mols = [smarts_to_mol(p) for p in products]
    p_mols = [m for m in p_mols if m]
    
    if not r_mols or not p_mols:
        return None
    
    mol_size = (150, 150)
    
    # Create images
    if len(r_mols) > 1:
        r_img = Draw.MolsToGridImage(r_mols, molsPerRow=len(r_mols), subImgSize=mol_size)
    else:
        r_img = Draw.MolToImage(r_mols[0], size=mol_size)
    
    if len(p_mols) > 1:
        p_img = Draw.MolsToGridImage(p_mols, molsPerRow=len(p_mols), subImgSize=mol_size)
    else:
        p_img = Draw.MolToImage(p_mols[0], size=mol_size)
    
    # Combine with arrow
    arrow_width = 60
    plus_width = 30
    
    # Calculate total width
    r_width = r_img.width
    p_width = p_img.width
    total_width = r_width + arrow_width + p_width
    max_height = max(r_img.height, p_img.height)
    
    # Create combined image
    combined = PILImage.new('RGB', (total_width, max_height), 'white')
    
    # Paste reactants
    r_y = (max_height - r_img.height) // 2
    combined.paste(r_img, (0, r_y))
    
    # Draw arrow
    draw = ImageDraw.Draw(combined)
    arrow_x = r_width + 10
    arrow_y = max_height // 2
    draw.line([(arrow_x, arrow_y), (arrow_x + 40, arrow_y)], fill='black', width=2)
    draw.polygon([(arrow_x + 40, arrow_y - 8), (arrow_x + 50, arrow_y), (arrow_x + 40, arrow_y + 8)], fill='black')
    
    # Paste products
    p_y = (max_height - p_img.height) // 2
    combined.paste(p_img, (r_width + arrow_width, p_y))
    
    # Convert to bytes
    buf = BytesIO()
    combined.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()


def main():
    # Load reactions
    reactions_path = Path(__file__).parent / "reactions.json"
    with open(reactions_path) as f:
        content = f.read()
        content = content.replace(': True', ': true').replace(': False', ': false').replace(': None', ': null')
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        reactions = json.loads(content)
    
    print(f"Loaded {len(reactions)} reactions")
    
    # Create PDF
    output_path = Path(__file__).parent / "reactions_catalog.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                           leftMargin=0.5*inch, rightMargin=0.5*inch,
                           topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=14, spaceAfter=6)
    desc_style = ParagraphStyle('Desc', parent=styles['Normal'], fontSize=10, spaceAfter=12)
    
    story = []
    
    # Title page
    story.append(Paragraph("Reaction Catalog", styles['Title']))
    story.append(Spacer(1, 0.5*inch))
    
    for rxn_name, rxn_data in reactions.items():
        description = rxn_data.get('description', 'No description')
        display_smarts = rxn_data.get('ld_data', {}).get('display_smarts', '')
        
        # Reaction name
        story.append(Paragraph(f"<b>{rxn_name}</b>", title_style))
        
        # Description
        story.append(Paragraph(description, desc_style))
        
        # Draw reaction
        if display_smarts:
            reactants, products = parse_display_smarts(display_smarts)
            if reactants and products:
                try:
                    img_data = draw_reaction_with_arrow(reactants, products)
                    if img_data:
                        img = Image(BytesIO(img_data), width=5*inch, height=1.2*inch)
                        story.append(img)
                except Exception as e:
                    story.append(Paragraph(f"<i>Could not render: {e}</i>", desc_style))
        
        story.append(Spacer(1, 0.3*inch))
    
    doc.build(story)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

