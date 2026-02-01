#!/usr/bin/env python
"""Generate PDFs showing synthesis paths for molecules that need/don't need protecting groups."""
import sys
sys.path.insert(0, '/home/ubuntu/fragmol')

from io import BytesIO

import pandas as pd
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from synthesis.helpers import get_reactions, create_reaction_diagram


def create_pdf(df_subset, output_path, title, reactions, rows_per_page=6):
    """Create PDF with synthesis traces - one line per recipe."""
    doc = SimpleDocTemplate(str(output_path), pagesize=landscape(letter),
                           leftMargin=0.2*inch, rightMargin=0.2*inch,
                           topMargin=0.2*inch, bottomMargin=0.2*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=14, spaceAfter=6)

    story = [Paragraph(title, title_style)]

    count = 0
    for idx, row in df_subset.iterrows():
        recipe = row['recipe']

        try:
            img = create_reaction_diagram(recipe, reactions, mol_size=(100, 100))
            if img:
                buf = BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)

                # Scale to fit page width (landscape letter is ~10 inches wide)
                max_width = 10*inch
                scale = min(1.0, max_width / img.width)
                img_width = img.width * scale
                img_height = img.height * scale

                story.append(Image(buf, width=img_width, height=img_height))
                story.append(Spacer(1, 0.05*inch))

                count += 1
                if count % rows_per_page == 0:
                    story.append(PageBreak())

                if count % 20 == 0:
                    print(f"  Added {count}/{len(df_subset)}")
        except Exception as e:
            print(f"  Error on row {idx}: {e}")
            continue

    doc.build(story)
    print(f"Saved {output_path} with {count} traces")


def main():
    print("Loading data...")
    df = pd.read_parquet('robosean_evals/enamine_examples_for_sean.parquet')
    
    print("Loading reactions...")
    reactions = get_reactions()
    
    # Sample 100 from each category
    df_needs_pg = df[df['needs_protecting_group'] == 1].sample(n=min(100, len(df[df['needs_protecting_group'] == 1])), random_state=42)
    df_no_pg = df[df['needs_protecting_group'] == 0].sample(n=min(100, len(df[df['needs_protecting_group'] == 0])), random_state=42)
    
    print(f"\nCreating PDF for molecules NEEDING protecting groups ({len(df_needs_pg)} samples)...")
    create_pdf(df_needs_pg, 'robosean_evals/needs_protecting_group.pdf', 
               'Molecules Needing Protecting Groups', reactions)
    
    print(f"\nCreating PDF for molecules NOT needing protecting groups ({len(df_no_pg)} samples)...")
    create_pdf(df_no_pg, 'robosean_evals/no_protecting_group.pdf',
               'Molecules NOT Needing Protecting Groups', reactions)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

