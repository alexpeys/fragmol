"""
Script to create a filtered binding database with gene classifier indices.

This script:
1. Loads class_map.json which maps gene perturbations (e.g., "GENE+", "GENE-") to indices
2. Loads binding_db_clean_jan2026.parquet containing binding data with gene names
3. For each row, finds the corresponding class indices for both + and - perturbations
4. Adds a column 'gene_classifier_idx' with the list of indices
5. Keeps only rows where the gene is found in class_map
6. Saves to binding_db_with_class_idx.parquet
"""

import json
import pandas as pd
import boto3


def load_class_map(s3_path: str) -> dict:
    """Load class_map.json from S3."""
    s3 = boto3.client("s3")
    # Parse s3 path
    bucket = s3_path.replace("s3://", "").split("/")[0]
    key = "/".join(s3_path.replace("s3://", "").split("/")[1:])
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def build_gene_to_idx_lookup(class_map: dict) -> dict:
    """
    Build a lookup from gene name (case-insensitive) to list of class indices.
    
    class_map has entries like {"GENE+": 0, "GENE-": 1, ...}
    We want to map "GENE" -> [0, 1] (both + and - indices)
    """
    gene_to_idx = {}
    
    for perturbation, idx in class_map.items():
        # Skip special entries that don't follow gene+/- pattern
        if not (perturbation.endswith("+") or perturbation.endswith("-")):
            continue
        
        # Extract gene name (without + or -)
        gene = perturbation[:-1]
        gene_upper = gene.upper()
        
        if gene_upper not in gene_to_idx:
            gene_to_idx[gene_upper] = []
        gene_to_idx[gene_upper].append(idx)
    
    # Sort indices for consistency
    for gene in gene_to_idx:
        gene_to_idx[gene] = sorted(gene_to_idx[gene])
    
    return gene_to_idx


def main():
    # S3 paths
    class_map_path = "s3://shvaibackups/cellpainto_v3/finetune_models/well_level_split_t25_other/class_map.json"
    binding_db_path = "s3://shvaibackups/unibio_data/clean_binding_data/binding_db_clean_jan2026.parquet"
    output_path = "s3://shvaibackups/cellpainto_v3/finetune_models/well_level_split_t25_other/binding_db_with_class_idx.parquet"

    print("Loading class_map.json...")
    class_map = load_class_map(class_map_path)
    print(f"  Loaded {len(class_map)} perturbation entries")

    print("Building gene to index lookup...")
    gene_to_idx = build_gene_to_idx_lookup(class_map)
    print(f"  Found {len(gene_to_idx)} unique genes")

    print("Loading binding database...")
    df = pd.read_parquet(binding_db_path)
    print(f"  Loaded {len(df)} rows")

    print("Mapping gene names to classifier indices...")
    # Create uppercase gene name column for matching
    df["gene_name_upper"] = df["gene_name"].str.upper()

    # Find which rows have a matching gene
    df["gene_classifier_idx"] = df["gene_name_upper"].map(gene_to_idx)

    # Count matches before filtering
    n_with_idx = df["gene_classifier_idx"].notna().sum()
    print(f"  {n_with_idx} rows have matching gene indices")

    # Filter to only rows with matching genes
    df_filtered = df[df["gene_classifier_idx"].notna()].copy()
    print(f"  Filtered to {len(df_filtered)} rows")

    # Drop the temporary column
    df_filtered = df_filtered.drop(columns=["gene_name_upper"])

    # Show some stats
    print("\nStats:")
    print(f"  Unique genes in filtered data: {df_filtered['gene_name'].nunique()}")
    print(f"  Sample gene_classifier_idx values:")
    for _, row in df_filtered.head(5).iterrows():
        print(f"    {row['gene_name']}: {row['gene_classifier_idx']}")

    print(f"\nSaving to {output_path}...")
    df_filtered.to_parquet(output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()

