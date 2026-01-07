"""Test the SMILES tokenizer on real ZINC data - ensure NO UNK tokens!"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tokenizers import Tokenizer


def test_on_parquet(parquet_path: str, tokenizer_path: str = "smiles_tokenizer_simple"):
    """Test tokenizer on parquet file, check for UNK tokens."""

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
    unk_id = tokenizer.token_to_id("[UNK]")

    print(f"Loaded tokenizer from {tokenizer_path}")
    print(f"UNK token ID: {unk_id}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print()

    # Read parquet directly from S3 or local
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Find SMILES column
    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower():
            smiles_col = col
            break
    
    if smiles_col is None:
        print(f"Available columns: {list(df.columns)}")
        # Try first column if it looks like SMILES
        smiles_col = df.columns[0]
        print(f"Using column: {smiles_col}")
    
    # Sample 100k
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)

    smiles_list = df[smiles_col].tolist()
    print(f"Testing on {len(smiles_list)} SMILES strings")
    print()
    
    # Test tokenization (with [BOS]/[EOS] like training)
    unk_count = 0
    unk_examples = []
    unk_tokens = Counter()
    total_tokens = 0
    token_lengths = []

    for i, smi in enumerate(smiles_list):
        if not isinstance(smi, str):
            continue

        # Wrap with special tokens like in training
        wrapped = f"[BOS]{smi}[EOS]"
        encoded = tokenizer.encode(wrapped)
        tokens = encoded.tokens
        ids = encoded.ids

        total_tokens += len(tokens)
        token_lengths.append(len(tokens))

        # Check for UNK
        for tok, tok_id in zip(tokens, ids):
            if tok_id == unk_id:
                unk_count += 1
                unk_tokens[tok] += 1
                if len(unk_examples) < 10:
                    unk_examples.append((smi, tok, tokens))

        if (i + 1) % 100000 == 0:
            print(f"Processed {i + 1}/{len(smiles_list)} SMILES...")
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total SMILES processed: {len(smiles_list)}")
    print(f"Total tokens: {total_tokens}")
    print(f"UNK tokens found: {unk_count}")

    # Token length statistics
    token_lengths = np.array(token_lengths)
    print()
    print("TOKEN LENGTH STATISTICS:")
    print(f"  Mean:   {np.mean(token_lengths):.1f}")
    print(f"  Median: {np.median(token_lengths):.1f}")
    print(f"  Min:    {np.min(token_lengths)}")
    print(f"  Max:    {np.max(token_lengths)}")
    print(f"  P95:    {np.percentile(token_lengths, 95):.1f}")
    print(f"  P99:    {np.percentile(token_lengths, 99):.1f}")
    print(f"  P99.9:  {np.percentile(token_lengths, 99.9):.1f}")

    # Show encode/decode examples (with [BOS]/[EOS])
    print()
    print("=" * 60)
    print("ENCODE/DECODE EXAMPLES (with [BOS]/[EOS])")
    print("=" * 60)
    for smi in smiles_list[:5]:
        if not isinstance(smi, str):
            continue
        wrapped = f"[BOS]{smi}[EOS]"
        encoded = tokenizer.encode(wrapped)
        decoded = tokenizer.decode(encoded.ids)
        print(f"Original:  {smi}")
        print(f"Wrapped:   {wrapped}")
        print(f"Tokens:    {encoded.tokens}")
        print(f"IDs:       {encoded.ids}")
        print(f"Decoded:   {decoded}")
        print()

    if unk_count > 0:
        print()
        print("!!! WARNING: UNK TOKENS FOUND !!!")
        print()
        print("Most common UNK tokens:")
        for tok, count in unk_tokens.most_common(20):
            print(f"  '{tok}': {count}")
        print()
        print("Example SMILES with UNK:")
        for smi, tok, toks in unk_examples[:10]:
            print(f"  SMILES: {smi}")
            print(f"  UNK token: '{tok}'")
            print(f"  All tokens: {toks}")
            print()
        return False
    else:
        print()
        print("SUCCESS! No UNK tokens found!")
        return True


if __name__ == "__main__":
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else "s3://shvaibackups/unibio_data/zinc22/shuffled/0000.parquet"
    tokenizer_dir = sys.argv[2] if len(sys.argv) > 2 else "tokenizers/smiles_tokenizer_simple"

    success = test_on_parquet(parquet_path, tokenizer_dir)
    sys.exit(0 if success else 1)

