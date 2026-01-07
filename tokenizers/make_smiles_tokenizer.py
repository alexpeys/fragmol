"""
SMILES Tokenizer - HuggingFace tokenizers compatible.
Base primitives only - small vocab, compositional.
"""

from pathlib import Path
from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split, Sequence


# Regex pattern that matches:
# 1. Special tokens first: [PAD], [UNK], [BOS], [EOS], [CLS], [MASK], [SEP]
# 2. Two-letter elements
# 3. Single characters
SPECIAL_TOKENS_PATTERN = r"(\[PAD\]|\[UNK\]|\[BOS\]|\[EOS\]|\[CLS\]|\[MASK\]|\[SEP\])"
SMILES_PATTERN = r"(Br|Cl|Si|Se|As|Te|Na|Mg|Al|Ar|Ca|Sc|Ti|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|Kr|Rb|Sr|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|Fr|Ra|Ac|Th|Pa|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv|Ts|Og|se|as|[A-Za-z@\.\-\+\=\#\:\\/\(\)\[\]\d\%])"

# Combined pattern: special tokens OR SMILES tokens
COMBINED_PATTERN = f"({SPECIAL_TOKENS_PATTERN[1:-1]}|{SMILES_PATTERN[1:-1]})"


def get_smiles_vocab() -> dict:
    """Generate base SMILES vocabulary - primitives only."""
    vocab = {}
    idx = 0

    # Special tokens
    for tok in ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[CLS]", "[MASK]", "[SEP]"]:
        vocab[tok] = idx
        idx += 1

    # All 118 elements (two-letter ones captured by regex)
    elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]

    # Aromatic
    aromatic = ["b", "c", "n", "o", "p", "s", "se", "as"]

    # Bonds
    bonds = ["-", "=", "#", ":", "/", "\\"]

    # Structural
    structural = ["(", ")", "[", "]", "."]

    # Digits
    digits = list("0123456789")

    # Other
    other = ["%", "@", "+"]

    # Add all
    for tok in elements + aromatic + bonds + structural + digits + other:
        if tok not in vocab:
            vocab[tok] = idx
            idx += 1

    return vocab


def create_and_save_tokenizer(output_dir: str = "smiles_tokenizer_simple"):
    """Create SMILES tokenizer and save to directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab = get_smiles_vocab()
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    # Use combined pattern that matches special tokens first, then SMILES tokens
    tokenizer.pre_tokenizer = Split(pattern=Regex(COMBINED_PATTERN), behavior="isolated")
    tokenizer.save(str(output_path / "tokenizer.json"))

    print(f"Saved tokenizer to {output_path}")
    print(f"Vocabulary size: {len(vocab)}")
    return tokenizer


def load_tokenizer(tokenizer_dir: str = "smiles_tokenizer_simple") -> Tokenizer:
    """Load tokenizer from directory."""
    return Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))


if __name__ == "__main__":
    tokenizer = create_and_save_tokenizer()

    # Test SMILES without special tokens
    test_smiles = ["CCO", "c1ccccc1", "[Cu+2]", "C[C@H](N)C(=O)O", "[13CH4]", "C/C=C\\C"]
    print("\nTest tokenization (raw SMILES):")
    for smi in test_smiles:
        enc = tokenizer.encode(smi)
        print(f"{smi} -> {enc.tokens}")

    # Test with special tokens
    print("\nTest tokenization (with [BOS]/[EOS]):")
    for smi in test_smiles:
        wrapped = f"[BOS]{smi}[EOS]"
        enc = tokenizer.encode(wrapped)
        print(f"{wrapped} -> {enc.tokens}")

