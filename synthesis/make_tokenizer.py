"""
Synthesis path tokenizer - extends SMILES tokenizer with <ADD>, <RXN>, and reaction names.
"""
import sys
sys.path.insert(0, '..')

from pathlib import Path
from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

from helpers import get_reactions


def get_synthesis_vocab():
    """Generate synthesis vocabulary - SMILES primitives + synthesis tokens."""
    vocab = {}
    idx = 0

    # Special tokens
    for tok in ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[CLS]", "[MASK]", "[SEP]"]:
        vocab[tok] = idx
        idx += 1

    # Synthesis-specific tokens
    vocab["<ADD>"] = idx
    idx += 1
    vocab["<RXN>"] = idx
    idx += 1
    vocab["<PRODUCT>"] = idx
    idx += 1

    # Property tokens
    property_tokens = [
        '[highly_druglike]',
        '[membrane_bbb]',
        '[membrane_oral]',
        '[synth_easy]',
        '[synth_medium]',
        '[logp_neg]',
        '[logp_0_1.5]',
        '[logp_1.5_3]',
        '[logp_3_5]',
        '[logp_5_plus]',
        '[flex_rigid]',
        '[flex_moderate]',
        '[flex_flexible]',
        '[flat_planar]',
        '[flat_mixed]',
        '[flat_3d]',
    ]
    for tok in property_tokens:
        vocab[tok] = idx
        idx += 1

    # Reaction names from helpers
    reactions = get_reactions()
    for rxn_name in sorted(reactions['syn'].keys()):
        vocab[rxn_name] = idx
        idx += 1

    # All 118 elements
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


def get_tokenizer_pattern(reactions):
    """Build regex pattern for tokenizer."""
    # Special tokens
    special = r"\[PAD\]|\[UNK\]|\[BOS\]|\[EOS\]|\[CLS\]|\[MASK\]|\[SEP\]"

    # Property tokens
    property_toks = r"\[highly_druglike\]|\[membrane_bbb\]|\[membrane_oral\]|\[synth_easy\]|\[synth_medium\]|\[logp_neg\]|\[logp_0_1\.5\]|\[logp_1\.5_3\]|\[logp_3_5\]|\[logp_5_plus\]|\[flex_rigid\]|\[flex_moderate\]|\[flex_flexible\]|\[flat_planar\]|\[flat_mixed\]|\[flat_3d\]"

    # Synthesis tokens
    synthesis = r"<ADD>|<RXN>|<PRODUCT>"

    # Reaction names (escape special chars, sort by length desc to match longer first)
    rxn_names = sorted(reactions['syn'].keys(), key=len, reverse=True)
    rxn_pattern = "|".join(rxn_names)

    # Two-letter elements
    two_letter = r"Br|Cl|Si|Se|As|Te|Na|Mg|Al|Ar|Ca|Sc|Ti|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|Kr|Rb|Sr|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|Fr|Ra|Ac|Th|Pa|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv|Ts|Og|se|as"

    # Single chars
    single = r"[A-Za-z@\.\-\+\=\#\:\\/\(\)\[\]\d\%]"

    return f"({special}|{property_toks}|{synthesis}|{rxn_pattern}|{two_letter}|{single})"


def create_and_save_tokenizer(output_dir: str = "synthesis_tokenizer"):
    """Create synthesis tokenizer and save."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reactions = get_reactions()
    vocab = get_synthesis_vocab()
    pattern = get_tokenizer_pattern(reactions)
    
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(pattern=Regex(pattern), behavior="isolated")
    tokenizer.save(str(output_path / "tokenizer.json"))

    print(f"Saved tokenizer to {output_path}")
    print(f"Vocabulary size: {len(vocab)}")
    return tokenizer


def load_tokenizer(tokenizer_dir: str = "synthesis_tokenizer") -> Tokenizer:
    """Load tokenizer from directory."""
    return Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))


if __name__ == "__main__":
    tokenizer = create_and_save_tokenizer()

    # Test
    test = "<ADD>CCO<ADD>BrC1=CC=CC=C1<RXN>suzuki-2"
    enc = tokenizer.encode(f"[BOS]{test}[EOS]")
    print(f"\nTest: {test}")
    print(f"Tokens: {enc.tokens}")
    print(f"IDs: {enc.ids}")

