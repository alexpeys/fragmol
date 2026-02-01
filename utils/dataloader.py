import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from rdkit import Chem
from rdkit.Chem import BRICS, Recap
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


def get_random_fragment(smiles):
    """
    Fragment a molecule using BRICS or RECAP and return a random fragment.
    Removes attachment points to return clean SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Randomly choose decomposition method
    method = random.choice(['brics', 'recap'])

    if method == 'brics':
        # BRICS decomposition
        fragments = list(BRICS.BRICSDecompose(mol))
    else:
        # RECAP decomposition
        recap_tree = Recap.RecapDecompose(mol)
        leaves = recap_tree.GetLeaves()
        fragments = list(leaves.keys()) if leaves else []

    if not fragments:
        return None

    # Pick a random fragment
    fragment_smiles = random.choice(fragments)

    # Remove attachment points (dummy atoms marked as [*], [1*], [2*], etc.)
    # Parse the fragment and remove dummy atoms
    frag_mol = Chem.MolFromSmiles(fragment_smiles)
    if frag_mol is None:
        return None

    # Remove dummy atoms (atomic number 0)
    edit_mol = Chem.RWMol(frag_mol)
    atoms_to_remove = []
    for atom in edit_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom
            atoms_to_remove.append(atom.GetIdx())

    # Remove in reverse order to maintain indices
    for idx in sorted(atoms_to_remove, reverse=True):
        edit_mol.RemoveAtom(idx)

    # Get clean SMILES
    try:
        clean_smiles = Chem.MolToSmiles(edit_mol)
        # Verify it's valid
        if Chem.MolFromSmiles(clean_smiles) is None:
            return None
        return clean_smiles
    except:
        return None


def get_random_smiles(mol):
    """Generate a random (non-canonical) SMILES for a molecule."""
    return Chem.MolToSmiles(mol, doRandom=True)


class SmilesVAEDataset(IterableDataset):
    def __init__(self, file_list, tokenizer, max_len):
        super().__init__()

        self.file_list = file_list
        self.mask_token_id = tokenizer.token_to_id("[MASK]")
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.bos_token_id = tokenizer.token_to_id("[BOS]")
        self.eos_token_id = tokenizer.token_to_id("[EOS]")
        self.unk_token_id = tokenizer.token_to_id("[UNK]")

        self.tokenizer = tokenizer
        self.max_len = max_len

    def _pad_and_mask(self, token_ids):
        """Pad token ids to max_len and create attention mask."""
        seq_len = len(token_ids)
        if seq_len > self.max_len:
            token_ids = token_ids[:self.max_len]
            seq_len = self.max_len

        # Pad on the right
        padding_len = self.max_len - seq_len
        padded_ids = token_ids + [self.pad_token_id] * padding_len
        attention_mask = [True] * seq_len + [False] * padding_len

        return padded_ids, attention_mask

    def __iter__(self):
        while True:
            # Set random seed based on current time
            random.seed(time.time_ns())
            np.random.seed(int(time.time_ns() % 2**32))

            df = pd.read_parquet(random.sample(self.file_list, 1)[0]).sample(frac=1)

            for _, row in df.iterrows():
                smiles = row.smiles

                if np.random.rand() < 0.1:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            if np.random.rand() < 0.1:
                                # 10% chance to strip all stereochemistry
                                Chem.RemoveStereochemistry(mol)
                                smiles = Chem.MolToSmiles(mol)
                            else:
                                # Get random stereoisomer
                                opts = StereoEnumerationOptions(tryEmbedding=False, unique=True, maxIsomers=16)
                                isomers = list(EnumerateStereoisomers(mol, options=opts))
                                if isomers:
                                    smiles = Chem.MolToSmiles(random.choice(isomers))
                    except:
                        smiles = smiles
                        
                # 50% chance to use fragment instead of full molecule
                if np.random.rand() >= 0.5:
                    try:
                        fragment = get_random_fragment(smiles)
                        if fragment is not None:
                            smiles = fragment
                    except:
                        pass  # Keep original smiles on failure

                # Validate molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Get canonical and random SMILES
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                random_smiles = get_random_smiles(mol)

                # Tokenize with BOS and EOS tokens
                canonical_ids = self.tokenizer.encode(f'[BOS]{canonical_smiles}[EOS]').ids
                random_ids = self.tokenizer.encode(f'[BOS]{random_smiles}[EOS]').ids

                # Pad and create attention masks
                canonical_ids, canonical_attention_mask = self._pad_and_mask(canonical_ids)
                random_ids, random_attention_mask = self._pad_and_mask(random_ids)

                # Check for UNK tokens
                if self.unk_token_id in canonical_ids or self.unk_token_id in random_ids:
                    print(f"UNK token detected in SMILES: {smiles}")
                    tokens = [self.tokenizer.id_to_token(int(tid)) for tid in canonical_ids]
                    print("Tokens:", tokens)
                    print("UNK token detected, if this happens often this is a problem.")

                yield {
                    'canonical_ids': torch.tensor(canonical_ids, dtype=torch.long),
                    'canonical_attention_mask': torch.tensor(canonical_attention_mask, dtype=torch.bool),
                    'random_ids': torch.tensor(random_ids, dtype=torch.long),
                    'random_attention_mask': torch.tensor(random_attention_mask, dtype=torch.bool),
                }




class SmilesJEPADataset(IterableDataset):
    def __init__(self, file_list, tokenizer, max_len, pubchem_files=None, use_pubchem=False):
        super().__init__()

        self.file_list = file_list
        self.mask_token_id = tokenizer.token_to_id("[MASK]")
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.bos_token_id = tokenizer.token_to_id("[BOS]")
        self.eos_token_id = tokenizer.token_to_id("[EOS]")
        self.unk_token_id = tokenizer.token_to_id("[UNK]")

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.pubchem_files = pubchem_files
        self.use_pubchem = use_pubchem

    def _pad_and_mask(self, token_ids):
        """Pad token ids to max_len and create attention mask."""
        seq_len = len(token_ids)
        if seq_len > self.max_len:
            token_ids = token_ids[:self.max_len]
            seq_len = self.max_len

        # Pad on the right
        padding_len = self.max_len - seq_len
        padded_ids = token_ids + [self.pad_token_id] * padding_len
        attention_mask = [True] * seq_len + [False] * padding_len

        return padded_ids, attention_mask

    def _apply_mask(self, token_ids, mask_ratio):
        """Apply random masking to token ids, excluding BOS and EOS tokens."""
        if mask_ratio == 0.0 or len(token_ids) <= 2:
            return token_ids

        # Don't mask BOS (first) and EOS (last) tokens
        inner_ids = token_ids[1:-1]
        num_to_mask = int(len(inner_ids) * mask_ratio)

        if num_to_mask > 0:
            mask_indices = random.sample(range(len(inner_ids)), num_to_mask)
            for idx in mask_indices:
                inner_ids[idx] = self.mask_token_id

        return [token_ids[0]] + inner_ids + [token_ids[-1]]

    def __iter__(self):
        while True:
            # Set random seed based on current time
            random.seed(time.time_ns())
            np.random.seed(int(time.time_ns() % 2**32))
            df = pd.read_parquet(random.sample(self.file_list, 1)[0], columns=['smiles'])

            if self.use_pubchem:
                pubchem_to_use = random.sample(self.pubchem_files, 10)
                pubchem = pd.concat([pd.read_parquet(f, columns=['smiles']) for f in pubchem_to_use])
                df = pd.concat([df, pubchem]).sample(frac=1)

            # Filter out multi-component SMILES (salts, mixtures, etc.) and long SMILES
            df = df[~df['smiles'].str.contains('.', regex=False, na=False)]
            df = df[df['smiles'].str.len() <= self.max_len - 2].sample(frac=1)
            print(f"Loading a dataframe with: {df.shape} smiles")

            for _, row in df.iterrows():
                smiles = row.smiles

                # Validate molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Get canonical and random SMILES
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                if np.random.rand() < .5:
                    view1_smiles = get_random_smiles(mol)
                else:
                    view1_smiles = canonical_smiles

                # 10% chance to use fragment instead of full molecule (only for short SMILES)
                if np.random.rand() >= 0.9 and len(smiles) < 75:
                    try:
                        fragment = get_random_fragment(smiles)
                        if fragment is not None:
                            view2_smiles = fragment
                        else:
                            view2_smiles = get_random_smiles(mol)
                    except:
                        view2_smiles = get_random_smiles(mol)
                else:
                    view2_smiles = get_random_smiles(mol)

                # Tokenize with BOS and EOS tokens
                canonical_ids = self.tokenizer.encode(f'[BOS]{canonical_smiles}[EOS]').ids
                view1_ids = self.tokenizer.encode(f'[BOS]{view1_smiles}[EOS]').ids
                view2_ids = self.tokenizer.encode(f'[BOS]{view2_smiles}[EOS]').ids

                # replace between 0 and 50% of view1_ids and view2_ids with MASK token
                # 10% of the time, no masking (0%)
                if random.random() < 0.1:
                    mask_ratio1 = 0.0
                    mask_ratio2 = 0.0
                else:
                    mask_ratio1 = random.uniform(0.0, 0.5)
                    mask_ratio2 = random.uniform(0.0, 0.5)

                view1_ids = self._apply_mask(view1_ids, mask_ratio1)
                view2_ids = self._apply_mask(view2_ids, mask_ratio2)

                # Pad and create attention masks
                canonical_ids, canonical_attention_mask = self._pad_and_mask(canonical_ids)
                view1_ids, view1_attention_mask = self._pad_and_mask(view1_ids)
                view2_ids, view2_attention_mask = self._pad_and_mask(view2_ids)

                # Check for UNK tokens
                if self.unk_token_id in canonical_ids or self.unk_token_id in view1_ids or self.unk_token_id in view2_ids:
                    print(f"UNK token detected in SMILES: {smiles}")
                    tokens = [self.tokenizer.id_to_token(int(tid)) for tid in canonical_ids]
                    print("Tokens:", tokens)
                    print("UNK token detected, if this happens often this is a problem.")

                yield {
                    'canonical_ids': torch.tensor(canonical_ids, dtype=torch.long),
                    'canonical_attention_mask': torch.tensor(canonical_attention_mask, dtype=torch.bool),
                    'view1_ids': torch.tensor(view1_ids, dtype=torch.long),
                    'view1_attention_mask': torch.tensor(view1_attention_mask, dtype=torch.bool),
                    'view2_ids': torch.tensor(view2_ids, dtype=torch.long),
                    'view2_attention_mask': torch.tensor(view2_attention_mask, dtype=torch.bool),
                }
