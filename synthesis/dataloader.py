"""Dataloader for synthesis path training."""
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer

from synthesis.helpers import get_reactions, generate_paths, execute_path




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

def calculate_properties(smiles):
    """
    Calculate molecular properties and return both raw values and quantized labels.
    
    Args:
        smiles: SMILES string of molecule
        
    Returns:
        dict with 'raw_properties' and 'quantized_properties'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Calculate raw properties
    qed = QED.qed(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    sa_score = sascorer.calculateScore(mol)
    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    
    # Fsp3
    num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    num_sp3_carbons = rdMolDescriptors.CalcFractionCSP3(mol) * num_carbons if num_carbons > 0 else 0
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    
    raw_properties = {
        'qed': qed,
        'logp': logp,
        'tpsa': tpsa,
        'sa_score': sa_score,
        'num_rotatable_bonds': num_rotatable_bonds,
        'fsp3': fsp3,
    }
    
    # Quantize properties
    quantized = []
    
    # Highly druglike (QED > 0.5)
    if qed > 0.5:
        quantized.append('[highly_druglike]')
    
    # Membrane penetration
    if tpsa < 90:
        quantized.append('[membrane_bbb]')
    elif tpsa < 140:
        quantized.append('[membrane_oral]')
    
    # Synthesizability
    if sa_score < 4:
        quantized.append('[synth_easy]')
    elif sa_score < 6:
        quantized.append('[synth_medium]')
    
    # LogP bins
    if logp < 0:
        quantized.append('[logp_neg]')
    elif logp < 1.5:
        quantized.append('[logp_0_1.5]')
    elif logp < 3:
        quantized.append('[logp_1.5_3]')
    elif logp < 5:
        quantized.append('[logp_3_5]')
    else:
        quantized.append('[logp_5_plus]')
    
    # Flexibility (using absolute cutoffs)
    if num_rotatable_bonds <= 2:
        quantized.append('[flex_rigid]')
    elif num_rotatable_bonds <= 6:
        quantized.append('[flex_moderate]')
    else:
        quantized.append('[flex_flexible]')
    
    # Flatness (Fsp3)
    if fsp3 < 0.25:
        quantized.append('[flat_planar]')
    elif fsp3 < 0.5:
        quantized.append('[flat_mixed]')
    else:
        quantized.append('[flat_3d]')
    
    return {
        'raw_properties': raw_properties,
        'quantized_properties': quantized,
    }

class SynthesisPathDataset(IterableDataset):
    def __init__(self, file_list, tokenizer, max_len, reactions=None, max_component_size=12, include_products=False, add_characterization_tokens=True):
        super().__init__()

        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_component_size = max_component_size
        self.include_products = include_products

        self.reactions = reactions if reactions is not None else get_reactions()

        # Support both tokenizers (HF tokenizers) and PreTrainedTokenizerFast (transformers)
        if hasattr(tokenizer, 'token_to_id'):
            self.pad_token_id = tokenizer.token_to_id('[PAD]')
            self.unk_token_id = tokenizer.token_to_id('[UNK]')
        else:
            self.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
            self.unk_token_id = tokenizer.convert_tokens_to_ids('[UNK]')
        self.add_characterization_tokens = add_characterization_tokens

    def _pad_and_mask(self, token_ids):
        """Pad token ids to max_len and create attention mask."""
        seq_len = len(token_ids)
        if seq_len > self.max_len:
            token_ids = token_ids[:self.max_len]
            seq_len = self.max_len

        padding_len = self.max_len - seq_len
        padded_ids = token_ids + [self.pad_token_id] * padding_len
        attention_mask = [True] * seq_len + [False] * padding_len

        return padded_ids, attention_mask

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = time.time_ns()
        if worker_info is not None:
            seed += worker_info.id

        while True:
            random.seed(seed)
            np.random.seed(int(seed % 2**32))
            seed += 1

            df = pd.read_parquet(random.choice(self.file_list)).sample(frac=1)

            for _, row in df.iterrows():
                smiles = row.smiles

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Generate a synthesis path
                paths = generate_paths(
                    smiles, self.reactions,
                    max_component_size=self.max_component_size,
                    num_paths=1,
                    seed=random.randint(0, 2**31)
                )

                if not paths:
                    continue

                # Execute to get product and annotated path
                result, annotated_path = execute_path(paths[0], self.reactions)

                if result is None:
                    continue

                # Use annotated path (with products) or raw path
                path_str = annotated_path if self.include_products else paths[0]

                # Tokenize the whole path string
                # Support both tokenizers (HF tokenizers) and PreTrainedTokenizerFast (transformers)
                encoded = self.tokenizer.encode(f'[BOS]{path_str}[EOS]')
                token_ids = encoded.ids if hasattr(encoded, 'ids') else encoded

                if self.add_characterization_tokens:
                    try:
                        # Use calculate_properties function to get all properties
                        props = calculate_properties(result)
                        quantized_tokens = props['quantized_properties']

                        # Convert property tokens to token IDs
                        characterization_tokens = []
                        for token in quantized_tokens:
                            if hasattr(self.tokenizer, 'token_to_id'):
                                token_id = self.tokenizer.token_to_id(token)
                            else:
                                token_id = self.tokenizer.convert_tokens_to_ids(token)
                            characterization_tokens.append(token_id)

                        # Randomly sample a subset of tokens (optional - you can remove this if you want all tokens)
                        if characterization_tokens and np.random.rand() < 0.5:  # 50% chance to add any tokens
                            # Randomly select some tokens (between 1 and min(3, available tokens))
                            num_to_select = np.random.randint(1, len(characterization_tokens) + 1)
                            selected_tokens = np.random.choice(characterization_tokens, size=num_to_select, replace=False).tolist()

                            # Randomly shuffle the selected tokens
                            random.shuffle(selected_tokens)

                            # Prepend characterization tokens to the beginning (before BOS)
                            token_ids = selected_tokens + token_ids
                    except Exception as e:
                        print(f"Property calculation failed: {e}")
                        pass

                # Skip if too long
                if len(token_ids) > self.max_len:
                    continue

                # Skip if has UNK
                if self.unk_token_id in token_ids:
                    print(f"Found UNK token in token ids")
                    continue

                padded_ids, attention_mask = self._pad_and_mask(token_ids)

                yield {
                    'recipe_string': path_str,
                    'product_smiles': result,
                    'input_ids': torch.tensor(padded_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
                }

