"""Dataloader for synthesis path training."""
import os
import random
import signal
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, QED, Descriptors, rdMolDescriptors, ChemicalFeatures
from rdkit.Contrib.SA_Score import sascorer

from synthesis.helpers import get_reactions, generate_paths, execute_path


# Pharmacophore feature types (1-indexed, 0 is padding)
PHARMACOPHORE_TYPES = ['Donor', 'Acceptor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'PosIonizable', 'NegIonizable']
PHARMACOPHORE_TYPE_TO_ID = {t: i + 1 for i, t in enumerate(PHARMACOPHORE_TYPES)}  # 1-indexed

# Global feature factory (lazy loaded)
_FEATURE_FACTORY = None


class TimeoutError(Exception):
    """Exception raised when a function times out."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Function timed out")


def _get_feature_factory():
    """Get or create the RDKit feature factory."""
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef_path)
    return _FEATURE_FACTORY


def _canonical_rotation(coords):
    """
    Apply canonical rotation using PCA to align coordinates.
    Principal axes become x, y, z axes.
    """
    if len(coords) == 0:
        return coords

    # Center the coordinates
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid

    if len(coords) < 3:
        # Not enough points for proper PCA, just return centered
        return centered

    # PCA via SVD - use full_matrices=True to get 3x3 Vt
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=True)
        # Vt is 3x3, so rotation preserves 3D
        rotated = centered @ Vt.T
        return rotated
    except:
        return centered


def get_pharmacophore(smiles, max_len=32, timeout_seconds=0.5, max_smiles_len=50):
    """
    Extract pharmacophore features from a SMILES string.

    Args:
        smiles: SMILES string of molecule
        max_len: Maximum number of pharmacophore points (pads with zeros)
        timeout_seconds: Maximum time allowed for conformer generation (default 0.5s)
        max_smiles_len: Skip molecules with SMILES longer than this (default 50)

    Returns:
        dict with:
            - 'pharmacophore_types': list of type names (unpadded)
            - 'pharmacophore_type_ids': list of type IDs (1-indexed, 0=pad), padded to max_len
            - 'pharmacophore_coords': numpy array of 3D coordinates, padded to max_len x 3
        Returns None if molecule is invalid, has no pharmacophore features, times out, or SMILES too long.
    """
    # Skip long SMILES - conformer generation is slow for complex molecules
    if len(smiles) > max_smiles_len:
        return None
    # Set up timeout (use setitimer for sub-second precision)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens and generate 3D coordinates (fast method)
        mol = Chem.AddHs(mol)
        # Use ETKDG with minimal iterations for speed
        params = AllChem.ETKDGv3()
        params.maxIterations = 50  # Reduce from default 1000
        params.numThreads = 1
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            # Embedding failed, try with random coords
            params.useRandomCoords = True
            params.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                return None

        # Skip UFF optimization - it's slow and pharmacophore positions are approximate anyway

        # Get feature factory and extract features
        factory = _get_feature_factory()
        features = factory.GetFeaturesForMol(mol)

        if len(features) == 0:
            return None

        # Extract coordinates and types
        coords_list = []
        types_list = []
        type_ids_list = []

        for feat in features:
            family = feat.GetFamily()
            if family in PHARMACOPHORE_TYPE_TO_ID:
                pos = feat.GetPos()
                coords_list.append([pos.x, pos.y, pos.z])
                types_list.append(family)
                type_ids_list.append(PHARMACOPHORE_TYPE_TO_ID[family])

        if len(coords_list) == 0:
            return None

        coords = np.array(coords_list, dtype=np.float32)

        # Apply canonical rotation
        coords = _canonical_rotation(coords)

        # Truncate if too many features
        if len(coords) > max_len:
            coords = coords[:max_len]
            types_list = types_list[:max_len]
            type_ids_list = type_ids_list[:max_len]

        # Pad to max_len
        num_features = len(coords)
        pad_len = max_len - num_features

        if pad_len > 0:
            coords = np.vstack([coords, np.zeros((pad_len, 3), dtype=np.float32)])
            type_ids_list = type_ids_list + [0] * pad_len

        return {
            'pharmacophore_types': types_list,  # Unpadded list of type names
            'pharmacophore_type_ids': type_ids_list,  # Padded list of type IDs (1-indexed, 0=pad)
            'pharmacophore_coords': coords,  # Padded numpy array [max_len, 3]
        }

    except TimeoutError:
        return None
    finally:
        # Reset timer and restore old handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)




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
    def __init__(self, file_list, tokenizer, max_len, reactions=None, max_component_size=12, include_products=False, add_characterization_tokens=True, prefix_mol_probability=0.0):
        super().__init__()

        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_component_size = max_component_size
        self.include_products = include_products

        self.prefix_mol_probability = prefix_mol_probability

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

                if np.random.rand() < self.prefix_mol_probability:
                    prefix_encoded = self.tokenizer.encode(f'[CLS]{result}[CLS]')
                    prefix_ids = prefix_encoded.ids if hasattr(prefix_encoded, 'ids') else prefix_encoded
                    token_ids = prefix_ids + token_ids

                else:
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


class PharmacophoreDataset(IterableDataset):
    """
    Dataset for pharmacophore-conditioned synthesis path generation.

    For each SMILES:
    1. Generate pharmacophore features (type IDs + 3D coords)
    2. Generate a synthesis path
    3. Return both for training the pharmacophore-conditioned model
    """
    def __init__(
        self,
        file_list,
        tokenizer,
        max_len,
        reactions=None,
        max_component_size=12,
        pharmacophore_max_len=30,
        pharmacophore_timeout=0.5,
    ):
        super().__init__()
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_component_size = max_component_size
        self.pharmacophore_max_len = pharmacophore_max_len
        self.pharmacophore_timeout = pharmacophore_timeout

        self.reactions = reactions if reactions is not None else get_reactions()

        # Support both tokenizers (HF tokenizers) and PreTrainedTokenizerFast (transformers)
        if hasattr(tokenizer, 'token_to_id'):
            self.pad_token_id = tokenizer.token_to_id('[PAD]')
            self.unk_token_id = tokenizer.token_to_id('[UNK]')
            self.bos_token_id = tokenizer.token_to_id('[BOS]')
            self.eos_token_id = tokenizer.token_to_id('[EOS]')
        else:
            self.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
            self.unk_token_id = tokenizer.convert_tokens_to_ids('[UNK]')
            self.bos_token_id = tokenizer.convert_tokens_to_ids('[BOS]')
            self.eos_token_id = tokenizer.convert_tokens_to_ids('[EOS]')

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

                # Get pharmacophore for this molecule
                pharm = get_pharmacophore(
                    smiles,
                    max_len=self.pharmacophore_max_len,
                    timeout_seconds=self.pharmacophore_timeout
                )
                if pharm is None:
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

                # Tokenize the path string
                encoded = self.tokenizer.encode(f'[BOS]{paths[0]}[EOS]')
                token_ids = encoded.ids if hasattr(encoded, 'ids') else encoded

                # Skip if too long
                if len(token_ids) > self.max_len:
                    continue

                # Skip if has UNK
                if self.unk_token_id in token_ids:
                    continue

                padded_ids, attention_mask = self._pad_and_mask(token_ids)

                yield {
                    'recipe_string': paths[0],
                    'product_smiles': result,
                    'input_ids': torch.tensor(padded_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
                    'pharmacophore_type_ids': torch.tensor(pharm['pharmacophore_type_ids'], dtype=torch.long),
                    'pharmacophore_coords': torch.tensor(pharm['pharmacophore_coords'], dtype=torch.float32),
                    'original_smiles': smiles,
                }

