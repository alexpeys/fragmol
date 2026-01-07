import numpy as np
from tqdm import tqdm


def get_deepchem_embeddings(smiles_list, model_name='molformer', batch_size=256, device=None):
    """
    Get molecular embeddings using HuggingFace models.

    Args:
        smiles_list: List of SMILES strings
        model_name: Model to use:
            - 'molformer': ibm/MoLFormer-XL-both-10pct (47M params, 768-dim)
            - 'chemberta': DeepChem/ChemBERTa-77M-MTR (77M params, 384-dim)
            - 'chemberta-mlm': DeepChem/ChemBERTa-100M-MLM (100M params, 768-dim)
        batch_size: Batch size for inference
        device: torch device (if None, uses cuda if available)

    Returns:
        numpy array of embeddings (N, embedding_dim)
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model configurations - only models that work with AutoModel
    MODEL_CONFIGS = {
        'molformer': {
            'model_id': 'ibm/MoLFormer-XL-both-10pct',
            'trust_remote_code': True,
        },
        'chemberta': {
            'model_id': 'DeepChem/ChemBERTa-77M-MTR',
            'trust_remote_code': False,
        },
        'chemberta-mlm': {
            'model_id': 'DeepChem/ChemBERTa-100M-MLM',
            'trust_remote_code': False,
        },
    }

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    print(f"Loading {model_name} from {config['model_id']}...")

    model = AutoModel.from_pretrained(
        config['model_id'],
        trust_remote_code=config['trust_remote_code']
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_id'],
        trust_remote_code=config['trust_remote_code']
    )

    model = model.to(device)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc=f"Getting {model_name} embeddings"):
        batch_smiles = smiles_list[i:i + batch_size]

        inputs = tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Use CLS token (first token) from last hidden state
                embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)

