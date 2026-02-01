import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

from utils.models import Smile2SmileVAE, Smile2SmileEncoderWithJEPA, LlamaConfig
from tokenizers import Tokenizer


def clean_smiles(smiles):
    """Clean up malformed SMILES strings."""
    if pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    # Remove anything after .[text] like ".[Relative stereochemistry]"
    if '.[' in smiles:
        smiles = smiles.split('.[')[0]
    # Fix common uppercase element issues
    smiles = smiles.replace('CL', 'Cl').replace('BR', 'Br')
    # Validate with RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on STAT3 Screen")
    parser.add_argument("--model", type=str,
                        default='s3://shvaibackups/mol_porque/e768_l20_h12_decode1_constrast1/40001.pt',
                        help="Model: 's3://...' or local path, or 'molformer'/'chemberta'/'chemberta-mlm'")
    parser.add_argument("--data_path", type=str, default="stat3.csv")
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4)
    parser.add_argument("--num_layers", type=int, default=20)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--max_mol_size", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--noise_scale", type=float, default=0.0, help="Noise scale for VAE encoding (0 = deterministic)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction for scaffold split")
    # Fine-tuning args
    parser.add_argument("--full_fine_tune", action="store_true", help="Full fine-tune encoder + linear head")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--epochs", type=int, default=8, help="Number of fine-tuning epochs")
    parser.add_argument("--ft_batch_size", type=int, default=256, help="Batch size for fine-tuning")
    parser.add_argument("--ft_weight_decay", type=float, default=0, help="Weight decay for fine-tuning")
    parser.add_argument("--out_file", type=str, default=None, help="Output CSV file for results (optional)")
    return parser.parse_args()


def get_scaffold(smiles):
    """Get Murcko scaffold for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except:
        return None


def scaffold_split(df, smiles_col='smiles', test_size=0.2, seed=42):
    """Split dataframe by scaffold, ensuring molecules with same scaffold stay together."""
    np.random.seed(seed)
    
    # Get scaffold for each molecule
    scaffolds = df[smiles_col].apply(get_scaffold)
    
    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in enumerate(scaffolds):
        if scaffold is not None:
            scaffold_to_indices[scaffold].append(idx)
        else:
            # Treat each invalid scaffold as its own group
            scaffold_to_indices[f'_invalid_{idx}'].append(idx)
    
    # Shuffle scaffolds
    scaffold_list = list(scaffold_to_indices.keys())
    np.random.shuffle(scaffold_list)
    
    # Split scaffolds into train/test
    train_indices, test_indices = [], []
    n_total = len(df)
    n_test_target = int(n_total * test_size)
    
    for scaffold in scaffold_list:
        indices = scaffold_to_indices[scaffold]
        if len(test_indices) < n_test_target:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)
    
    return train_indices, test_indices


def is_vae_model(model_path):
    """Check if model path indicates VAE or JEPA model."""
    return 'mol_vae_v9000' in model_path


def load_model(args, tokenizer, device):
    vocab_size = max(tokenizer.get_vocab().values()) + 1
    config_kwargs = dict(
        vocab_size=vocab_size, hidden_size=args.emb_dim,
        intermediate_size=int(args.emb_dim * args.intermediate_size_multiplier),
        num_hidden_layers=args.num_layers, num_attention_heads=args.num_attention_heads,
        pad_token_id=tokenizer.token_to_id('[PAD]'), bos_token_id=tokenizer.token_to_id('[BOS]'),
        eos_token_id=tokenizer.token_to_id('[EOS]'), cls_token_id=tokenizer.token_to_id('[BOS]'),
        max_position_embeddings=args.max_mol_size, rms_norm_eps=1e-6, initializer_range=0.02,
        attention_dropout=0.0, rope_base=10_000, use_rope=True,
    )
    encoder_config = LlamaConfig(**config_kwargs)
    decoder_config = LlamaConfig(**config_kwargs)

    use_vae = is_vae_model(args.model)
    if use_vae:
        model = Smile2SmileVAE(encoder_config, decoder_config)
        print("Using VAE model")
    else:
        model = Smile2SmileEncoderWithJEPA(encoder_config, decoder_config)
        print("Using JEPA encoder model")

    print(f"Loading checkpoint from: {args.model}")
    if args.model.startswith('s3://'):
        from s3torchconnector import S3Checkpoint
        with S3Checkpoint("us-west-2").reader(args.model) as reader:
            checkpoint = torch.load(reader, map_location='cpu')
    else:
        checkpoint = torch.load(args.model, map_location='cpu')

    if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
        checkpoint = {k.replace('_orig_mod.', '', 1): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    del checkpoint
    return model.to(device).eval(), use_vae


class EncoderWithHead(nn.Module):
    """Encoder with single classification head."""
    def __init__(self, encoder, emb_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = output['hidden_state'][:, 0, :]  # [CLS] token
        return self.head(cls_emb).squeeze(-1)


class SmilesDataset(Dataset):
    """Dataset for SMILES with single binary label."""
    def __init__(self, smiles_list, labels, tokenizer, max_len):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        ids = self.tokenizer.encode(f'[BOS]{smiles}[EOS]').ids[:self.max_len]
        padded_ids = ids + [self.pad_token_id] * (self.max_len - len(ids))
        mask = [1] * len(ids) + [0] * (self.max_len - len(ids))
        return {
            'input_ids': torch.tensor(padded_ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.bool),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
        }


def fine_tune(encoder, train_smiles, train_labels, test_smiles, test_labels,
              tokenizer, args, device):
    """Full fine-tune encoder + head."""
    model = EncoderWithHead(encoder, args.emb_dim).to(device)

    train_dataset = SmilesDataset(train_smiles, train_labels, tokenizer, args.max_mol_size)
    test_dataset = SmilesDataset(test_smiles, test_labels, tokenizer, args.max_mol_size)

    train_loader = DataLoader(train_dataset, batch_size=args.ft_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.ft_batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.ft_weight_decay)

    # pos_weight for imbalanced data
    y = train_labels
    pw = (1 - y.mean()) / max(y.mean(), 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Test AUC
        model.eval()
        test_preds, test_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(input_ids, attention_mask)
                test_preds.append(torch.sigmoid(logits).float().cpu().numpy())
                test_true.append(batch['labels'].numpy())

        test_preds = np.concatenate(test_preds)
        test_true = np.concatenate(test_true)
        auc = roc_auc_score(test_true, test_preds)
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, test_auc={auc:.4f}")

    return model


def predict(model, smiles_list, tokenizer, args, device):
    """Get predictions from fine-tuned model."""
    dummy_labels = np.zeros(len(smiles_list))
    dataset = SmilesDataset(smiles_list, dummy_labels, tokenizer, args.max_mol_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
            preds.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(preds)


def get_embeddings(smiles_list, model, tokenizer, args, device, use_vae):
    pad_token_id = tokenizer.token_to_id("[PAD]")
    embeddings = []
    for i in tqdm(range(0, len(smiles_list), args.batch_size), desc="Embeddings", leave=False):
        batch_smiles = smiles_list[i:i + args.batch_size]
        all_ids, all_masks = [], []
        for smiles in batch_smiles:
            ids = tokenizer.encode(f'[BOS]{smiles}[EOS]').ids[:args.max_mol_size]
            padded_ids = ids + [pad_token_id] * (args.max_mol_size - len(ids))
            all_ids.append(padded_ids)
            all_masks.append([1] * len(ids) + [0] * (args.max_mol_size - len(ids)))
        input_ids = torch.tensor(all_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(all_masks, dtype=torch.bool, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if use_vae:
                encode_out = model.encode(input_ids=input_ids, attention_mask=attention_mask, noise_ratio=args.noise_scale)
                embeddings.append(encode_out['latent'].squeeze(1).float().cpu().numpy())
            else:
                output = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(output['hidden_state'][:, 0, :].float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def auc_ci_normal_approx(y_true, y_pred):
    """Compute AUC with CI using Hanley-McNeil normal approximation."""
    auc = roc_auc_score(y_true, y_pred)
    n1 = np.sum(y_true == 1)
    n2 = np.sum(y_true == 0)
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)) / (n1 * n2))
    return {
        'mean': auc, 'std': se,
        'p025': auc - 1.96 * se, 'p05': auc - 1.645 * se,
        'p95': auc + 1.645 * se, 'p975': auc + 1.96 * se,
    }


def format_ci(stats, ci_level='95'):
    if ci_level == '95':
        return f"{stats['mean']:.3f} [{stats['p025']:.3f}-{stats['p975']:.3f}]"
    return f"{stats['mean']:.3f} [{stats['p05']:.3f}-{stats['p95']:.3f}]"


def evaluate_task(X_train, y_train, X_test, y_test, args):
    """Train linear and XGBoost classifiers, return test AUC stats."""
    # Linear
    lr = LogisticRegressionCV(cv=5, max_iter=1000, random_state=args.seed, solver='lbfgs', n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict_proba(X_test)[:, 1]
    lr_stats = auc_ci_normal_approx(y_test, lr_preds)

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=args.seed,
                                   eval_metric='auc', verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    xgb_stats = auc_ci_normal_approx(y_test, xgb_preds)

    return lr_stats, xgb_stats


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.device}')
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Mode: {'Full Fine-Tune' if args.full_fine_tune else 'Frozen Embeddings'}")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    # Clean SMILES (fix uppercase elements, remove trailing text)
    print("Cleaning SMILES...")
    df['smiles_clean'] = df['smiles'].apply(clean_smiles)
    n_before = len(df)
    df = df.dropna(subset=['smiles_clean', 'is_hit'])
    n_after = len(df)
    print(f"Dropped {n_before - n_after} invalid SMILES, {n_after} remaining")
    df['smiles'] = df['smiles_clean']
    df = df.drop(columns=['smiles_clean'])

    # Scaffold split
    print(f"Performing scaffold split with seed={args.seed}, test_size={args.test_size}...")
    train_indices, test_indices = scaffold_split(df, smiles_col='smiles', test_size=args.test_size, seed=args.seed)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Create labels
    train_labels = train_df['is_hit'].astype(int).values
    test_labels = test_df['is_hit'].astype(int).values

    print(f"  is_hit: train {train_labels.sum()}/{len(train_labels)} ({100*train_labels.mean():.1f}%), "
          f"test {test_labels.sum()}/{len(test_labels)} ({100*test_labels.mean():.1f}%)")

    # Load model/tokenizer
    model, tokenizer, use_vae = None, None, False
    if args.model not in ['molformer', 'chemberta', 'chemberta-mlm']:
        tokenizer = Tokenizer.from_file('tokenizers/smiles_tokenizer_simple/tokenizer.json')
        model, use_vae = load_model(args, tokenizer, device)
        print("Model loaded successfully")

    if args.full_fine_tune:
        if args.model in ['molformer', 'chemberta', 'chemberta-mlm']:
            raise ValueError("Full fine-tuning not supported for baseline models")

        train_smiles = train_df['smiles'].tolist()
        test_smiles = test_df['smiles'].tolist()

        print(f"\n--- Fine-tuning on is_hit ---")
        print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")

        # Fine-tune
        ft_model = fine_tune(model.encoder, train_smiles, train_labels, test_smiles, test_labels,
                             tokenizer, args, device)

        # Final predictions
        test_preds = predict(ft_model, test_smiles, tokenizer, args, device)

        print(f"\n{'='*60}")
        print("FINAL RESULTS (Full Fine-Tune)")
        print(f"{'='*60}")

        ft_stats = auc_ci_normal_approx(test_labels, test_preds)
        print(f"is_hit AUC (95% CI): {format_ci(ft_stats, '95')}")

        del ft_model
        torch.cuda.empty_cache()

    else:
        # Frozen embeddings mode
        print("\nComputing embeddings...")
        if args.model in ['molformer', 'chemberta', 'chemberta-mlm']:
            from utils.deepchem_embeddings import get_deepchem_embeddings
            train_emb = get_deepchem_embeddings(train_df['smiles'].tolist(), args.model, args.batch_size, device)
            test_emb = get_deepchem_embeddings(test_df['smiles'].tolist(), args.model, args.batch_size, device)
        else:
            train_emb = get_embeddings(train_df['smiles'].tolist(), model, tokenizer, args, device, use_vae)
            test_emb = get_embeddings(test_df['smiles'].tolist(), model, tokenizer, args, device, use_vae)

        print(f"\n{'='*90}")
        print("STAT3 SCREEN RESULTS (Frozen Embeddings)")
        print(f"{'='*90}")
        print(f"{'Outcome':<15} {'Linear AUC (95% CI)':<30} {'XGBoost AUC (95% CI)':<30}")
        print("-" * 90)

        lr_stats, xgb_stats = evaluate_task(train_emb, train_labels, test_emb, test_labels, args)
        print(f"{'is_hit':<15} {format_ci(lr_stats, '95'):<30} {format_ci(xgb_stats, '95'):<30}")
        print(f"{'(90% CI)':<15} {format_ci(lr_stats, '90'):<30} {format_ci(xgb_stats, '90'):<30}")

        # Save results to CSV if --out_file is specified
        if args.out_file is not None:
            rows = [{
                'dataset': 'stat3_screen',
                'model': args.model,
                'metric_type': 'stat3_is_hit',
                'metric_value': lr_stats['mean'],
                'metric_90pct_ci': f"[{lr_stats['p05']:.4f}-{lr_stats['p95']:.4f}]",
                'metric_95pct_ci': f"[{lr_stats['p025']:.4f}-{lr_stats['p975']:.4f}]",
            }]
            # Also add as avg for consistency with other scripts
            rows.append({
                'dataset': 'stat3_screen',
                'model': args.model,
                'metric_type': 'stat3_avg',
                'metric_value': lr_stats['mean'],
                'metric_90pct_ci': f"[{lr_stats['p05']:.4f}-{lr_stats['p95']:.4f}]",
                'metric_95pct_ci': f"[{lr_stats['p025']:.4f}-{lr_stats['p975']:.4f}]",
            })
            out_df = pd.DataFrame(rows)
            out_dir = os.path.dirname(args.out_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            out_df.to_csv(args.out_file, index=False)
            print(f"\nResults saved to {args.out_file}")


if __name__ == "__main__":
    main()

