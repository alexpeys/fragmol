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

from utils.models import Smile2SmileEncoderWithJEPA, LlamaConfig
from tokenizers import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on Senolytic Screen")
    parser.add_argument("--model", type=str,
                        default='s3://shvaibackups/mol_porque/molporque_10layer_jepa0_contrastive1_decode1/40001.pt',
                        help="Model: 's3://...' or local path, or 'molformer'/'chemberta'/'chemberta-mlm'")
    parser.add_argument("--data_path", type=str, default="seno_data_combined.parquet")
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--max_mol_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    # Fine-tuning args
    parser.add_argument("--full_fine_tune", action="store_true", help="Full fine-tune encoder + linear head")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--epochs", type=int, default=8, help="Number of fine-tuning epochs")
    parser.add_argument("--ft_batch_size", type=int, default=256, help="Batch size for fine-tuning")
    parser.add_argument("--ft_weight_decay", type=float, default=0, help="Weight decay for fine-tuning")
    return parser.parse_args()


def load_encoder(args, tokenizer, device):
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
    model = Smile2SmileEncoderWithJEPA(encoder_config, decoder_config)

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
    return model.encoder.to(device).eval()


class EncoderWithMultiHead(nn.Module):
    """Encoder with 3 classification heads for multi-task fine-tuning."""
    def __init__(self, encoder, emb_dim, num_tasks=3):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList([nn.Linear(emb_dim, 1) for _ in range(num_tasks)])

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = output['hidden_state'][:, 0, :]  # [CLS] token
        # Return logits for all 3 tasks: [batch, 3]
        return torch.cat([head(cls_emb) for head in self.heads], dim=-1)


class SmilesMultiTaskDataset(Dataset):
    """Dataset for SMILES with multi-task labels."""
    def __init__(self, smiles_list, labels_dict, tokenizer, max_len):
        self.smiles_list = smiles_list
        self.labels = np.stack([labels_dict[k] for k in ['kills_young', 'kills_sen', 'senolytic']], axis=1)
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),  # [3]
        }


def fine_tune_multitask(encoder, train_smiles, train_labels, test_smiles, test_labels,
                        tokenizer, args, device):
    """Full fine-tune encoder + 3 heads simultaneously."""
    model = EncoderWithMultiHead(encoder, args.emb_dim, num_tasks=3).to(device)

    train_dataset = SmilesMultiTaskDataset(train_smiles, train_labels, tokenizer, args.max_mol_size)
    test_dataset = SmilesMultiTaskDataset(test_smiles, test_labels, tokenizer, args.max_mol_size)

    train_loader = DataLoader(train_dataset, batch_size=args.ft_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.ft_batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.ft_weight_decay)

    # Per-task pos_weight for imbalanced data
    pos_weights = []
    for key in ['kills_young', 'kills_sen', 'senolytic']:
        y = train_labels[key]
        pw = (1 - y.mean()) / max(y.mean(), 1e-6)
        pos_weights.append(pw)
    pos_weight_tensor = torch.tensor(pos_weights, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction='mean')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [batch, 3]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)  # [batch, 3]
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

        test_preds = np.concatenate(test_preds, axis=0)  # [N, 3]
        test_true = np.concatenate(test_true, axis=0)    # [N, 3]

        aucs = [roc_auc_score(test_true[:, i], test_preds[:, i]) for i in range(3)]
        mean_auc = np.mean(aucs)
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
              f"test_auc=[{aucs[0]:.3f}, {aucs[1]:.3f}, {aucs[2]:.3f}], mean={mean_auc:.4f}")

    return model


def predict_multitask(model, smiles_list, tokenizer, args, device):
    """Get predictions from fine-tuned multi-task model."""
    dummy_labels = {k: np.zeros(len(smiles_list)) for k in ['kills_young', 'kills_sen', 'senolytic']}
    dataset = SmilesMultiTaskDataset(smiles_list, dummy_labels, tokenizer, args.max_mol_size)
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
    return np.concatenate(preds, axis=0)  # [N, 3]


def get_embeddings(smiles_list, encoder, tokenizer, args, device):
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
            output = encoder(input_ids=input_ids, attention_mask=attention_mask)
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


def create_labels(df):
    """Create binary labels for the three outcomes."""
    labels = {}
    # kills_young: young < 0.5
    labels['kills_young'] = (df['young'] < 0.5).astype(int).values
    # kills_sen: senescence < 0.5
    labels['kills_sen'] = (df['senescence'] < 0.5).astype(int).values
    # senolytic: young > 0.5 AND senescence < 0.5 AND senescence/young <= 0.7
    ratio = df['senescence'] / df['young']
    labels['senolytic'] = ((df['young'] > 0.5) & (df['senescence'] < 0.5) & (ratio <= 0.7)).astype(int).values
    return labels


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


def mean_auc_ci(task_stats):
    """Compute mean AUC across tasks with CI."""
    aucs = [s['mean'] for s in task_stats]
    stds = [s['std'] for s in task_stats]
    mean_auc = np.mean(aucs)
    se_mean = np.sqrt(np.sum(np.array(stds)**2)) / len(task_stats)
    return {
        'mean': mean_auc, 'std': se_mean,
        'p025': mean_auc - 1.96 * se_mean, 'p05': mean_auc - 1.645 * se_mean,
        'p95': mean_auc + 1.645 * se_mean, 'p975': mean_auc + 1.96 * se_mean,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Mode: {'Full Fine-Tune' if args.full_fine_tune else 'Frozen Embeddings'}")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)

    # Filter to rows with both young and senescence values
    df = df.dropna(subset=['young', 'senescence'])

    train_df = df[df['scaffold_split'] == 'train'].reset_index(drop=True)
    test_df = df[df['scaffold_split'] == 'test'].reset_index(drop=True)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Create labels
    train_labels = create_labels(train_df)
    test_labels = create_labels(test_df)

    for name, y in train_labels.items():
        print(f"  {name}: train {y.sum()}/{len(y)} ({100*y.mean():.1f}%), "
              f"test {test_labels[name].sum()}/{len(test_labels[name])} ({100*test_labels[name].mean():.1f}%)")

    # Load encoder/tokenizer
    encoder, tokenizer = None, None
    if args.model not in ['molformer', 'chemberta', 'chemberta-mlm']:
        tokenizer = Tokenizer.from_file('tokenizers/smiles_tokenizer_simple/tokenizer.json')
        encoder = load_encoder(args, tokenizer, device)
        print("Encoder loaded successfully")

    if args.full_fine_tune:
        if args.model in ['molformer', 'chemberta', 'chemberta-mlm']:
            raise ValueError("Full fine-tuning not supported for baseline models")

        train_smiles = train_df['SMILES'].tolist()
        test_smiles = test_df['SMILES'].tolist()

        print(f"\n--- Multi-task Fine-tuning (all 3 outcomes simultaneously) ---")
        print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")

        # Fine-tune on all 3 tasks
        model = fine_tune_multitask(encoder, train_smiles, train_labels,
                                    test_smiles, test_labels, tokenizer, args, device)

        # Final predictions
        test_preds = predict_multitask(model, test_smiles, tokenizer, args, device)  # [N, 3]

        print(f"\n{'='*60}")
        print("FINAL RESULTS (Full Fine-Tune)")
        print(f"{'='*60}")
        print(f"{'Outcome':<15} {'AUC (95% CI)':<35}")
        print("-" * 60)

        all_ft_stats = []
        for i, outcome in enumerate(['kills_young', 'kills_sen', 'senolytic']):
            y_test = test_labels[outcome]
            ft_stats = auc_ci_normal_approx(y_test, test_preds[:, i])
            all_ft_stats.append(ft_stats)
            print(f"{outcome:<15} {format_ci(ft_stats, '95'):<35}")

        # Mean across outcomes
        print("-" * 60)
        ft_mean = mean_auc_ci(all_ft_stats)
        print(f"{'MEAN (95% CI)':<15} {format_ci(ft_mean, '95'):<35}")

        del model
        torch.cuda.empty_cache()

    else:
        # Frozen embeddings mode
        print("\nComputing embeddings...")
        if args.model in ['molformer', 'chemberta', 'chemberta-mlm']:
            from utils.deepchem_embeddings import get_deepchem_embeddings
            train_emb = get_deepchem_embeddings(train_df['SMILES'].tolist(), args.model, args.batch_size, device)
            test_emb = get_deepchem_embeddings(test_df['SMILES'].tolist(), args.model, args.batch_size, device)
        else:
            train_emb = get_embeddings(train_df['SMILES'].tolist(), encoder, tokenizer, args, device)
            test_emb = get_embeddings(test_df['SMILES'].tolist(), encoder, tokenizer, args, device)

        print(f"\n{'='*90}")
        print("SENOLYTIC SCREEN RESULTS (Frozen Embeddings)")
        print(f"{'='*90}")
        print(f"{'Outcome':<15} {'Linear AUC (95% CI)':<30} {'XGBoost AUC (95% CI)':<30}")
        print("-" * 90)

        all_lr_stats, all_xgb_stats = [], []
        for outcome in ['kills_young', 'kills_sen', 'senolytic']:
            y_train = train_labels[outcome]
            y_test = test_labels[outcome]

            lr_stats, xgb_stats = evaluate_task(train_emb, y_train, test_emb, y_test, args)
            all_lr_stats.append(lr_stats)
            all_xgb_stats.append(xgb_stats)

            print(f"{outcome:<15} {format_ci(lr_stats, '95'):<30} {format_ci(xgb_stats, '95'):<30}")

        # Mean across outcomes
        print("-" * 90)
        lr_mean = mean_auc_ci(all_lr_stats)
        xgb_mean = mean_auc_ci(all_xgb_stats)
        print(f"{'MEAN (95% CI)':<15} {format_ci(lr_mean, '95'):<30} {format_ci(xgb_mean, '95'):<30}")
        print(f"{'MEAN (90% CI)':<15} {format_ci(lr_mean, '90'):<30} {format_ci(xgb_mean, '90'):<30}")


if __name__ == "__main__":
    main()
