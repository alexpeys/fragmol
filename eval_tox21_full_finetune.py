import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

from utils.models import Smile2SmileEncoderWithJEPA, LlamaConfig
from tokenizers import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Full finetune on Tox21")
    parser.add_argument("--model_path", type=str,
                        default="s3://shvaibackups/mol_porque/molporque_10layer_jepa0_contrastive1_decode1/40001.pt")
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--tox_21_path", type=str,
                        default="/home/ubuntu/chemberta3/chemberta3_benchmarking/data/datasets/deepchem_splits/tox21/")
    parser.add_argument("--max_mol_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


class Tox21Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len, task_cols):
        self.smiles = df['smiles'].tolist()
        self.labels = df[task_cols].values.astype(np.float32)  # (N, num_tasks)
        self.mask = ~np.isnan(self.labels)  # True where valid
        self.labels = np.nan_to_num(self.labels, nan=0.0)  # Replace NaN with 0 for tensor
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        ids = self.tokenizer.encode(f'[BOS]{smiles}[EOS]').ids
        seq_len = len(ids)
        if seq_len > self.max_len:
            ids = ids[:self.max_len]
            seq_len = self.max_len
        padding_len = self.max_len - seq_len
        padded_ids = ids + [self.pad_token_id] * padding_len
        attention_mask = [1] * seq_len + [0] * padding_len
        
        return {
            'input_ids': torch.tensor(padded_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32),
            'mask': torch.tensor(self.mask[idx], dtype=torch.bool),
        }


class Tox21Model(nn.Module):
    def __init__(self, encoder, hidden_size, num_tasks):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_size, num_tasks)
    
    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = output['hidden_state'][:, 0, :]  # CLS token
        logits = self.head(cls_emb)
        return logits


def load_encoder(args, tokenizer, device):
    """Load the model and extract only the encoder."""
    vocab_size = max(tokenizer.get_vocab().values()) + 1
    
    encoder_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.emb_dim,
        intermediate_size=int(args.emb_dim * args.intermediate_size_multiplier),
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        pad_token_id=tokenizer.token_to_id('[PAD]'),
        bos_token_id=tokenizer.token_to_id('[BOS]'),
        eos_token_id=tokenizer.token_to_id('[EOS]'),
        cls_token_id=tokenizer.token_to_id('[BOS]'),
        max_position_embeddings=args.max_mol_size,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        rope_base=10_000,
        use_rope=True,
    )
    
    decoder_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.emb_dim,
        intermediate_size=int(args.emb_dim * args.intermediate_size_multiplier),
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        pad_token_id=tokenizer.token_to_id('[PAD]'),
        bos_token_id=tokenizer.token_to_id('[BOS]'),
        eos_token_id=tokenizer.token_to_id('[EOS]'),
        cls_token_id=tokenizer.token_to_id('[BOS]'),
        max_position_embeddings=args.max_mol_size,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        rope_base=10_000,
        use_rope=True,
    )
    
    model = Smile2SmileEncoderWithJEPA(encoder_config, decoder_config)
    
    print(f"Loading checkpoint from: {args.model_path}")
    if args.model_path.startswith('s3://'):
        from s3torchconnector import S3Checkpoint
        with S3Checkpoint("us-west-2").reader(args.model_path) as reader:
            checkpoint = torch.load(reader, map_location='cpu')
    else:
        checkpoint = torch.load(args.model_path, map_location='cpu')
    
    if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
        print("Removing _orig_mod. prefix from checkpoint...")
        checkpoint = {k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k: v
                      for k, v in checkpoint.items()}
    
    model.load_state_dict(checkpoint)
    del checkpoint
    
    return model.encoder


def compute_test_aucs(model, test_loader, task_cols, device):
    """Compute per-task AUCs on test set."""
    model.eval()
    all_logits = []
    all_labels = []
    all_masks = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu())
            all_labels.append(batch['labels'])
            all_masks.append(batch['mask'])
    
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_masks = torch.cat(all_masks, dim=0).numpy()

    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid

    aucs = []
    for i, task in enumerate(task_cols):
        mask = all_masks[:, i]
        if mask.sum() < 10:
            continue
        y_true = all_labels[mask, i]
        y_pred = probs[mask, i]
        if len(np.unique(y_true)) < 2:
            continue
        try:
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
        except:
            pass

    return aucs


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = Tokenizer.from_file('tokenizers/smiles_tokenizer_simple/tokenizer.json')

    # Load data
    train_df = pd.read_csv(os.path.join(args.tox_21_path, 'train_cleaned.csv'))
    val_df = pd.read_csv(os.path.join(args.tox_21_path, 'valid_cleaned.csv'))
    test_df = pd.read_csv(os.path.join(args.tox_21_path, 'test_cleaned.csv'))

    task_cols = [col for col in train_df.columns if col != 'smiles']
    num_tasks = len(task_cols)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Tasks ({num_tasks}): {task_cols}")

    # Create datasets
    train_dataset = Tox21Dataset(train_df, tokenizer, args.max_mol_size, task_cols)
    val_dataset = Tox21Dataset(val_df, tokenizer, args.max_mol_size, task_cols)
    test_dataset = Tox21Dataset(test_df, tokenizer, args.max_mol_size, task_cols)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load encoder and create model
    encoder = load_encoder(args, tokenizer, device)
    model = Tox21Model(encoder, args.emb_dim, num_tasks).to(device)
    print("Model loaded successfully")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                loss_per_sample = criterion(logits, labels)  # (B, num_tasks)
                # Mask out NaN labels
                loss_per_sample = loss_per_sample * mask.float()
                # Average over valid labels only
                num_valid = mask.sum()
                if num_valid > 0:
                    loss = loss_per_sample.sum() / num_valid
                else:
                    loss = loss_per_sample.sum() * 0  # Zero loss if no valid labels

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            print(f"Epoch {epoch+1}/{args.num_epochs} | Step {global_step} | Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / num_batches

        # Evaluate on test set
        test_aucs = compute_test_aucs(model, test_loader, task_cols, device)
        test_mean = np.mean(test_aucs)
        test_se = np.std(test_aucs, ddof=1) / np.sqrt(len(test_aucs))

        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} COMPLETE | Avg Loss: {avg_epoch_loss:.4f}")
        print(f"TEST AVG AUC: {test_mean:.4f} +/- {test_se:.4f}")
        print(f"{'='*60}\n")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    final_aucs = compute_test_aucs(model, test_loader, task_cols, device)
    final_mean = np.mean(final_aucs)
    final_se = np.std(final_aucs, ddof=1) / np.sqrt(len(final_aucs))
    print(f"FULL FINETUNE AVG AUC: {final_mean:.4f} +/- {final_se:.4f}")


if __name__ == "__main__":
    main()

