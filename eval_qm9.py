import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

from utils.models import Smile2SmileVAE, Smile2SmileEncoderWithJEPA, LlamaConfig
from tokenizers import Tokenizer


# QM9 target properties (all regression)
QM9_TARGETS = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
               'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on QM9 benchmark")
    parser.add_argument("--model", type=str,
                        default='s3://shvaibackups/mol_porque/e768_l20_h12_decode1_constrast1/40001.pt',
                        help="Model: 's3://...' or local path, or 'molformer'/'chemberta'/'chemberta-mlm'")
    parser.add_argument("--data_path", type=str, default="data/qm9/")
    parser.add_argument("--targets", type=str, nargs='+', default=QM9_TARGETS,
                        choices=QM9_TARGETS, help="Target properties to evaluate")
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4)
    parser.add_argument("--num_layers", type=int, default=20)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--max_mol_size", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--noise_scale", type=float, default=0.0, help="Noise scale for VAE encoding (0 = deterministic)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_file", type=str, default=None, help="Output CSV file for results (optional)")
    return parser.parse_args()


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
                # latent is [bs, 1, hidden_size], squeeze the middle dim
                embeddings.append(encode_out['latent'].squeeze(1).float().cpu().numpy())
            else:
                output = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(output['hidden_state'][:, 0, :].float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def corr_ci_normal_approx(y_true, y_pred):
    """Compute Pearson correlation with CI using Fisher's z-transformation."""
    r, _ = pearsonr(y_true, y_pred)
    n = len(y_true)
    z = 0.5 * np.log((1 + r) / (1 - r))
    se_z = 1.0 / np.sqrt(n - 3)

    z_lower_95, z_upper_95 = z - 1.96 * se_z, z + 1.96 * se_z
    z_lower_90, z_upper_90 = z - 1.645 * se_z, z + 1.645 * se_z

    def z_to_r(z_val):
        return (np.exp(2 * z_val) - 1) / (np.exp(2 * z_val) + 1)

    return {
        'mean': r, 'std': se_z,
        'p025': z_to_r(z_lower_95), 'p05': z_to_r(z_lower_90),
        'p95': z_to_r(z_upper_90), 'p975': z_to_r(z_upper_95),
    }


def format_ci(stats, ci_level='95'):
    if ci_level == '95':
        return f"{stats['mean']:.3f} [{stats['p025']:.3f}-{stats['p975']:.3f}]"
    return f"{stats['mean']:.3f} [{stats['p05']:.3f}-{stats['p95']:.3f}]"


def evaluate_target(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Ridge regression and return test correlation stats."""
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X_train_val, y_train_val)
    preds = ridge.predict(X_test)

    return corr_ci_normal_approx(y_test, preds)


def mean_corr_ci(task_stats):
    """Compute mean correlation across tasks with CI (propagated uncertainty)."""
    corrs = [s['mean'] for s in task_stats]
    stds = [s['std'] for s in task_stats]
    mean_corr = np.mean(corrs)
    se_mean = np.sqrt(np.sum(np.array(stds)**2)) / len(task_stats)
    return {
        'mean': mean_corr, 'std': se_mean,
        'p025': mean_corr - 1.96 * se_mean, 'p05': mean_corr - 1.645 * se_mean,
        'p95': mean_corr + 1.645 * se_mean, 'p975': mean_corr + 1.96 * se_mean,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.device}')
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Targets: {len(args.targets)} properties")

    # Load data
    print(f"\nLoading QM9 data from {args.data_path}...")
    train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(args.data_path, 'val.csv'))
    test_df = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load model or use baseline
    model, tokenizer, use_vae = None, None, False
    if args.model not in ['molformer', 'chemberta', 'chemberta-mlm']:
        tokenizer = Tokenizer.from_file('tokenizers/smiles_tokenizer_simple/tokenizer.json')
        model, use_vae = load_model(args, tokenizer, device)
        print("Model loaded successfully")

    # Get embeddings
    print("\nComputing embeddings...")
    if args.model in ['molformer', 'chemberta', 'chemberta-mlm']:
        from utils.deepchem_embeddings import get_deepchem_embeddings
        train_emb = get_deepchem_embeddings(train_df['smiles'].tolist(), args.model, args.batch_size, device)
        val_emb = get_deepchem_embeddings(val_df['smiles'].tolist(), args.model, args.batch_size, device)
        test_emb = get_deepchem_embeddings(test_df['smiles'].tolist(), args.model, args.batch_size, device)
    else:
        train_emb = get_embeddings(train_df['smiles'].tolist(), model, tokenizer, args, device, use_vae)
        val_emb = get_embeddings(val_df['smiles'].tolist(), model, tokenizer, args, device, use_vae)
        test_emb = get_embeddings(test_df['smiles'].tolist(), model, tokenizer, args, device, use_vae)

    # Evaluate each target
    print(f"\n{'='*70}")
    print("QM9 RESULTS (Ridge Regression)")
    print(f"{'='*70}")
    print(f"{'Target':<15} {'Correlation (95% CI)':<30} {'Correlation (90% CI)':<30}")
    print("-" * 75)

    all_stats = []
    for target in tqdm(args.targets, desc="Evaluating targets"):
        y_train = train_df[target].values
        y_val = val_df[target].values
        y_test = test_df[target].values

        stats = evaluate_target(train_emb, y_train, val_emb, y_val, test_emb, y_test)
        all_stats.append(stats)

        print(f"{target:<15} {format_ci(stats, '95'):<30} {format_ci(stats, '90'):<30}")

    # Mean across all targets
    print("-" * 75)
    mean_stats = mean_corr_ci(all_stats)
    print(f"{'MEAN':<15} {format_ci(mean_stats, '95'):<30} {format_ci(mean_stats, '90'):<30}")

    # Save results to CSV if --out_file is specified
    if args.out_file is not None:
        rows = []
        for target, stats in zip(args.targets, all_stats):
            rows.append({
                'dataset': 'qm9',
                'model': args.model,
                'metric_type': f"qm9_{target}",
                'metric_value': stats['mean'],
                'metric_90pct_ci': f"[{stats['p05']:.4f}-{stats['p95']:.4f}]",
                'metric_95pct_ci': f"[{stats['p025']:.4f}-{stats['p975']:.4f}]",
            })
        # Average row
        rows.append({
            'dataset': 'qm9',
            'model': args.model,
            'metric_type': 'qm9_avg',
            'metric_value': mean_stats['mean'],
            'metric_90pct_ci': f"[{mean_stats['p05']:.4f}-{mean_stats['p95']:.4f}]",
            'metric_95pct_ci': f"[{mean_stats['p025']:.4f}-{mean_stats['p975']:.4f}]",
        })
        out_df = pd.DataFrame(rows)
        out_dir = os.path.dirname(args.out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df.to_csv(args.out_file, index=False)
        print(f"\nResults saved to {args.out_file}")


if __name__ == "__main__":
    main()
