import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import xgboost as xgb

from utils.models import Smile2SmileEncoderWithJEPA, LlamaConfig
from tokenizers import Tokenizer


# Classification datasets
CLASSIFICATION_DATASETS = {
    'tox21': {'label_cols': None, 'multitask': True},
    'hiv': {'label_cols': ['HIV_active'], 'multitask': False},
    'bbbp': {'label_cols': ['p_np'], 'multitask': False},
    'bace_classification': {'label_cols': ['Class'], 'multitask': False},
    'clintox': {'label_cols': None, 'multitask': True},
    'sider': {'label_cols': None, 'multitask': True},
}

# Regression datasets
REGRESSION_DATASETS = {
    'bace_regression': {'label_cols': ['pIC50'], 'multitask': False},
    'clearance': {'label_cols': ['target'], 'multitask': False},
    'lipo': {'label_cols': ['exp'], 'multitask': False},
    'freesolv': {'label_cols': ['y'], 'multitask': False},
    'delaney': {'label_cols': ['measured log solubility in mols per litre'], 'multitask': False},
}

ALL_DATASETS = {**CLASSIFICATION_DATASETS, **REGRESSION_DATASETS}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on MolNet benchmarks")
    parser.add_argument("--model", type=str,
                        default='s3://shvaibackups/mol_porque/molporque_10layer_jepa0_contrastive1_decode1/40001.pt',
                        help="Model: 's3://...' or local path for our encoder, or 'molformer'/'chemberta'/'chemberta-mlm'")
    parser.add_argument("--data_path", type=str,
                        default="/home/ubuntu/chemberta3/chemberta3_benchmarking/data/datasets/deepchem_splits/")
    parser.add_argument("--datasets", type=str, nargs='+', default=list(ALL_DATASETS.keys()),
                        choices=list(ALL_DATASETS.keys()), help="Datasets to evaluate")
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--intermediate_size_multiplier", type=float, default=4)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--max_mol_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--ci_type", type=str, default='normal_approx', choices=['normal_approx', 'bootstrap'],
                        help="CI method: 'normal_approx' or 'bootstrap'")
    parser.add_argument("--seed", type=int, default=42)
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
    n1 = np.sum(y_true == 1)  # positive cases
    n2 = np.sum(y_true == 0)  # negative cases

    # Hanley-McNeil approximation
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)) / (n1 * n2))

    # Z-scores: 1.96 for 95%, 1.645 for 90%
    return {
        'mean': auc,
        'std': se,
        'p025': auc - 1.96 * se,
        'p05': auc - 1.645 * se,
        'p95': auc + 1.645 * se,
        'p975': auc + 1.96 * se,
    }


def auc_ci_bootstrap(y_true, y_pred, n_bootstrap=1000, seed=42):
    """Bootstrap AUC to get confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    aucs = np.array(aucs)
    return {
        'mean': roc_auc_score(y_true, y_pred),  # Use actual AUC, not bootstrap mean
        'std': np.std(aucs),
        'p025': np.percentile(aucs, 2.5),
        'p05': np.percentile(aucs, 5),
        'p95': np.percentile(aucs, 95),
        'p975': np.percentile(aucs, 97.5),
    }


def compute_auc_ci(y_true, y_pred, ci_type='normal_approx', n_bootstrap=1000, seed=42):
    """Compute AUC with confidence intervals."""
    if ci_type == 'bootstrap':
        return auc_ci_bootstrap(y_true, y_pred, n_bootstrap, seed)
    else:
        return auc_ci_normal_approx(y_true, y_pred)


# ============== Correlation CI (for regression) ==============

def corr_ci_normal_approx(y_true, y_pred):
    """Compute Pearson correlation with CI using Fisher's z-transformation."""
    r, _ = pearsonr(y_true, y_pred)
    n = len(y_true)

    # Fisher's z-transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    se_z = 1.0 / np.sqrt(n - 3)

    # CI in z-space, then transform back
    z_lower_95 = z - 1.96 * se_z
    z_upper_95 = z + 1.96 * se_z
    z_lower_90 = z - 1.645 * se_z
    z_upper_90 = z + 1.645 * se_z

    # Transform back to r
    def z_to_r(z_val):
        return (np.exp(2 * z_val) - 1) / (np.exp(2 * z_val) + 1)

    return {
        'mean': r,
        'std': se_z,  # SE in z-space (approximate)
        'p025': z_to_r(z_lower_95),
        'p05': z_to_r(z_lower_90),
        'p95': z_to_r(z_upper_90),
        'p975': z_to_r(z_upper_95),
    }


def corr_ci_bootstrap(y_true, y_pred, n_bootstrap=1000, seed=42):
    """Bootstrap correlation to get confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    corrs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        r, _ = pearsonr(y_true[idx], y_pred[idx])
        corrs.append(r)
    corrs = np.array(corrs)
    r_actual, _ = pearsonr(y_true, y_pred)
    return {
        'mean': r_actual,
        'std': np.std(corrs),
        'p025': np.percentile(corrs, 2.5),
        'p05': np.percentile(corrs, 5),
        'p95': np.percentile(corrs, 95),
        'p975': np.percentile(corrs, 97.5),
    }


def compute_corr_ci(y_true, y_pred, ci_type='normal_approx', n_bootstrap=1000, seed=42):
    """Compute correlation with confidence intervals."""
    if ci_type == 'bootstrap':
        return corr_ci_bootstrap(y_true, y_pred, n_bootstrap, seed)
    else:
        return corr_ci_normal_approx(y_true, y_pred)


# ============== Classification evaluation ==============

def evaluate_classification_task(train_emb, train_labels, val_emb, val_labels, test_emb, test_labels, args):
    """Train linear and XGBoost classifiers, return test predictions."""
    # Filter NaN
    train_mask = ~np.isnan(train_labels)
    val_mask = ~np.isnan(val_labels)
    test_mask = ~np.isnan(test_labels)

    X_train, y_train = train_emb[train_mask], train_labels[train_mask].astype(int)
    X_val, y_val = val_emb[val_mask], val_labels[val_mask].astype(int)
    X_test, y_test = test_emb[test_mask], test_labels[test_mask].astype(int)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None, None, None

    # Combine train+val for linear model with CV
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    # Linear
    lr = LogisticRegressionCV(cv=5, max_iter=1000, random_state=args.seed, solver='lbfgs', n_jobs=-1)
    lr.fit(X_train_val, y_train_val)
    lr_preds = lr.predict_proba(X_test)[:, 1]

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=args.seed,
                                   eval_metric='auc', verbosity=0)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]

    return y_test, lr_preds, xgb_preds


# ============== Regression evaluation ==============

def evaluate_regression_task(train_emb, train_labels, val_emb, val_labels, test_emb, test_labels, args):
    """Train linear and XGBoost regressors, return test predictions."""
    # Filter NaN
    train_mask = ~np.isnan(train_labels)
    val_mask = ~np.isnan(val_labels)
    test_mask = ~np.isnan(test_labels)

    X_train, y_train = train_emb[train_mask], train_labels[train_mask]
    X_val, y_val = val_emb[val_mask], val_labels[val_mask]
    X_test, y_test = test_emb[test_mask], test_labels[test_mask]

    if len(y_test) < 3:
        return None, None, None

    # Combine train+val for linear model with CV
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    # Linear (Ridge regression with CV)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X_train_val, y_train_val)
    ridge_preds = ridge.predict(X_test)

    # XGBoost regressor
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=args.seed, verbosity=0)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = xgb_model.predict(X_test)

    return y_test, ridge_preds, xgb_preds


# ============== Dataset evaluation ==============

def get_embeddings_for_dataset(train_df, val_df, test_df, args, encoder, tokenizer, device):
    """Get embeddings for train/val/test splits."""
    if args.model in ['molformer', 'chemberta', 'chemberta-mlm']:
        from utils.deepchem_embeddings import get_deepchem_embeddings
        train_emb = get_deepchem_embeddings(train_df['smiles'].tolist(), args.model, args.batch_size, device)
        val_emb = get_deepchem_embeddings(val_df['smiles'].tolist(), args.model, args.batch_size, device)
        test_emb = get_deepchem_embeddings(test_df['smiles'].tolist(), args.model, args.batch_size, device)
    else:
        train_emb = get_embeddings(train_df['smiles'].tolist(), encoder, tokenizer, args, device)
        val_emb = get_embeddings(val_df['smiles'].tolist(), encoder, tokenizer, args, device)
        test_emb = get_embeddings(test_df['smiles'].tolist(), encoder, tokenizer, args, device)
    return train_emb, val_emb, test_emb


def evaluate_classification_dataset(dataset_name, args, encoder, tokenizer, device):
    """Evaluate a classification dataset."""
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION: {dataset_name.upper()}")
    print(f"{'='*70}")

    data_dir = os.path.join(args.data_path, dataset_name)
    train_df = pd.read_csv(os.path.join(data_dir, 'train_cleaned.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'valid_cleaned.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_cleaned.csv'))
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    config = CLASSIFICATION_DATASETS[dataset_name]
    if config['label_cols'] is None:
        task_cols = [c for c in train_df.columns if c != 'smiles']
    else:
        task_cols = config['label_cols']
    print(f"Tasks ({len(task_cols)}): {task_cols[:5]}{'...' if len(task_cols) > 5 else ''}")

    train_emb, val_emb, test_emb = get_embeddings_for_dataset(
        train_df, val_df, test_df, args, encoder, tokenizer, device)

    results = {'tasks': {}, 'linear_scores': [], 'xgb_scores': [], 'task_type': 'classification'}

    for task in tqdm(task_cols, desc="Tasks"):
        y_test, lr_preds, xgb_preds = evaluate_classification_task(
            train_emb, train_df[task].values,
            val_emb, val_df[task].values,
            test_emb, test_df[task].values, args
        )
        if y_test is None:
            continue

        lr_stats = compute_auc_ci(y_test, lr_preds, args.ci_type, args.n_bootstrap, args.seed)
        xgb_stats = compute_auc_ci(y_test, xgb_preds, args.ci_type, args.n_bootstrap, args.seed)

        results['tasks'][task] = {'linear': lr_stats, 'xgb': xgb_stats}
        results['linear_scores'].append(lr_stats['mean'])
        results['xgb_scores'].append(xgb_stats['mean'])

    return results


def evaluate_regression_dataset(dataset_name, args, encoder, tokenizer, device):
    """Evaluate a regression dataset."""
    print(f"\n{'='*70}")
    print(f"REGRESSION: {dataset_name.upper()}")
    print(f"{'='*70}")

    data_dir = os.path.join(args.data_path, dataset_name)
    train_df = pd.read_csv(os.path.join(data_dir, 'train_cleaned.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'valid_cleaned.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_cleaned.csv'))
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    config = REGRESSION_DATASETS[dataset_name]
    task_cols = config['label_cols']
    print(f"Tasks: {task_cols}")

    train_emb, val_emb, test_emb = get_embeddings_for_dataset(
        train_df, val_df, test_df, args, encoder, tokenizer, device)

    results = {'tasks': {}, 'linear_scores': [], 'xgb_scores': [], 'task_type': 'regression'}

    for task in tqdm(task_cols, desc="Tasks"):
        y_test, ridge_preds, xgb_preds = evaluate_regression_task(
            train_emb, train_df[task].values,
            val_emb, val_df[task].values,
            test_emb, test_df[task].values, args
        )
        if y_test is None:
            continue

        ridge_stats = compute_corr_ci(y_test, ridge_preds, args.ci_type, args.n_bootstrap, args.seed)
        xgb_stats = compute_corr_ci(y_test, xgb_preds, args.ci_type, args.n_bootstrap, args.seed)

        results['tasks'][task] = {'linear': ridge_stats, 'xgb': xgb_stats}
        results['linear_scores'].append(ridge_stats['mean'])
        results['xgb_scores'].append(xgb_stats['mean'])

    return results


def mean_score_ci(task_results, model_type, ci_type='normal_approx', n_bootstrap=1000, seed=42):
    """Compute mean score across tasks with CI."""
    tasks = list(task_results.keys())
    scores = [task_results[task][model_type]['mean'] for task in tasks]
    stds = [task_results[task][model_type]['std'] for task in tasks]
    mean_score = np.mean(scores)

    if ci_type == 'normal_approx':
        # Propagate uncertainty: SE of mean = sqrt(sum(se_i^2)) / n
        se_mean = np.sqrt(np.sum(np.array(stds)**2)) / len(tasks)
        return {
            'mean': mean_score,
            'std': se_mean,
            'p025': mean_score - 1.96 * se_mean,
            'p05': mean_score - 1.645 * se_mean,
            'p95': mean_score + 1.645 * se_mean,
            'p975': mean_score + 1.96 * se_mean,
        }
    else:
        # Bootstrap across tasks
        rng = np.random.RandomState(seed)
        n_tasks = len(tasks)
        mean_scores = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n_tasks, n_tasks)
            mean_scores.append(np.mean([scores[i] for i in idx]))
        mean_scores = np.array(mean_scores)
        return {
            'mean': mean_score,
            'std': np.std(mean_scores),
            'p025': np.percentile(mean_scores, 2.5),
            'p05': np.percentile(mean_scores, 5),
            'p95': np.percentile(mean_scores, 95),
            'p975': np.percentile(mean_scores, 97.5),
        }


def format_ci(stats, ci_level='95'):
    """Format stats as 'mean [lower-upper]'."""
    if ci_level == '95':
        return f"{stats['mean']:.3f} [{stats['p025']:.3f}-{stats['p975']:.3f}]"
    else:  # 90%
        return f"{stats['mean']:.3f} [{stats['p05']:.3f}-{stats['p95']:.3f}]"


def print_results(results, dataset_name, ci_type, n_bootstrap, seed):
    """Print formatted results for a dataset."""
    task_type = results.get('task_type', 'classification')
    metric = 'AUC' if task_type == 'classification' else 'Corr'

    print(f"\n--- {dataset_name.upper()} Results ({task_type}) ---")
    print(f"{'Task':<40} {'Linear '+metric+' (95% CI)':<28} {'XGB '+metric+' (95% CI)':<28}")
    print("-" * 96)

    for task, stats in results['tasks'].items():
        task_short = task[:37] + '...' if len(task) > 40 else task
        lr_str = format_ci(stats['linear'], '95')
        xgb_str = format_ci(stats['xgb'], '95')
        print(f"{task_short:<40} {lr_str:<28} {xgb_str:<28}")

    # Mean across tasks with both 90% and 95% CI
    if len(results['tasks']) > 1:
        lr_mean = mean_score_ci(results['tasks'], 'linear', ci_type, n_bootstrap, seed)
        xgb_mean = mean_score_ci(results['tasks'], 'xgb', ci_type, n_bootstrap, seed)

        print("-" * 96)
        print(f"{'MEAN (95% CI)':<40} {format_ci(lr_mean, '95'):<28} {format_ci(xgb_mean, '95'):<28}")
        print(f"{'MEAN (90% CI)':<40} {format_ci(lr_mean, '90'):<28} {format_ci(xgb_mean, '90'):<28}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"CI type: {args.ci_type}")

    # Load encoder if using our model
    encoder, tokenizer = None, None
    if args.model not in ['molformer', 'chemberta', 'chemberta-mlm']:
        tokenizer = Tokenizer.from_file('tokenizers/smiles_tokenizer_simple/tokenizer.json')
        encoder = load_encoder(args, tokenizer, device)
        print("Encoder loaded successfully")

    # Separate classification and regression datasets
    clf_datasets = [d for d in args.datasets if d in CLASSIFICATION_DATASETS]
    reg_datasets = [d for d in args.datasets if d in REGRESSION_DATASETS]

    clf_results, reg_results = {}, {}

    # Classification
    for dataset_name in clf_datasets:
        results = evaluate_classification_dataset(dataset_name, args, encoder, tokenizer, device)
        clf_results[dataset_name] = results
        print_results(results, dataset_name, args.ci_type, args.n_bootstrap, args.seed)

    # Regression
    for dataset_name in reg_datasets:
        results = evaluate_regression_dataset(dataset_name, args, encoder, tokenizer, device)
        reg_results[dataset_name] = results
        print_results(results, dataset_name, args.ci_type, args.n_bootstrap, args.seed)

    # Final summaries
    def print_summary(all_results, title, metric):
        if not all_results:
            return
        print(f"\n{'='*100}")
        print(f"{title} SUMMARY")
        print(f"{'='*100}")
        print(f"{'Dataset':<25} {'Linear '+metric+' (95% CI)':<35} {'XGB '+metric+' (95% CI)':<35}")
        print("-" * 100)

        all_lr_stats, all_xgb_stats = [], []
        for dataset_name, results in all_results.items():
            if len(results['tasks']) > 1:
                lr_stats = mean_score_ci(results['tasks'], 'linear', args.ci_type, args.n_bootstrap, args.seed)
                xgb_stats = mean_score_ci(results['tasks'], 'xgb', args.ci_type, args.n_bootstrap, args.seed)
            else:
                task = list(results['tasks'].keys())[0]
                lr_stats = results['tasks'][task]['linear']
                xgb_stats = results['tasks'][task]['xgb']

            all_lr_stats.append(lr_stats)
            all_xgb_stats.append(xgb_stats)
            print(f"{dataset_name:<25} {format_ci(lr_stats, '95'):<35} {format_ci(xgb_stats, '95'):<35}")

        # Overall mean across datasets
        if len(all_results) > 1:
            print("-" * 100)
            overall_lr_mean = np.mean([s['mean'] for s in all_lr_stats])
            overall_xgb_mean = np.mean([s['mean'] for s in all_xgb_stats])
            overall_lr_se = np.sqrt(np.sum([s['std']**2 for s in all_lr_stats])) / len(all_lr_stats)
            overall_xgb_se = np.sqrt(np.sum([s['std']**2 for s in all_xgb_stats])) / len(all_xgb_stats)

            overall_lr = {'mean': overall_lr_mean, 'std': overall_lr_se,
                          'p025': overall_lr_mean - 1.96*overall_lr_se, 'p05': overall_lr_mean - 1.645*overall_lr_se,
                          'p95': overall_lr_mean + 1.645*overall_lr_se, 'p975': overall_lr_mean + 1.96*overall_lr_se}
            overall_xgb = {'mean': overall_xgb_mean, 'std': overall_xgb_se,
                           'p025': overall_xgb_mean - 1.96*overall_xgb_se, 'p05': overall_xgb_mean - 1.645*overall_xgb_se,
                           'p95': overall_xgb_mean + 1.645*overall_xgb_se, 'p975': overall_xgb_mean + 1.96*overall_xgb_se}

            print(f"{'MEAN (95% CI)':<25} {format_ci(overall_lr, '95'):<35} {format_ci(overall_xgb, '95'):<35}")
            print(f"{'MEAN (90% CI)':<25} {format_ci(overall_lr, '90'):<35} {format_ci(overall_xgb, '90'):<35}")

    print_summary(clf_results, "CLASSIFICATION", "AUC")
    print_summary(reg_results, "REGRESSION", "Corr")


if __name__ == "__main__":
    main()

