import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Load all data
models = {
    'MolPorQue (Ours)': {
        'molnet': pd.read_csv('mol_evals/e768_l20_h12_decode1_constrast1_molnet.csv'),
        'qm9': pd.read_csv('mol_evals/e768_l20_h12_decode1_constrast1_qm9.csv'),
        'seno': pd.read_csv('mol_evals/e768_l20_h12_decode1_constrast1_seno.csv'),
        'stat3': pd.read_csv('mol_evals/e768_l20_h12_decode1_constrast1_stat.csv'),
    },
    'MolFormer': {
        'molnet': pd.read_csv('mol_evals/molformer_molnet.csv'),
        'qm9': pd.read_csv('mol_evals/molformer_qm9.csv'),
        'seno': pd.read_csv('mol_evals/molformer_seno.csv'),
        'stat3': pd.read_csv('mol_evals/molformer_stat.csv'),
    },
    'ChemBERTa': {
        'molnet': pd.read_csv('mol_evals/chemberta_molnet.csv'),
        'qm9': pd.read_csv('mol_evals/chemberta_qm9.csv'),
        'seno': pd.read_csv('mol_evals/chemberta_seno.csv'),
        'stat3': pd.read_csv('mol_evals/chemberta_stat.csv') if os.path.exists('mol_evals/chemberta_stat.csv') else None,
    },
}

colors = {
    'MolPorQue (Ours)': '#1f77b4',
    'MolFormer': '#ff7f0e',
    'ChemBERTa': '#2ca02c',
}


def parse_ci(ci_str):
    """Parse '[0.1234-0.5678]' or '[-0.1234-0.5678]' into (lower, upper)."""
    # Handle negative numbers: [-0.5794--0.3731] means -0.5794 to -0.3731
    match = re.match(r'\[(-?[\d.]+)-(-?[\d.]+)\]', ci_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    # Handle case like [-0.5794--0.3731]
    match = re.match(r'\[(-?[\d.]+)--(-?[\d.]+)\]', ci_str)
    if match:
        return float(match.group(1)), -float(match.group(2))
    return None, None


def get_metric(df, metric_type):
    """Get metric value and 90% CI from dataframe."""
    row = df[df['metric_type'] == metric_type].iloc[0]
    value = row['metric_value']
    lower, upper = parse_ci(row['metric_90pct_ci'])
    return value, lower, upper


# Prepare data for plot
metrics = [
    ('MolNet Classification\nAverage AUC', 'molnet', 'classification_avg'),
    ('QM9 Average\nCorrelation', 'qm9', 'qm9_avg'),
    ('Internal Screen 1\nAUC', 'seno', 'seno_senolytic'),
    ('Internal Screen 2\nAUC', 'stat3', 'stat3_avg'),
]

fig, ax = plt.subplots(figsize=(7, 4))

n_metrics = len(metrics)
n_models = len(models)
width = 0.25
x = np.arange(n_metrics)

for i, (model_name, data) in enumerate(models.items()):
    values = []
    yerr_lower = []
    yerr_upper = []

    for metric_label, dataset_key, metric_type in metrics:
        df = data.get(dataset_key)
        if df is None:
            values.append(0)
            yerr_lower.append(0)
            yerr_upper.append(0)
        else:
            val, lower, upper = get_metric(df, metric_type)
            values.append(val)
            yerr_lower.append(val - lower if lower else 0)
            yerr_upper.append(upper - val if upper else 0)

    offset = (i - (n_models - 1) / 2) * width
    bars = ax.bar(x + offset, values, width,
                  yerr=[yerr_lower, yerr_upper],
                  label=model_name, color=colors[model_name],
                  capsize=3, error_kw={'linewidth': 1})

ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0.4, 0.9)
ax.set_xticks(x)
ax.set_xticklabels([m[0] for m in metrics], fontsize=10)
ax.set_title('Molecular Representation Evaluation', fontsize=14)

# Add grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Legend at bottom
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig('mol_evals/simple_plot.png', dpi=300, bbox_inches='tight')
print("Saved mol_evals/simple_plot.png")

plt.show()

