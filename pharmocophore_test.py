import pandas as pd
import numpy as np
from synthesis.dataloader import get_pharmacophore
import time

print('Loading data...', flush=True)
df = pd.read_parquet('s3://shvaibackups/unibio_data/zinc22/shuffled/0000.parquet', columns=['smiles'])
print(f'Loaded {len(df)} smiles', flush=True)

sample = df['smiles'].sample(200, random_state=42).tolist()
lengths = []
times = []
failed = 0

for i, smi in enumerate(sample):
    print(f'{i}/10000... (failed so far: {failed})', flush=True)
    
    t0 = time.time()
    result = get_pharmacophore(smi, max_len=128)
    elapsed = time.time() - t0
    
    if result:
        lengths.append(len(result['pharmacophore_types']))
        times.append(elapsed)
    else:
        failed += 1

lengths = np.array(lengths)
times = np.array(times)

print(f'\\n=== RESULTS ===', flush=True)
print(f'Failed: {failed}/{len(sample)} ({100*failed/len(sample):.1f}%)', flush=True)
print(f'\\n--- Pharmacophore Lengths ---', flush=True)
print(f'Min: {np.min(lengths)}, Max: {np.max(lengths)}', flush=True)
print(f'Mean: {np.mean(lengths):.1f}, Median: {np.median(lengths):.1f}', flush=True)
print(f'P90: {np.percentile(lengths, 90):.0f}', flush=True)
print(f'P95: {np.percentile(lengths, 95):.0f}', flush=True)
print(f'P99: {np.percentile(lengths, 99):.0f}', flush=True)

print(f'\\n--- Conformer Calculation Times (seconds) ---', flush=True)
print(f'Min: {np.min(times):.3f}, Max: {np.max(times):.3f}', flush=True)
print(f'Mean: {np.mean(times):.3f}, Median: {np.median(times):.3f}', flush=True)
print(f'P90: {np.percentile(times, 90):.3f}', flush=True)
print(f'P99: {np.percentile(times, 99):.3f}', flush=True)
print(f'Total time: {np.sum(times):.1f}s', flush=True)
