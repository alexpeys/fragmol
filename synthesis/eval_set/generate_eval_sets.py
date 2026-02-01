import pandas as pd

zinc_path = "s3://shvaibackups/unibio_data/zinc22/shuffled/0000.parquet"
df = pd.read_parquet(zinc_path, columns=['smiles'])
sample = df.sample(n=1000, random_state=1337)
sample.to_parquet("zinc22_1000_evals.parquet", index=False)

import pandas as pd

pubchem = 's3://shvaibackups/unibio_data/pubchem/processed/0.parquet'
df = pd.read_parquet(pubchem, columns=['smiles'])
df = df[df['smiles'].str.len() <= 150]
sample = df.sample(n=1000, random_state=1337)
sample.to_parquet("pubchem_1000_evals.parquet", index=False)