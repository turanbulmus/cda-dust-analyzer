import json
import pandas as pd

cache_file = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
labels = []
with open(cache_file, 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            labels.append(data.get('label'))

print("Cached examples label counts:")
print(pd.Series(labels).value_counts())

train_file = "cda_dust_agent/data/raw/cda_train.parquet"
df = pd.read_parquet(train_file)
print("\nRaw train label counts:")
print(df['class'].value_counts(dropna=False))

df['consolidated'] = df['class'].apply(lambda x: str(x).split('-')[0].strip() if pd.notna(x) else 'Noise')
print("\nConsolidated train label counts:")
print(df['consolidated'].value_counts(dropna=False))
