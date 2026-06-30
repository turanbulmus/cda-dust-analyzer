import huggingface_hub
import pandas as pd
import os

REPO_ID = "CosmicDustGroup/cassini-cda-spectra"
FILENAME_TRAIN = "data/lvl2/cda_qm_spectra_pre2008277_train_lvl2.parquet"

print("Downloading dataset...")
file_train_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME_TRAIN, repo_type="dataset")
print("Loading dataset...")
df = pd.read_parquet(file_train_path)

print("Columns:", df.columns)
print("Unique classes:", df['class'].unique().tolist())
print("Value counts:\n", df['class'].value_counts())
