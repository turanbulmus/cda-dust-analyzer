"""
This script is a standalone utility to download datasets from Hugging Face.
Note that the agent can also do this automatically if `fetch_data=True` in the config.
"""
import os
import shutil
import huggingface_hub

REPO_ID = "CosmicDustGroup/cassini-cda-spectra"
FILENAME_TRAIN = "data/lvl2/cda_qm_spectra_pre2008277_train_lvl2.parquet"
FILENAME_INF = "data/lvl2/cda_qm_spectra_pre2008277_inf_lvl2.parquet"

TRAIN_OUT_PATH = "cda_dust_agent/data/raw/cda_train.parquet"
INF_OUT_PATH = "cda_dust_agent/data/raw/cda_inf.parquet"

def main():
    print("Downloading datasets from Hugging Face...")
    
    # Download files
    file_train_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME_TRAIN, repo_type="dataset")
    file_inf_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME_INF, repo_type="dataset")
    
    print(f"Downloaded train file to: {file_train_path}")
    print(f"Downloaded inf file to: {file_inf_path}")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(TRAIN_OUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(INF_OUT_PATH), exist_ok=True)
    
    # Copy files to target locations
    shutil.copy(file_train_path, TRAIN_OUT_PATH)
    shutil.copy(file_inf_path, INF_OUT_PATH)
    
    print(f"Saved train dataset to: {TRAIN_OUT_PATH}")
    print(f"Saved inf dataset to: {INF_OUT_PATH}")
    print("Dataset update complete.")

if __name__ == "__main__":
    main()
