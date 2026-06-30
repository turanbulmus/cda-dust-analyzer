import os
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()

class DataFetchAndParseAgent(BaseAgent):
    """Fetches raw spectra from HuggingFace, filtering, cropping, and log-scaling it into a unified parquet dataset."""
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path not in ["local", "batch"]:
            return

        train_out_path = "cda_dust_agent/data/raw/cda_train.parquet"
        test_out_path = "cda_dust_agent/data/testing/cda_test.parquet"
        inf_out_path = "cda_dust_agent/data/raw/cda_inf.parquet"
        
        if not configs.agent_settings.fetch_data:
            if os.path.exists(train_out_path) and os.path.exists(test_out_path) and os.path.exists(inf_out_path):
                yield log_and_yield(self.name, "Data files exist and fetch_data is False. Skipping fetch and parse.")
                return
            else:
                yield log_and_yield(self.name, "Data files do not exist but fetch_data is False. Proceeding anyway or this might fail later.")

        yield log_and_yield(self.name, "Fetching and processing spectra from HuggingFace... (this may take a moment)")
        
        import huggingface_hub
        import pandas as pd
        import numpy as np
        
        os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_out_path), exist_ok=True)
        
        REPO_ID = "CosmicDustGroup/cassini-cda-spectra"
        FILENAME_INF = "data/lvl2/cda_qm_spectra_pre2008277_inf_lvl2.parquet"
        FILENAME_TRAIN = "data/lvl2/cda_qm_spectra_pre2008277_train_lvl2.parquet"
        
        # Download files
        file_inf_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME_INF, repo_type="dataset")
        file_train_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME_TRAIN, repo_type="dataset")
        
        train_df = pd.read_parquet(file_train_path)
        inf_df = pd.read_parquet(file_inf_path)
        
        # Amplitude filtering
        if 'qi_ampl' in train_df.columns:
            train_df = train_df[train_df['qi_ampl'] >= 10 * 10**-15].copy()
        if 'qi_ampl' in inf_df.columns:
            inf_df = inf_df[inf_df['qi_ampl'] >= 10 * 10**-15].copy()
        
        # 1018 filtering
        train_df_1018 = train_df[train_df['spectrum'].apply(len) == 1018].copy()
        inf_df_1018 = inf_df[inf_df['spectrum'].apply(len) == 1018].copy()
        
        # Crop spectra to index 10 to 640
        def crop_spectrum(spectrum):
            return spectrum[10:641]
            
        train_df_1018['spectrum'] = train_df_1018['spectrum'].apply(crop_spectrum)
        inf_df_1018['spectrum'] = inf_df_1018['spectrum'].apply(crop_spectrum)
        
        # Re-assign labels
        train_df_1018['class'] = train_df_1018['class'].apply(lambda x: '?' if "X" in x else x)
        
        class_3_df_1018 = train_df_1018[train_df_1018['class'] == '3'].copy()
        train_df_1018 = train_df_1018[train_df_1018['class'] != '3']
        inf_df_1018 = pd.concat([inf_df_1018, class_3_df_1018], ignore_index=True)
        
        # Scaling
        def qm_scaling(spectrum):
            spectrum = np.log10(spectrum + np.abs(np.min(spectrum)))
            spectrum = np.nan_to_num(spectrum, neginf=0)
            spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
            return spectrum

        train_df_1018['spectrum'] = train_df_1018['spectrum'].apply(qm_scaling)
        inf_df_1018['spectrum'] = inf_df_1018['spectrum'].apply(qm_scaling)
        
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # Filter out classes with fewer than 2 samples to allow stratified splitting
        class_counts = train_df_1018['class'].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        df_to_split = train_df_1018[train_df_1018['class'].isin(valid_classes)].reset_index(drop=True)
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(df_to_split, df_to_split['class']):
            train_data = df_to_split.iloc[train_index]
            test_data = df_to_split.iloc[test_index]
            
        few_shot_n = configs.agent_settings.few_shot_n
        test_n = configs.agent_settings.test_n if configs.agent_settings.test_mode else None
            
        # Cap training data
        train_data = train_data.groupby('class').head(few_shot_n).reset_index(drop=True)
        
        # Cap test data if test_n is set
        if test_n is not None:
            test_data = test_data.groupby('class').head(test_n).reset_index(drop=True)
        
        train_data.to_parquet(train_out_path)
        test_data.to_parquet(test_out_path)
        inf_df_1018.to_parquet(inf_out_path)
        
        yield log_and_yield(self.name, f"Data fetched, split (80/20), capped at {few_shot_n} per class for training, and saved to {train_out_path}, {test_out_path}, and {inf_out_path}.")
