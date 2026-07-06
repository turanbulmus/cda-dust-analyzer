import os
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
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
    """Fetches raw spectra from HuggingFace, filtering, cropping, log-scaling, and applying Savitzky-Golay smoothing matching the academic study."""
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path not in ["local", "batch"]:
            return

        train_out_path = "cda_dust_agent/data/raw/cda_train.parquet"
        test_out_path = "cda_dust_agent/data/testing/cda_test.parquet"
        inf_out_path = "cda_dust_agent/data/raw/cda_inf.parquet"
        
        if not configs.agent_settings.fetch_data:
            if os.path.exists(train_out_path) and os.path.exists(test_out_path):
                yield log_and_yield(self.name, "Data files exist and fetch_data is False. Skipping fetch and parse.")
                return

        yield log_and_yield(self.name, "Fetching and processing spectra from HuggingFace matching the Academic Study pipeline...")
        
        import huggingface_hub
        
        os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_out_path), exist_ok=True)
        
        REPO_ID = "CosmicDustGroup/cassini-cda-spectra"
        FILENAME_TRAIN = "data/lvl2/cda_qm_spectra_pre2008277_train_lvl2.parquet"
        
        file_train_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME_TRAIN, repo_type="dataset")
        df = pd.read_parquet(file_train_path)
        
        # 1018 length filtering & crop spectrum [10:641]
        df = df[df['spectrum'].apply(len) == 1018].copy()
        df['spectrum'] = df['spectrum'].apply(lambda s: s[10:641])
        
        # Label mapping matching academic study
        def map_label(x):
            if not isinstance(x, str) or x in ["?", "X", "2-X"]:
                return "?"
            if x == "3-P":
                return "3-P"
            if x.startswith("3-") or x == "3":
                return "3"
            return x

        df['class'] = df['class'].apply(map_label)
        df = df[df['class'] != '?'].copy()
        
        # QM scaling with Savitzky-Golay filter
        def qm_scaling_savgol(spectrum):
            spectrum = np.array(spectrum, dtype=float)
            log_spec = np.log10(spectrum + np.abs(np.min(spectrum)) + 1e-12)
            finite_mask = np.isfinite(log_spec)
            min_finite_val = np.min(log_spec[finite_mask]) if np.any(finite_mask) else 0
            log_spec = np.nan_to_num(log_spec, neginf=min_finite_val, nan=min_finite_val)
            spec_min, spec_max = np.min(log_spec), np.max(log_spec)
            range_val = spec_max - spec_min
            scaled = (log_spec - spec_min) / range_val if range_val > 0 else np.zeros_like(log_spec)
            return savgol_filter(scaled, 5, 3)

        df['spectrum'] = df['spectrum'].apply(qm_scaling_savgol)
        
        classes = ['Noise', '1', '2', '3', '4', '5', '5-Na', '3-P']
        train_rows = []
        train_sclks = []
        
        for cls in classes:
            cls_df = df[df['class'] == cls]
            n_train = 4 if cls == '3-P' else 16
            cls_train = cls_df.sample(n=min(n_train, len(cls_df)), random_state=123)
            train_rows.append(cls_train)
            train_sclks.extend(cls_train['sclk'].tolist())
            
        train_data = pd.concat(train_rows, ignore_index=True)
        test_data = df[~df['sclk'].isin(train_sclks)].copy()
        
        train_data.to_parquet(train_out_path)
        test_data.to_parquet(test_out_path)
        test_data.to_parquet(inf_out_path)
        
        yield log_and_yield(self.name, f"Academic study data prep complete: Train={len(train_data)} samples, Test={len(test_data)} samples across 8 classes saved to {train_out_path} and {test_out_path}.")
