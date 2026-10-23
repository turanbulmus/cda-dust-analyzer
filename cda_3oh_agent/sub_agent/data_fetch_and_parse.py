import os
import logging
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
import numpy as np
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config
from ..tools.utils import qm_scaling_lee_v2, CROP_START, CROP_STOP, SPECTRUM_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))


configs = Config()

# The 18 stratified exemplar SCLKs (6 3-OH + 12 non-3-OH across 1, 2, 4, 5-Na, Noise, 3-Car, 3-Cl, 3-K, 3-KNa)
STRATIFIED_18_EXEMPLAR_SCLKS = [
    1514186105,  # 1 (non-3-OH)
    1519489376,  # 5-Na (non-3-OH)
    1516106821,  # 4 (non-3-OH)
    1511732906,  # 3-Car (non-3-OH)
    1557520217,  # 3-OH
    1557518998,  # 3-OH
    1543802106,  # 3-OH
    1543798599,  # 3-OH
    1521663324,  # 3-KNa (non-3-OH)
    1519602129,  # Noise (non-3-OH)
    1511733538,  # 3-K (non-3-OH)
    1489103125,  # 3-Cl (non-3-OH)
    1571975703,  # 5-Na (non-3-OH)
    1589073259,  # 3-OH
    1498472699,  # 3-Cl (non-3-OH)
    1519537838,  # 3-Car (non-3-OH)
    1589073689,  # 3-OH
    1514176569,  # 2 (non-3-OH)
]


class DataFetchAndParseAgent(BaseAgent):
    """Fetches raw CDA spectra from HuggingFace, crops to 0.10-6.40 us (samples 10..640),
    applies Lee v2 log-smoothing, and constructs the binary 3-OH vs non-3-OH train/test splits."""

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path not in ["local", "batch"]:
            return

        train_out_path = "cda_3oh_agent/data/raw/cda_train.parquet"
        test_out_path = "cda_3oh_agent/data/testing/cda_test.parquet"
        inf_out_path = "cda_3oh_agent/data/raw/cda_inf.parquet"

        if not configs.agent_settings.fetch_data:
            if os.path.exists(train_out_path) and os.path.exists(test_out_path):
                yield log_and_yield(
                    self.name,
                    "Data files exist and fetch_data is False. Skipping fetch and parse.",
                )
                return

        yield log_and_yield(
            self.name,
            "Fetching and preprocessing CDA spectra for binary 3-OH vs non-3-OH classification...",
        )

        import huggingface_hub

        os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_out_path), exist_ok=True)

        repo_id = "CosmicDustGroup/cassini-cda-spectra"
        filename_train = "data/lvl2/cda_qm_spectra_pre2008277_train_lvl2.parquet"

        file_train_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=filename_train, repo_type="dataset"
        )
        df = pd.read_parquet(file_train_path)

        # 1. Keep 1018-sample records and crop to the 100 MHz window [10:641] (0.10-6.40 us)
        df = df[df["spectrum"].apply(len) == SPECTRUM_LENGTH].copy()
        df["spectrum"] = df["spectrum"].apply(
            lambda s: qm_scaling_lee_v2(np.asarray(s, dtype=float)[CROP_START:CROP_STOP])
        )

        # 2. Exclude unlabelled/ambiguous records (?, X, 2-X) and retain catalogue_class + binary class
        df["catalogue_class"] = df["class"].astype(str)
        df = df[~df["catalogue_class"].isin(["?", "X", "2-X", "nan", "None"])].copy()
        df["class"] = np.where(df["catalogue_class"] == "3-OH", "3-OH", "non-3-OH")
        df["sclk"] = df["sclk"].astype("int64")

        # 3. Build ordered 18-exemplar training split and hold out all remaining 18,307 rows for testing
        sclk_to_row = {int(r["sclk"]): r for _, r in df.iterrows()}
        train_rows = [sclk_to_row[s] for s in STRATIFIED_18_EXEMPLAR_SCLKS if s in sclk_to_row]
        train_data = pd.DataFrame(train_rows).reset_index(drop=True)
        train_sclks = set(train_data["sclk"].astype(int).tolist())

        test_data = df[~df["sclk"].isin(train_sclks)].copy().reset_index(drop=True)

        train_data.to_parquet(train_out_path)
        test_data.to_parquet(test_out_path)
        test_data.to_parquet(inf_out_path)

        n_3oh_train = int((train_data["class"] == "3-OH").sum())
        n_3oh_test = int((test_data["class"] == "3-OH").sum())
        yield log_and_yield(
            self.name,
            f"3-OH data prep complete: Train={len(train_data)} exemplars ({n_3oh_train} 3-OH, "
            f"{len(train_data) - n_3oh_train} non-3-OH), Test={len(test_data)} labelled spectra "
            f"({n_3oh_test} 3-OH, {len(test_data) - n_3oh_test} non-3-OH across 12 catalogue subclasses).",
        )
