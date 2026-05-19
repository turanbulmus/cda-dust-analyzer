from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config
from .state import SHARED_STATE
from ..tools.utils import create_batch_input_file

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()

class DataPrepAgent(BaseAgent):
    """Reads parquet data, extracts few-shot examples, and generates JSONL."""
    bucket_name: str
    data_path: str = "cda_dust_agent/data/testing/cda_test.parquet"
    limit: int = 0
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return
            
        yield log_and_yield(self.name, f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        limit_val = self.limit if self.limit > 0 else None
        if limit_val:
            df = df.head(limit_val)
            
        few_shot_examples = SHARED_STATE.get("few_shot_examples", [])
        used_ids = SHARED_STATE.get("used_ids", [])
        
        jsonl_file, _ = create_batch_input_file(df, few_shot_examples=few_shot_examples, limit=limit_val)
        yield log_and_yield(self.name, f"Generated {jsonl_file} with {len(df)} spectra for batch inference (excluding few-shot IDs: {used_ids})")
        
        # Save to shared workflow state
        SHARED_STATE["jsonl_file"] = jsonl_file
        SHARED_STATE["bucket_name"] = self.bucket_name
