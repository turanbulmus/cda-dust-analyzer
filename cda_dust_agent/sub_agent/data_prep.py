from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config

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
            
        if ctx.session.state.get("fsa_state") != "done":
            return
            
        yield log_and_yield(self.name, f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        limit_val = self.limit if self.limit > 0 else None
        if limit_val and limit_val < len(df):
            df = df.sample(n=limit_val, random_state=42)
            
        few_shot_examples = ctx.session.state.get("few_shot_examples", [])
        used_ids = ctx.session.state.get("used_ids", [])
        
        jsonl_file, _ = create_batch_input_file(df, few_shot_examples=few_shot_examples, limit=limit_val)
        yield log_and_yield(self.name, f"Generated {jsonl_file} with {len(df)} spectra for batch inference (excluding few-shot IDs: {used_ids})")
        
        # Save to shared workflow state
        ctx.session.state["jsonl_file"] = jsonl_file
        ctx.session.state["bucket_name"] = self.bucket_name
