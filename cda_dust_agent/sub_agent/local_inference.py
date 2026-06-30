import os
import json

os.environ["GOOGLE_API_USE_MTLS"] = "never"
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config

from ..tools.utils import generate_spectrum_image_bytes
from ..prompts import SYSTEM_INSTRUCTION_TEXT, CLASSIFICATION_USER_PROMPT

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()

class LocalInferenceAgent(BaseAgent):
    """Runs local inference using Gemini SDK directly."""
    model_id: str
    data_path: str = "cda_dust_agent/data/testing/cda_test.parquet"
    limit: int = 0
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:

            
        if ctx.session.state.get("fsa_state") != "done":
            return
            
        limit_val = self.limit if self.limit > 0 else None
        yield log_and_yield(self.name, f"Starting local inference on up to {limit_val if limit_val else 'all'} samples using {self.model_id}...")
        
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
        import pandas as pd
        import json
        
        vertexai.init(project=configs.agent_settings.project_id, location="global")
        model = GenerativeModel(self.model_id, system_instruction=SYSTEM_INSTRUCTION_TEXT)
        
        df = pd.read_parquet(self.data_path)
        
        few_shot_examples = ctx.session.state.get("few_shot_examples", [])
        used_ids = ctx.session.state.get("used_ids", [])
        
        few_shot_parts = [CLASSIFICATION_USER_PROMPT]
        if few_shot_examples:
            few_shot_parts.append("Here are reference examples:")
            for ex in few_shot_examples:
                few_shot_parts.append(f"Example: {ex['label']} ({ex['explanation']})")
                few_shot_parts.append(Part.from_data(data=ex['image'], mime_type="image/png"))
            few_shot_parts.append("Now, analyze the following spectrum:")
            
        config = GenerationConfig(
            response_mime_type="application/json",
        )
        
        df_eval = df[~df['sclk'].isin(used_ids)]
        if limit_val:
            # Use random sampling instead of picking the first N samples
            df_eval = df_eval.sample(n=limit_val, random_state=42) if limit_val < len(df_eval) else df_eval
            
        yield log_and_yield(self.name, f"Prepared {len(df_eval)} spectra for local inference.")
        
        preds = []
        
        import os
        import base64
        import json
        
        os.makedirs("cda_dust_agent/data/input", exist_ok=True)
        local_reqs_file = "cda_dust_agent/data/input/local_requests.jsonl"
        with open(local_reqs_file, "w") as f:
            pass # clear the file
            
        for i, (_, row) in enumerate(df_eval.iterrows()):
            try:
                spect = row['spectrum']
                img_bytes = generate_spectrum_image_bytes(spect, title=f"Sample {row['sclk']}")
                
                parts = list(few_shot_parts)
                parts.append(Part.from_data(data=img_bytes, mime_type="image/png"))
                parts.append(f"Sample ID (sclk): {row['sclk']}")
                
                yield log_and_yield(self.name, f"Requesting prediction for item {i+1}/{len(df_eval)} (sclk: {row['sclk']})...")
                response = await model.generate_content_async(
                    parts,
                    generation_config=config
                )
                
                # Format to match batch prediction output shape for analysis agent
                pred_record = {
                    "request": {},
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": response.text}]
                                }
                            }
                        ]
                    }
                }
                preds.append(json.dumps(pred_record))
                
            except Exception as e:
                yield log_and_yield(self.name, f"Record {i+1} failed: {e}")
        
        import os
        os.makedirs("cda_dust_agent/data/output", exist_ok=True)
        preds_file = "cda_dust_agent/data/output/predictions.jsonl"
        with open(preds_file, "w") as f:
            for p in preds:
                f.write(p + "\n")
                
        ctx.session.state["job_success"] = True
        yield log_and_yield(self.name, f"Local inference complete. Saved to {preds_file}")
