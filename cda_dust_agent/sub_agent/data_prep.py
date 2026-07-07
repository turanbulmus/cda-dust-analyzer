import os
import json
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
from google.cloud import storage
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config
from ..tools.utils import create_batch_input_files

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()

class DataPrepAgent(BaseAgent):
    """Reads parquet data, extracts few-shot examples, generates chunked JSONL files, and streams them to GCS."""
    bucket_name: str
    data_path: str = "cda_dust_agent/data/testing/cda_test.parquet"
    limit: int = 0
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return
            
        if ctx.session.state.get("fsa_state") != "done":
            return

        # Set client Certificate overrides
        os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
        os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"
        storage_client = storage.Client(project=configs.agent_settings.project_id)
        bucket = storage_client.bucket(self.bucket_name.replace("gs://", ""))
        
        # Deleting old pre-rendered GCS batch inputs to guarantee fresh run with clean few-shots & pre-filter
        existing_blobs = [b for b in bucket.list_blobs(prefix="input/batch_requests_part") if b.name.endswith(".jsonl")]
        if existing_blobs:
            yield log_and_yield(self.name, f"Cleaning up {len(existing_blobs)} old batch chunks in GCS input folder to force fresh render...")
            for b in existing_blobs:
                b.delete()
            yield log_and_yield(self.name, "GCS input folder cleared.")

        yield log_and_yield(self.name, f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        limit_val = self.limit if self.limit > 0 else (configs.agent_settings.test_n if configs.agent_settings.test_mode else None)
        if limit_val and limit_val < len(df):
            yield log_and_yield(self.name, f"Sampling exactly {limit_val} test spectra from {len(df)} total spectra...")
            df = df.sample(n=limit_val, random_state=42)
            
        few_shot_examples = ctx.session.state.get("few_shot_examples", [])
        used_ids = set(ctx.session.state.get("used_ids", []))
        
        # Clean local bypassed predictions from previous runs
        bypassed_file = "cda_dust_agent/data/output/bypassed_predictions.jsonl"
        if os.path.exists(bypassed_file):
            os.remove(bypassed_file)
            
        # Apply Stage 1 Gating Pre-filter
        yield log_and_yield(self.name, "Applying Stage 1 Gating Pre-filter to identify obvious Noise...")
        from ..tools.utils import is_obvious_noise
        
        bypassed_count = 0
        bypassed_records = []
        keep_indices = []
        
        for idx, row in df.iterrows():
            spec = row["spectrum"]
            qi = row.get("qi_ampl")
            sclk_id = str(int(float(row["sclk"])))
            
            # Exclude few-shots from target classification
            if row["sclk"] in used_ids:
                continue
                
            if is_obvious_noise(spec, qi):
                bypassed_count += 1
                # Format to match standard Vertex AI Batch prediction line schema
                pred_text = json.dumps({
                    "id": sclk_id,
                    "class": "Noise",
                    "explanation": "Bypassed by Stage 1 peak detection pre-filter (flat quiet baseline, no physical peaks)."
                })
                bypassed_records.append({"response": {"candidates": [{"content": {"parts": [{"text": pred_text}]}}]}})
            else:
                keep_indices.append(idx)
                
        if bypassed_count > 0:
            yield log_and_yield(self.name, f" -> Pre-filter identified {bypassed_count} obvious Noise spectra. Saving directly to output.")
            os.makedirs(os.path.dirname(bypassed_file), exist_ok=True)
            with open(bypassed_file, "w") as f_out:
                for rec in bypassed_records:
                    f_out.write(json.dumps(rec) + "\n")
                    
        # Filter dataframe for VLM rendering
        df_vlm = df.loc[keep_indices]
        yield log_and_yield(self.name, f" -> Remaining {len(df_vlm)} spectra passed to VLM queue.")
        df = df_vlm

        def upload_and_delete(chunk_file_path):
            file_name = os.path.basename(chunk_file_path)
            gcs_path = f"input/{file_name}"
            logger.info(f"[DataPrep] Uploading {chunk_file_path} -> gs://{bucket.name}/{gcs_path}...")
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(chunk_file_path)
            gcs_url = f"gs://{bucket.name}/{gcs_path}"
            if os.path.exists(chunk_file_path):
                os.remove(chunk_file_path)
            logger.info(f"[DataPrep] Uploaded {gcs_url} and deleted local chunk file.")
            return gcs_url

        yield log_and_yield(self.name, f"Generating & streaming chunked batch input JSONL files for {len(df)} spectra (excluding few-shot IDs: {used_ids})...")
        gcs_sources, _ = create_batch_input_files(
            df,
            few_shot_examples=few_shot_examples,
            limit=limit_val,
            chunk_size=1500,
            on_chunk_complete=upload_and_delete
        )
        
        yield log_and_yield(self.name, f"Successfully uploaded {len(gcs_sources)} chunk files to GCS and cleaned up local temporary chunk files.")
        
        # Save to shared workflow state
        ctx.session.state["gcs_sources"] = gcs_sources
        ctx.session.state["bucket_name"] = self.bucket_name
