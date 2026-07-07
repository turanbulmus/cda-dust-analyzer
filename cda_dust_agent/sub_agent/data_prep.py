import os
import json
import subprocess
from datetime import datetime
from typing import AsyncGenerator
from typing_extensions import override

from google.cloud import storage
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

class DataPrepAgent(BaseAgent):
    """Pre-filters noise, renders spectrum plots, and generates visual VLM batch requests using Apache Beam (Cloud Dataflow)."""
    bucket_name: str
    data_path: str = "cda_dust_agent/data/testing/cda_test.parquet"
    limit: int = 0
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return
            
        if ctx.session.state.get("fsa_state") != "done":
            return

        # Setup GCS Storage Client
        os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
        os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"
        
        project_id = configs.agent_settings.project_id
        clean_bucket = self.bucket_name.replace("gs://", "").strip("/")
        
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(clean_bucket)
        
        # 1. Clean existing input/bypassed files in GCS
        existing_inputs = [b for b in bucket.list_blobs(prefix="input/batch_requests_part") if b.name.endswith(".jsonl")]
        existing_bypassed = [b for b in bucket.list_blobs(prefix="output/bypassed/bypassed_predictions") if b.name.endswith(".jsonl")]
        
        if existing_inputs or existing_bypassed:
            yield log_and_yield(self.name, f"Cleaning up {len(existing_inputs)} old input files and {len(existing_bypassed)} old bypassed files in GCS...")
            for b in existing_inputs + existing_bypassed:
                b.delete()
            yield log_and_yield(self.name, "GCS input/bypassed folders cleared.")

        # 2. Upload fresh dataset Parquet and cached few-shot examples to GCS
        few_shot_cache_local = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
        
        yield log_and_yield(self.name, "Uploading test Parquet and few-shot annotations to GCS...")
        
        parquet_blob = bucket.blob("data/cda_test.parquet")
        parquet_blob.upload_from_filename(self.data_path)
        
        few_shot_blob = bucket.blob("data/cached_examples.jsonl")
        few_shot_blob.upload_from_filename(few_shot_cache_local)
        
        yield log_and_yield(self.name, "Upload complete! Files staged on GCS.")

        # 3. Determine runner options
        test_mode = configs.agent_settings.test_mode
        runner = "DirectRunner" if test_mode else "DataflowRunner"
        limit_val = self.limit if self.limit > 0 else (configs.agent_settings.test_n if test_mode else 0)
        
        job_name = f"cda-data-prep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        yield log_and_yield(self.name, f"Submitting Apache Beam pipeline using {runner} (Job Name: {job_name}, Limit: {limit_val})...")
        
        # 4. Construct Dataflow pipeline command
        cmd = [
            ".venv/bin/python", "agent/scripts/dataflow_data_prep.py",
            "--input_parquet", f"gs://{clean_bucket}/data/cda_test.parquet",
            "--few_shot_gcs", f"gs://{clean_bucket}/data/cached_examples.jsonl",
            "--output_queued_prefix", f"gs://{clean_bucket}/input/batch_requests_part",
            "--output_bypassed_prefix", f"gs://{clean_bucket}/output/bypassed/bypassed_predictions",
            "--project_id", project_id,
            "--runner", runner,
            "--limit", str(limit_val)
        ]
        
        # Append Dataflow-specific options if running in cloud
        if runner == "DataflowRunner":
            cmd.extend([
                "--project", project_id,
                "--region", configs.agent_settings.location,
                "--temp_location", f"gs://{clean_bucket}/temp",
                "--staging_location", f"gs://{clean_bucket}/staging",
                "--requirements_file", "requirements_dataflow.txt",
                f"--job_name={job_name}"
            ])
            
        # Execute the pipeline script
        # Set PIP_INDEX_URL inside python command environment to use public PyPI as primary index
        env = os.environ.copy()
        env["PIP_INDEX_URL"] = "https://pypi.org/simple"
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            # Stream logs in real-time
            for line in iter(process.stdout.readline, ""):
                if line.strip():
                    logger.info(f"[{runner}] {line.strip()}")
                    
            process.stdout.close()
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
                
            yield log_and_yield(self.name, "Apache Beam pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error executing Beam pipeline: {e}")
            yield log_and_yield(self.name, f"Pipeline Error: {e}")
            raise e

        # 5. Populate session state with generated GCS file URLs
        gcs_sources = []
        blobs = bucket.list_blobs(prefix="input/batch_requests_part")
        for b in blobs:
            if b.name.endswith(".jsonl"):
                gcs_sources.append(f"gs://{clean_bucket}/{b.name}")
                
        yield log_and_yield(self.name, f"Discovered {len(gcs_sources)} chunk files generated in GCS input folder.")
        
        ctx.session.state["gcs_sources"] = gcs_sources
        ctx.session.state["bucket_name"] = self.bucket_name
        ctx.session.state["bypassed_noise_saved"] = True
