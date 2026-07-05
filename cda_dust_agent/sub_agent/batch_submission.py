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

class BatchSubmissionAgent(BaseAgent):
    """Uploads the JSONL to GCS and submits the Vertex AI batch prediction job."""
    project_id: str
    model_id: str
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return
            
        if ctx.session.state.get("fsa_state") != "done":
            return

        jsonl_file = ctx.session.state.get("jsonl_file")
        bucket_name = ctx.session.state.get("bucket_name")
        
        if not jsonl_file or not bucket_name:
            yield log_and_yield(self.name, "Missing state: jsonl_file or bucket_name")
            return
            
        yield log_and_yield(self.name, f"Uploading {jsonl_file} to GCS bucket: {bucket_name}")
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
        
        blob = bucket.blob(f"input/{jsonl_file}")
        blob.upload_from_filename(jsonl_file)
        gcs_source = f"gs://{bucket.name}/input/{jsonl_file}"
        
        yield log_and_yield(self.name, f"Uploaded to {gcs_source}. Submitting Batch Job via curl...")
        
        from google.cloud import aiplatform
        
        aiplatform.init(project=self.project_id, location="global")
        
        job_display_name = f"cda-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        model_to_use = f"publishers/google/models/{self.model_id}"
        try:
            job = aiplatform.BatchPredictionJob.create(
                job_display_name=job_display_name,
                model_name=model_to_use,
                instances_format="jsonl",
                gcs_source=gcs_source,
                predictions_format="jsonl",
                gcs_destination_prefix=f"gs://{bucket.name}/output",
            )
            yield log_and_yield(self.name, f"Job Submitted Successfully! Job Name: {job.name}")
            ctx.session.state["job_name"] = job.name
        except Exception as e:
            if "404" in str(e) or "NOT_FOUND" in str(e):
                fallback_model = "publishers/google/models/gemini-2.5-flash"
                yield log_and_yield(self.name, f"Model {model_to_use} not found (404), falling back to {fallback_model}...")
                job = aiplatform.BatchPredictionJob.create(
                    job_display_name=job_display_name,
                    model_name=fallback_model,
                    instances_format="jsonl",
                    gcs_source=gcs_source,
                    predictions_format="jsonl",
                    gcs_destination_prefix=f"gs://{bucket.name}/output",
                )
                yield log_and_yield(self.name, f"Job Submitted Successfully! Job Name: {job.name}")
                ctx.session.state["job_name"] = job.name
            else:
                logger.error(f"Error submitting job: {e}")
                yield log_and_yield(self.name, f"Error submitting job: {e}")
