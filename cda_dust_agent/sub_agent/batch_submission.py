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

        gcs_sources = ctx.session.state.get("gcs_sources")
        bucket_name = ctx.session.state.get("bucket_name")
        
        if not gcs_sources or not bucket_name:
            yield log_and_yield(self.name, "Missing state: gcs_sources or bucket_name")
            return
            
        clean_bucket = bucket_name.replace("gs://", "").strip("/")
        yield log_and_yield(self.name, f"Submitting Batch Job for sources: {gcs_sources}...")
        
        os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
        os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"
        from google.cloud import aiplatform
        
        aiplatform.init(project=self.project_id, location="global")
        
        job_display_name = f"cda-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        model_to_use = f"publishers/google/models/{self.model_id}"
        try:
            job = aiplatform.BatchPredictionJob.create(
                job_display_name=job_display_name,
                model_name=model_to_use,
                instances_format="jsonl",
                gcs_source=gcs_sources,
                predictions_format="jsonl",
                gcs_destination_prefix=f"gs://{clean_bucket}/output",
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
                    gcs_source=gcs_sources,
                    predictions_format="jsonl",
                    gcs_destination_prefix=f"gs://{clean_bucket}/output",
                )
                yield log_and_yield(self.name, f"Job Submitted Successfully! Job Name: {job.name}")
                ctx.session.state["job_name"] = job.name
            else:
                logger.error(f"Error submitting job: {e}")
                yield log_and_yield(self.name, f"Error submitting job: {e}")
