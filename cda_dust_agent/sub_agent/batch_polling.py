import json
import time
import subprocess
from typing import AsyncGenerator
from typing_extensions import override

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

class BatchPollingAgent(BaseAgent):
    """Polls Vertex AI until the job completes."""
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return
            
        if ctx.session.state.get("fsa_state") != "done":
            return

        from google.cloud import aiplatform
        
        aiplatform.init(project=configs.agent_settings.project_id, location="global")
        
        job_name = ctx.session.state.get("job_name")
        if not job_name:
            yield log_and_yield(self.name, "No job_name in state to poll.")
            return
            
        yield log_and_yield(self.name, f"Polling job status for {job_name}...")
        
        try:
            while True:
                job = aiplatform.BatchPredictionJob(job_name)
                state = job.state.name
                yield log_and_yield(self.name, f"Job State: {state}")
                
                if state == "JOB_STATE_SUCCEEDED":
                    ctx.session.state["job_success"] = True
                    break
                elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
                    yield log_and_yield(self.name, f"Job Ended with state: {state}")
                    if hasattr(job, 'error') and job.error:
                         yield log_and_yield(self.name, f"Error Details: {job.error}")
                    ctx.session.state["job_success"] = False
                    break
                    
                time.sleep(30)
        except Exception as e:
            yield log_and_yield(self.name, f"Error during polling: {e}")
            ctx.session.state["job_success"] = False
