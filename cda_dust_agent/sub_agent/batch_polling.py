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
from .state import SHARED_STATE

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
            
        if SHARED_STATE.get("fsa_state") != "done":
            return

        job_name = SHARED_STATE.get("job_name")
        access_token = SHARED_STATE.get("access_token")
        
        if not job_name:
            yield log_and_yield(self.name, "No job_name in state to poll.")
            return
            
        check_url = f"https://aiplatform.googleapis.com/v1/{job_name}"
        yield log_and_yield(self.name, "Polling job status...")
        
        while True:
            check_cmd = [
                "curl", "-s", "-X", "GET",
                check_url,
                "-H", f"Authorization: Bearer {access_token}"
            ]
            check_res = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if check_res.returncode != 0:
                yield log_and_yield(self.name, f"Error checking status: {check_res.stderr}")
                time.sleep(30)
                continue
                
            status_data = json.loads(check_res.stdout)
            state = status_data.get("state", "UNKNOWN")
            
            yield log_and_yield(self.name, f"Job State: {state}")
            
            if state == "JOB_STATE_SUCCEEDED":
                SHARED_STATE["job_success"] = True
                break
            elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
                yield log_and_yield(self.name, f"Job Ended with state: {state}")
                if "error" in status_data:
                    yield log_and_yield(self.name, f"Error Details: {status_data['error']}")
                SHARED_STATE["job_success"] = False
                break
                
            time.sleep(30)
