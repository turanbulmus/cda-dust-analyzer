import json
import asyncio
import subprocess
import logging
from datetime import datetime
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config
from .batch_submission import _get_access_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))


configs = Config()


class BatchPollingAgent(BaseAgent):
    """Polls the Vertex AI Batch Prediction Job until completion."""

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return

        if ctx.session.state.get("bs_state") != "done":
            yield log_and_yield(self.name, "Batch submission did not complete; skipping polling.")
            return

        job_resource_name = ctx.session.state.get("job_resource_name")
        yield log_and_yield(self.name, f"Polling Vertex AI BatchPredictionJob: {job_resource_name}...")

        url = f"https://aiplatform.googleapis.com/v1/{job_resource_name}"

        while True:
            token = _get_access_token()
            res = subprocess.run(
                ["curl", "-s", "-X", "GET", url, "-H", f"Authorization: Bearer {token}"],
                capture_output=True,
                text=True,
            )
            try:
                status_data = json.loads(res.stdout)
                state = status_data.get("state", "UNKNOWN")
                stats = status_data.get("completionStats", {})
                ok = stats.get("successfulCount", 0)
                fail = stats.get("failedCount", 0)
                inc = stats.get("incompleteCount", 0)
                yield log_and_yield(
                    self.name,
                    f"[{datetime.now().strftime('%H:%M:%S')}] Job State: {state} "
                    f"(ok={ok}, fail={fail}, incomplete={inc})",
                )

                if state in ["JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"] or (
                    state == "JOB_STATE_RUNNING" and int(ok) > 0 and int(inc) == 0
                ):
                    out_dir = status_data.get("outputInfo", {}).get("gcsOutputDirectory")
                    if out_dir:
                        ctx.session.state["gcs_output_directory"] = out_dir
                    yield log_and_yield(self.name, f"Batch Job Finished ({state}, ok={ok}, incomplete={inc})! Output dir: {out_dir}")
                    ctx.session.state["job_success"] = True
                    break
                elif state in [
                    "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED",
                    "JOB_STATE_PAUSED",
                    "JOB_STATE_EXPIRED",
                ]:
                    error_msg = status_data.get("error", "No error details.")
                    yield log_and_yield(self.name, f"Job ended with state {state}. Details: {error_msg}")
                    ctx.session.state["job_success"] = False
                    break
            except Exception as e:
                yield log_and_yield(self.name, f"Error parsing poll response: {e}")

            await asyncio.sleep(30)
