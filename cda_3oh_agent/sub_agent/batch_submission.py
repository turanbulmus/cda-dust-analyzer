import json
import subprocess
import logging
from datetime import datetime
from typing import AsyncGenerator
from typing_extensions import override

from google.cloud import storage
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))


def _get_access_token() -> str:
    try:
        return subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token


configs = Config()


class BatchSubmissionAgent(BaseAgent):
    """Submits the Vertex AI Batch Prediction Job on Gemini 4 Argon (`publishers/google/models/gemini-4-argon`)."""

    project_id: str
    location: str
    model_id: str

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return

        if ctx.session.state.get("dp_state") != "done":
            yield log_and_yield(self.name, "Data prep did not complete; skipping batch submission.")
            return

        bucket_name = ctx.session.state.get("bucket_name", configs.agent_settings.bucket_name)
        clean_bucket = bucket_name.replace("gs://", "").strip("/")
        input_prefix = configs.agent_settings.gcs_input_prefix.strip("/")
        output_prefix = configs.agent_settings.gcs_output_prefix.strip("/")

        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(clean_bucket)
        input_blobs = sorted(
            [
                f"gs://{clean_bucket}/{b.name}"
                for b in bucket.list_blobs(prefix=f"{input_prefix}/batch_requests")
                if b.name.endswith(".jsonl") and b.size > 0
            ]
        )

        if not input_blobs:
            yield log_and_yield(self.name, f"Error: No non-empty JSONL shards found in gs://{clean_bucket}/{input_prefix}/")
            ctx.session.state["bs_state"] = "failed"
            return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        display_name = f"cda-3oh-agent-argon-{timestamp}"
        gcs_output_uri = f"gs://{clean_bucket}/{output_prefix}/{display_name}"

        yield log_and_yield(
            self.name,
            f"Submitting Vertex AI BatchPredictionJob '{display_name}' on model '{self.model_id}' "
            f"with {len(input_blobs)} input shard(s) -> {gcs_output_uri}...",
        )

        token = _get_access_token()
        url = f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/global/batchPredictionJobs"
        payload = {
            "displayName": display_name,
            "model": f"publishers/google/models/{self.model_id}",
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {"uris": input_blobs},
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": gcs_output_uri},
            },
        }

        res = subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                url,
                "-H",
                f"Authorization: Bearer {token}",
                "-H",
                "Content-Type: application/json; charset=utf-8",
                "-d",
                "@-",
            ],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
        )

        try:
            response_data = json.loads(res.stdout)
            if "name" not in response_data:
                raise RuntimeError(f"Failed to submit batch job: {res.stdout}")
            job_resource_name = response_data["name"]
            yield log_and_yield(self.name, f"Batch Job Submitted! Resource Name: {job_resource_name}")
            ctx.session.state["job_resource_name"] = job_resource_name
            ctx.session.state["gcs_output_uri"] = gcs_output_uri
            ctx.session.state["bs_state"] = "done"
        except Exception as e:
            yield log_and_yield(self.name, f"Error submitting batch job: {e}")
            ctx.session.state["bs_state"] = "failed"
            raise e
