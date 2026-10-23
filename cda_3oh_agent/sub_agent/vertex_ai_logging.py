import os
import json
import logging
from datetime import datetime
from typing import AsyncGenerator
from typing_extensions import override

from google.cloud import aiplatform
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


configs = Config()


class VertexAIExperimentsLoggingAgent(BaseAgent):
    """Logs binary 3-OH vs non-3-OH evaluation metrics and config parameters to Vertex AI Experiments."""

    project_id: str

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        yield log_and_yield(self.name, "Starting Vertex AI Experiments logging for cda_3oh_agent...")

        metrics_path = "cda_3oh_agent/data/results/metrics_3oh_agent.json"
        if not os.path.exists(metrics_path):
            yield log_and_yield(self.name, f"Metrics file not found at {metrics_path}. Skipping logging.")
            return

        try:
            with open(metrics_path, "r") as f:
                m = json.load(f)

            metrics = {
                "accuracy": float(m["accuracy"]),
                "precision_3oh": float(m["precision_3oh"]),
                "recall_3oh": float(m["recall_3oh"]),
                "f1_3oh": float(m["f1_3oh"]),
                "macro_f1": float(m["macro_f1"]),
            }

            params = {
                "agent": "cda_3oh_agent",
                "task": "binary_3oh_vs_rest",
                "model": configs.agent_settings.model,
                "inference_path": configs.agent_settings.inference_path,
                "few_shot_n": configs.agent_settings.few_shot_n,
                "test_n": configs.agent_settings.test_n,
                "prompt_optimization": str(configs.agent_settings.prompt_optimization),
                "stage1_bypassed": int(m.get("stage1_bypassed", 0)),
                "vlm_evaluated": int(m.get("vlm_evaluated", 0)),
            }

            experiment_name = "cda-3oh-analyzer-experiment"
            run_name = f"run-3oh-argon-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            aiplatform.init(
                project=self.project_id,
                location=configs.agent_settings.location,
                experiment=experiment_name,
            )

            with aiplatform.start_run(run=run_name):
                aiplatform.log_params(params)
                aiplatform.log_metrics(metrics)

            yield log_and_yield(
                self.name,
                f"Logged parameters and metrics to Vertex AI Experiment '{experiment_name}' (run '{run_name}').",
            )
        except Exception as e:
            yield log_and_yield(self.name, f"Error logging to Vertex AI Experiments: {e}")
