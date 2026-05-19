import os
from datetime import datetime
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
from google.cloud import aiplatform
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

class VertexAIExperimentsLoggingAgent(BaseAgent):
    """Logs analysis results and config parameters to Vertex AI Experiments."""
    project_id: str
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        yield log_and_yield(self.name, "Starting Vertex AI Experiments logging...")
        
        results_csv = "cda_dust_agent/data/results/results.csv"
        if not os.path.exists(results_csv):
            yield log_and_yield(self.name, f"Results file not found at {results_csv}. Skipping logging.")
            return
            
        try:
            from sklearn.metrics import accuracy_score
            
            # Read results
            df = pd.read_csv(results_csv)
            if df.empty:
                yield log_and_yield(self.name, "Results file is empty. Skipping logging.")
                return
                
            # Compute metrics
            y_true = df['true_class'].astype(str)
            y_pred = df['predicted_class'].astype(str)
            
            accuracy = accuracy_score(y_true, y_pred)
            
            metrics = {
                "accuracy": float(accuracy)
            }
            
            # Log parameters
            params = {
                "model": configs.agent_settings.model,
                "inference_path": configs.agent_settings.inference_path,
                "few_shot_n": configs.agent_settings.few_shot_n,
                "test_n": configs.agent_settings.test_n,
                "prompt_optimization": configs.agent_settings.prompt_optimization,
                "prompt_opt_iterations": configs.agent_settings.prompt_opt_iterations,
                "test_mode": configs.agent_settings.test_mode
            }
            
            # Initialize Vertex AI
            experiment_name = "cda-dust-analyzer-experiment"
            run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            yield log_and_yield(self.name, f"Initializing Vertex AI Experiment: {experiment_name}, Run: {run_name}")
            
            aiplatform.init(
                project=self.project_id,
                location=configs.agent_settings.location,
                experiment=experiment_name
            )
            
            aiplatform.start_run(run=run_name)
            
            yield log_and_yield(self.name, f"Logging parameters: {params}")
            aiplatform.log_params(params)
            
            yield log_and_yield(self.name, f"Logging metrics: {metrics}")
            aiplatform.log_metrics(metrics)
            
            yield log_and_yield(self.name, "Vertex AI Experiments logging complete.")
            
        except Exception as e:
            yield log_and_yield(self.name, f"Error logging to Vertex AI Experiments: {e}")
