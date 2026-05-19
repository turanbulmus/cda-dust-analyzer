import os
import json
import time
import subprocess
from datetime import datetime
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
from google.cloud import storage
from google.cloud import aiplatform

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from .config import Config
from .tools.utils import create_batch_input_file
from .tools.prompt_refinery import run_prompt_refinery

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()


from .sub_agent.data_fetch_and_parse import DataFetchAndParseAgent
from .sub_agent.few_shot_annotation import FewShotAnnotationAgent
from .sub_agent.prompt_optimizer import PromptOptimizerAgent
from .sub_agent.data_prep import DataPrepAgent
from .sub_agent.local_inference import LocalInferenceAgent
from .sub_agent.batch_submission import BatchSubmissionAgent
from .sub_agent.batch_polling import BatchPollingAgent
from .sub_agent.result_analysis import ResultAnalysisAgent
from .sub_agent.vertex_ai_logging import VertexAIExperimentsLoggingAgent

# The root agent that ADK expects
root_agent = SequentialAgent(
    name=configs.agent_settings.name,
    sub_agents=[
        DataFetchAndParseAgent(name="DataFetchAndParse"),
        PromptOptimizerAgent(name="PromptOptimizer"),
        FewShotAnnotationAgent(name="FewShotAnnotation", data_path="cda_dust_agent/data/raw/cda_train.parquet"),
        DataPrepAgent(name="DataPrep", bucket_name=configs.agent_settings.bucket_name, limit=0),
        LocalInferenceAgent(name="LocalInference", model_id=configs.agent_settings.model),
        BatchSubmissionAgent(name="BatchSubmission", project_id=configs.agent_settings.project_id, model_id=configs.agent_settings.model),
        BatchPollingAgent(name="BatchPolling"),
        ResultAnalysisAgent(name="ResultAnalysis", project_id=configs.agent_settings.project_id),
        VertexAIExperimentsLoggingAgent(name="VertexAIExperimentsLogging", project_id=configs.agent_settings.project_id)
    ],
    description="Orchestrates the data prep, job submission, polling, results analysis, and logging sequentially."
)
