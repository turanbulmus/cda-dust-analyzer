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

from google.adk.agents import BaseAgent
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


from .sub_agent.workflow import CdaWorkflowAgent

# The root agent that ADK expects
root_agent = CdaWorkflowAgent(name=configs.agent_settings.name)
