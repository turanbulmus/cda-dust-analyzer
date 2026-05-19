from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config
from ..tools.prompt_refinery import run_prompt_refinery

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()

class PromptOptimizerAgent(BaseAgent):
    """Runs the prompt optimization study to improve prompts."""
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if not configs.agent_settings.prompt_optimization:
            yield log_and_yield(self.name, "Prompt optimization is disabled. Skipping.")
            return
            
        yield log_and_yield(self.name, "Starting prompt optimization study...")
        try:
            system_prompt = await run_prompt_refinery(iterations=configs.agent_settings.prompt_opt_iterations)
            yield log_and_yield(self.name, "Prompt optimization complete. Refined prompt generated.")
        except Exception as e:
            yield log_and_yield(self.name, f"Prompt optimization failed: {e}")
