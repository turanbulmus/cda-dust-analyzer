import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config
from ..tools.prompt_refinery import run_prompt_refinery

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))


configs = Config()


class PromptOptimizerAgent(BaseAgent):
    """Runs the Anchored Academic Multi-Agent Prompt Optimization (Research -> Critic -> Referee)
    starting from ANCHOR_3OH_AI_PROMPT using Gemini 4 Argon."""

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if not configs.agent_settings.prompt_optimization:
            yield log_and_yield(self.name, "Prompt optimization is disabled. Using anchor prompt.")
            return

        yield log_and_yield(
            self.name,
            f"Starting Anchored Academic Multi-Agent Prompt Optimization with {configs.agent_settings.model}...",
        )
        try:
            system_prompt = await run_prompt_refinery(
                max_levels=configs.agent_settings.prompt_opt_iterations,
                project_id=configs.agent_settings.project_id,
            )
            ctx.session.state["optimized_system_prompt"] = system_prompt
            yield log_and_yield(
                self.name,
                f"Prompt optimization complete ({len(system_prompt)} chars). Saved to cda_3oh_agent/refined_prompt.txt.",
            )
        except Exception as e:
            yield log_and_yield(self.name, f"Prompt optimization failed, falling back to anchor prompt: {e}")
