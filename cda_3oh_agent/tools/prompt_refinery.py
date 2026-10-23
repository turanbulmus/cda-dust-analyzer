"""Prompt Optimization Refinery for the binary 3-OH vs non-3-OH CDA Dust Analyzer Agent.

Combines:
1. The Anchored Academic Multi-Agent optimization (`AcademicResearchAgent` -> `AcademicCriticAgent` -> `AcademicRefereeAgent`)
   starting from `ANCHOR_3OH_AI_PROMPT`.
2. Vertex AI Prompt Management registration of both the `before` (anchor) and `after` (optimized) prompts.
"""
import logging
from datetime import datetime
import vertexai
from vertexai import generative_models
from vertexai import types as vx_types

from ..config import Config
from ..prompts import ANCHOR_3OH_AI_PROMPT
from ..academic_multi_agent import run_anchored_academic_multi_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

configs = Config()


async def run_prompt_refinery(
    autonomous: bool = True,
    max_levels: int = 3,
    use_batch: bool = False,
    project_id: str = "turan-genai-bb",
) -> str:
    """Runs the anchored prompt optimization pipeline using Gemini 4 Argon and logs prompts to Vertex AI."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vx_client = None
    try:
        vx_client = vertexai.Client(project=project_id, location="us-central1")
        before_prompt = vx_client.prompts.create(
            prompt_id=f"cda_3oh_anchor_prompt_{timestamp}",
            prompt=vx_types.Prompt(
                prompt_data=vx_types.PromptData(
                    contents=[
                        generative_models.Content(
                            role="user",
                            parts=[generative_models.Part.from_text(ANCHOR_3OH_AI_PROMPT)],
                        )
                    ],
                    model=configs.agent_settings.model,
                )
            ),
        )
        logger.info(f"Saved anchor 'before' prompt to Vertex AI Prompt Management: {before_prompt.name}")
    except Exception as e:
        logger.warning(f"Vertex AI Prompt Management 'before' logging skipped: {e}")

    refined_prompt = await run_anchored_academic_multi_agent(
        project_id=project_id,
        model_id=configs.agent_settings.model,
    )

    if vx_client is not None:
        try:
            after_prompt = vx_client.prompts.create(
                prompt_id=f"cda_3oh_refined_prompt_{timestamp}",
                prompt=vx_types.Prompt(
                    prompt_data=vx_types.PromptData(
                        contents=[
                            generative_models.Content(
                                role="user",
                                parts=[generative_models.Part.from_text(refined_prompt)],
                            )
                        ],
                        model=configs.agent_settings.model,
                    )
                ),
            )
            logger.info(f"Saved optimized 'after' prompt to Vertex AI Prompt Management: {after_prompt.name}")
        except Exception as e:
            logger.warning(f"Vertex AI Prompt Management 'after' logging skipped: {e}")

    return refined_prompt
