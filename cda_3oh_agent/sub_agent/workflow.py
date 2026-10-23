import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config
from .data_fetch_and_parse import DataFetchAndParseAgent
from .prompt_optimizer import PromptOptimizerAgent
from .few_shot_annotation import FewShotAnnotationAgent
from .data_prep import DataPrepAgent
from .batch_submission import BatchSubmissionAgent
from .batch_polling import BatchPollingAgent
from .result_analysis import ResultAnalysisAgent
from .vertex_ai_logging import VertexAIExperimentsLoggingAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))


configs = Config()

data_fetch_agent = DataFetchAndParseAgent(name="DataFetchAndParseAgent")
prompt_optimizer_agent = PromptOptimizerAgent(name="PromptOptimizerAgent")
few_shot_agent = FewShotAnnotationAgent(
    name="FewShotAnnotationAgent",
    project_id=configs.agent_settings.project_id,
    location=configs.agent_settings.location,
    model_id=configs.agent_settings.model,
)
data_prep_agent = DataPrepAgent(
    name="DataPrepAgent",
    project_id=configs.agent_settings.project_id,
    bucket_name=configs.agent_settings.bucket_name,
)
batch_submit_agent = BatchSubmissionAgent(
    name="BatchSubmissionAgent",
    project_id=configs.agent_settings.project_id,
    location=configs.agent_settings.location,
    model_id=configs.agent_settings.model,
)
batch_poll_agent = BatchPollingAgent(name="BatchPollingAgent")
result_analysis_agent = ResultAnalysisAgent(
    name="ResultAnalysisAgent",
    project_id=configs.agent_settings.project_id,
)
vertex_ai_logging_agent = VertexAIExperimentsLoggingAgent(
    name="VertexAIExperimentsLoggingAgent",
    project_id=configs.agent_settings.project_id,
)


class Cda3ohWorkflowAgent(BaseAgent):
    """Orchestrator agent that runs the end-to-end binary 3-OH vs non-3-OH CDA classification workflow."""

    data_fetch_agent: BaseAgent
    prompt_optimizer_agent: BaseAgent
    few_shot_agent: BaseAgent
    data_prep_agent: BaseAgent
    batch_submit_agent: BaseAgent
    batch_poll_agent: BaseAgent
    result_analysis_agent: BaseAgent
    vertex_ai_logging_agent: BaseAgent

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        yield log_and_yield(self.name, "Starting CDA Binary 3-OH vs non-3-OH Workflow Pipeline...")

        # 0. Data Fetch and Parse
        async for event in self.data_fetch_agent.run_async(ctx):
            yield event

        # 0.5 Anchored Prompt Optimization (Academic Research -> Critic -> Referee on Gemini 4 Argon)
        async for event in self.prompt_optimizer_agent.run_async(ctx):
            yield event

        # 1. Few-Shot Annotation (18 stratified exemplars annotated by Gemini 4 Argon)
        async for event in self.few_shot_agent.run_async(ctx):
            yield event

        if ctx.session.state.get("fsa_state") != "done":
            yield log_and_yield(self.name, "Stopping workflow due to FewShotAnnotationAgent failure.")
            return

        # 2. Data Prep (Cloud Dataflow + Stage-1 Noise Pre-Filter)
        async for event in self.data_prep_agent.run_async(ctx):
            yield event

        if ctx.session.state.get("dp_state") != "done":
            yield log_and_yield(self.name, "Stopping workflow due to DataPrepAgent failure.")
            return

        # 3. Vertex AI Batch Submission (Gemini 4 Argon)
        async for event in self.batch_submit_agent.run_async(ctx):
            yield event

        if ctx.session.state.get("bs_state") != "done":
            yield log_and_yield(self.name, "Stopping workflow due to BatchSubmissionAgent failure.")
            return

        # 4. Batch Polling
        async for event in self.batch_poll_agent.run_async(ctx):
            yield event

        if not ctx.session.state.get("job_success"):
            yield log_and_yield(self.name, "Stopping workflow because Batch Job did not succeed.")
            return

        # 5. Result Analysis (Cloud Dataflow extraction + Stage-1 merge + Binary & 12-Subclass metrics)
        async for event in self.result_analysis_agent.run_async(ctx):
            yield event

        # 6. Vertex AI Experiments Logging
        async for event in self.vertex_ai_logging_agent.run_async(ctx):
            yield event

        yield log_and_yield(self.name, "CDA Binary 3-OH Workflow Pipeline Completed Successfully!")
