from typing import AsyncGenerator
from typing_extensions import override
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()

from .data_fetch_and_parse import DataFetchAndParseAgent
from .prompt_optimizer import PromptOptimizerAgent
from .few_shot_annotation import FewShotAnnotationAgent
from .data_prep import DataPrepAgent
from .local_inference import LocalInferenceAgent
from .batch_submission import BatchSubmissionAgent
from .batch_polling import BatchPollingAgent
from .result_analysis import ResultAnalysisAgent
from .vertex_ai_logging import VertexAIExperimentsLoggingAgent

class CdaWorkflowAgent(BaseAgent):
    """Orchestrates the CDA Dust Analyzer workflow with custom logic."""
    
    data_fetch: DataFetchAndParseAgent
    prompt_opt: PromptOptimizerAgent
    few_shot: FewShotAnnotationAgent
    data_prep: DataPrepAgent
    local_inf: LocalInferenceAgent
    batch_sub: BatchSubmissionAgent
    batch_poll: BatchPollingAgent
    result_analysis: ResultAnalysisAgent
    vertex_logging: VertexAIExperimentsLoggingAgent
    
    def __init__(self, name: str):
        # Instantiate sub-agents
        data_fetch = DataFetchAndParseAgent(name="DataFetchAndParse")
        prompt_opt = PromptOptimizerAgent(name="PromptOptimizer")
        few_shot = FewShotAnnotationAgent(name="FewShotAnnotation", data_path="cda_dust_agent/data/raw/cda_train.parquet")
        limit_val = configs.agent_settings.test_n if configs.agent_settings.test_mode else 0
        data_prep = DataPrepAgent(name="DataPrep", bucket_name=configs.agent_settings.bucket_name, limit=limit_val)
        local_inf = LocalInferenceAgent(name="LocalInference", model_id=configs.agent_settings.model, limit=limit_val)
        batch_sub = BatchSubmissionAgent(name="BatchSubmission", project_id=configs.agent_settings.project_id, model_id=configs.agent_settings.model)
        batch_poll = BatchPollingAgent(name="BatchPolling")
        result_analysis = ResultAnalysisAgent(name="ResultAnalysis", project_id=configs.agent_settings.project_id)
        vertex_logging = VertexAIExperimentsLoggingAgent(name="VertexAIExperimentsLogging", project_id=configs.agent_settings.project_id)
        
        sub_agents = [
            data_fetch, prompt_opt, few_shot, data_prep,
            local_inf, batch_sub, batch_poll, result_analysis,
            vertex_logging
        ]
        
        super().__init__(
            name=name,
            data_fetch=data_fetch,
            prompt_opt=prompt_opt,
            few_shot=few_shot,
            data_prep=data_prep,
            local_inf=local_inf,
            batch_sub=batch_sub,
            batch_poll=batch_poll,
            result_analysis=result_analysis,
            vertex_logging=vertex_logging,
            sub_agents=sub_agents
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        yield log_and_yield(self.name, "Starting CDA Workflow...")
        
        # 1. Data Fetch and Parse
        async for event in self.data_fetch.run_async(ctx): yield event
        
        # 2. Prompt Optimizer
        async for event in self.prompt_opt.run_async(ctx): yield event
        
        # 3. Few Shot Annotation
        async for event in self.few_shot.run_async(ctx): yield event
        
        # 4. Inference (Conditional)
        if configs.agent_settings.inference_path == "local":
            async for event in self.local_inf.run_async(ctx): yield event
        elif configs.agent_settings.inference_path == "batch":
            # Data Prep for batch is only needed here
            async for event in self.data_prep.run_async(ctx): yield event
            async for event in self.batch_sub.run_async(ctx): yield event
            async for event in self.batch_poll.run_async(ctx): yield event
        else:
            yield log_and_yield(self.name, f"Unknown inference path: {configs.agent_settings.inference_path}")
            return
            
        # 5. Result Analysis
        async for event in self.result_analysis.run_async(ctx): yield event
        
        # 6. Vertex AI Logging
        async for event in self.vertex_logging.run_async(ctx): yield event
        
        yield log_and_yield(self.name, "CDA Workflow complete.")
