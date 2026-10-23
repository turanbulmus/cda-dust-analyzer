import logging
from .config import Config
from .sub_agent.workflow import (
    Cda3ohWorkflowAgent,
    data_fetch_agent,
    prompt_optimizer_agent,
    few_shot_agent,
    data_prep_agent,
    batch_submit_agent,
    batch_poll_agent,
    result_analysis_agent,
    stage2_adjudicator_agent,
    vertex_ai_logging_agent,
)

logger = logging.getLogger(__name__)
configs = Config()

root_agent = Cda3ohWorkflowAgent(
    name=configs.agent_settings.name,
    data_fetch_agent=data_fetch_agent,
    prompt_optimizer_agent=prompt_optimizer_agent,
    few_shot_agent=few_shot_agent,
    data_prep_agent=data_prep_agent,
    batch_submit_agent=batch_submit_agent,
    batch_poll_agent=batch_poll_agent,
    result_analysis_agent=result_analysis_agent,
    stage2_adjudicator_agent=stage2_adjudicator_agent,
    vertex_ai_logging_agent=vertex_ai_logging_agent,
    sub_agents=[
        data_fetch_agent,
        prompt_optimizer_agent,
        few_shot_agent,
        data_prep_agent,
        batch_submit_agent,
        batch_poll_agent,
        result_analysis_agent,
        stage2_adjudicator_agent,
        vertex_ai_logging_agent,
    ],
)
