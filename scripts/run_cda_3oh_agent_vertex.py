import os
import sys
import asyncio
import logging

import certifi
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.runners import Runner
from google.genai.types import Content, Part

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["GOOGLE_API_USE_MTLS"] = "never"
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cda_3oh_agent.config as cfg_mod
import cda_3oh_agent.agent as agent_mod
import cda_3oh_agent.sub_agent.workflow as wf_mod
import cda_3oh_agent.sub_agent.data_fetch_and_parse as dfp_mod
import cda_3oh_agent.sub_agent.prompt_optimizer as po_mod
import cda_3oh_agent.sub_agent.few_shot_annotation as fsa_mod
import cda_3oh_agent.sub_agent.data_prep as dp_mod
import cda_3oh_agent.sub_agent.batch_submission as bs_mod
import cda_3oh_agent.sub_agent.batch_polling as bp_mod
import cda_3oh_agent.sub_agent.result_analysis as ra_mod
import cda_3oh_agent.sub_agent.vertex_ai_logging as vl_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    logger.info("Initializing Forked CDA Binary 3-OH Agent (cda_3oh_agent) on Vertex AI with Gemini 4 Argon...")

    for mod in [
        cfg_mod,
        agent_mod,
        wf_mod,
        dfp_mod,
        po_mod,
        fsa_mod,
        dp_mod,
        bs_mod,
        bp_mod,
        ra_mod,
        vl_mod,
    ]:
        if hasattr(mod, "configs"):
            mod.configs.agent_settings.model = "gemini-4-argon"
            mod.configs.agent_settings.inference_path = "batch"
            mod.configs.agent_settings.test_n = -1
            mod.configs.agent_settings.few_shot_n = 18
            mod.configs.agent_settings.fetch_data = False
            mod.configs.agent_settings.prompt_optimization = False
            mod.configs.agent_settings.force_new_annotations = False
            mod.configs.agent_settings.test_mode = False

    wf_mod.few_shot_agent.model_id = "gemini-4-argon"
    wf_mod.batch_submit_agent.model_id = "gemini-4-argon"

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="cda_3oh_app",
        user_id="turan_user",
        session_id="session_3oh_full_argon_01",
    )

    runner = Runner(
        agent=agent_mod.root_agent,
        app_name="cda_3oh_app",
        session_service=session_service,
    )

    msg = Content(
        role="user",
        parts=[
            Part.from_text(
                text=(
                    "Execute the binary 3-OH vs non-3-OH classification workflow on the full "
                    "labelled test dataset using Gemini 4 Argon, starting from the anchor AI-generated "
                    "prompt and optimizing it via the Academic Research/Critic/Referee multi-agent pipeline."
                )
            )
        ],
    )

    async for event in runner.run_async(
        user_id="turan_user",
        session_id="session_3oh_full_argon_01",
        new_message=msg,
    ):
        if event.content and event.content.parts:
            for p in event.content.parts:
                if p.text:
                    print(f"[{event.author}] {p.text}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
