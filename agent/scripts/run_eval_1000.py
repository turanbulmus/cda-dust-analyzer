import os
import asyncio
import pandas as pd
from dotenv import load_dotenv

load_dotenv('.env')

from cda_dust_agent.config import Config
from cda_dust_agent.agent import root_agent
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.agents.invocation_context import InvocationContext

async def main():
    print("=================================================================")
    print(" CASSINI CDA DUST AGENT: EVALUATION ON 1,000 SAMPLES ")
    print("=================================================================")
    
    # Configure 1,000 sample evaluation settings
    configs = Config()
    configs.agent_settings.inference_path = "batch"
    configs.agent_settings.project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'turan-genai-bb')
    configs.agent_settings.bucket_name = os.environ.get('GCS_BUCKET_NAME', 'turansgenaibb').replace("gs://", "")
    configs.agent_settings.test_mode = True
    configs.agent_settings.test_n = 1000
    configs.agent_settings.fetch_data = True
    configs.agent_settings.few_shot_n = 16

    # Update limit on root_agent subagents explicitly
    root_agent.data_prep.limit = 1000
    if hasattr(root_agent, 'local_inf'):
        root_agent.local_inf.limit = 1000

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="cda_app", user_id="user_1", session_id="eval_1000_session")
    
    ctx = InvocationContext(
        session=session,
        session_service=session_service,
        invocation_id="cda_1000_eval_invocation",
        agent=root_agent
    )

    async for event in root_agent._run_async_impl(ctx):
        if event.content and event.content.parts:
            print(f"[{event.author}] {event.content.parts[0].text}")

    print("\n=================================================================")
    print(" 1,000 SAMPLE EVALUATION COMPLETED SUCCESSFULLY ")
    print("=================================================================")

if __name__ == "__main__":
    asyncio.run(main())
