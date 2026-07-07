import os
import asyncio
from dotenv import load_dotenv

load_dotenv('.env')

from cda_dust_agent.config import Config
from cda_dust_agent.agent import root_agent
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.agents.invocation_context import InvocationContext

async def main():
    print("=================================================================")
    print(" CASSINI CDA DUST AGENT: FULL DATASET (~18K) BATCH RUN ")
    print("=================================================================")
    
    configs = Config()
    configs.agent_settings.inference_path = "batch"
    configs.agent_settings.project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'turan-genai-bb')
    configs.agent_settings.bucket_name = os.environ.get('GCS_BUCKET_NAME', 'turansgenaibb').replace("gs://", "")
    configs.agent_settings.test_mode = False
    configs.agent_settings.prompt_optimization = False
    configs.agent_settings.fetch_data = True
    configs.agent_settings.few_shot_n = 16

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="cda_app", user_id="user_1", session_id="full_batch_session")
    
    ctx = InvocationContext(
        session=session,
        session_service=session_service,
        invocation_id="cda_full_batch_invocation",
        agent=root_agent
    )

    async for event in root_agent._run_async_impl(ctx):
        if event.content and event.content.parts:
            print(f"[{event.author}] {event.content.parts[0].text}")

    print("\n=================================================================")
    print(" FULL DATASET BATCH RUN PIPELINE COMPLETED ")
    print("=================================================================")

if __name__ == "__main__":
    asyncio.run(main())
