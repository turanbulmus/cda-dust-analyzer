import os
import asyncio
import sys

# Add parent directory to path to find cda_dust_agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cda_dust_agent.tools.prompt_refinery import run_prompt_refinery

async def main():
    print("Starting Prompt Improvement Study via script...")
    try:
        # Using the new autonomous mode where the agent decides levels
        system_prompt = await run_prompt_refinery(autonomous=True)
        print("Study complete! Outputs generated in cda_dust_agent/data/results/")
        print("Refined prompt saved to cda_dust_agent/refined_prompt.txt")
        print("Documentation saved to cda_dust_agent/prompt_study_documentation.md")
    except Exception as e:
        print(f"Study failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

