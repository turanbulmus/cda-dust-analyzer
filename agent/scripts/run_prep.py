import asyncio
from cda_dust_agent.agent import DataFetchAndParseAgent

async def main():
    agent = DataFetchAndParseAgent(name="Fetcher")
    async for e in agent._run_async_impl(None):
        if e.content and e.content.parts:
            print(e.content.parts[0].text)

asyncio.run(main())
