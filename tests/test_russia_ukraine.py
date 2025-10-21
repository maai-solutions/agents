"""Test script to reproduce the Russia-Ukraine query issue."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry
from linus.agents.logging_config import setup_rich_logging
from loguru import logger

# Setup logging
setup_rich_logging(level="DEBUG")

async def main():
    # Initialize Langfuse
    tracer = initialize_telemetry(
        service_name='test-agent',
        exporter_type='langfuse',
        enabled=True
    )

    # Create agent
    agent = Agent(
        api_base='http://localhost:11434/v1',
        model='gemma3:27b',
        tools=get_default_tools(),
        tracer=tracer,
        use_async=True,
        verbose=True
    )

    # Run the Russia-Ukraine query
    query = "There is currently a conflict between Russia and Ukraine. I want to know which cities are involved"

    logger.info(f"Running query: {query}")
    response = await agent.run(query)

    logger.info(f"\n\n{'='*80}")
    logger.info(f"FINAL RESULT:")
    logger.info(f"{'='*80}")
    logger.info(response.result)
    logger.info(f"{'='*80}\n")

    # Flush traces
    if hasattr(tracer, 'flush'):
        tracer.flush()

if __name__ == "__main__":
    asyncio.run(main())
