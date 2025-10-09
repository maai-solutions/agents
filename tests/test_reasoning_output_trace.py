"""Test script to verify reasoning phase and LLM output tracing."""

import asyncio
import os
from dotenv import load_dotenv
from loguru import logger

from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

# Load environment variables
load_dotenv()

async def test_reasoning_output_trace():
    """Test that reasoning phase and LLM output are properly traced."""

    # Initialize Langfuse tracer
    tracer = initialize_telemetry(
        service_name="test-reasoning-output",
        exporter_type="langfuse",
        session_id="test-reasoning-trace-123",
        enabled=True
    )

    logger.info(f"Tracer initialized: {type(tracer)}")
    logger.info(f"Tracer enabled: {tracer.enabled}")

    # Create agent with Langfuse tracing
    agent = Agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=0.7,
        tools=get_default_tools(),
        verbose=True,
        use_async=True,
        tracer=tracer
    )

    logger.info("Running agent query...")

    # Run a simple query
    response = await agent.run("What is 42 * 17?")

    logger.info(f"Response: {response.result}")
    logger.info(f"Metrics: {response.metrics}")

    # Flush traces to Langfuse
    logger.info("Flushing traces to Langfuse...")
    tracer.flush()

    logger.info("✅ Test complete! Check Langfuse dashboard for traces.")
    logger.info("Expected traces:")
    logger.info("  1. agent_run (main trace)")
    logger.info("  2. reasoning_phase (should have output with parsed result)")
    logger.info("  3. llm_reasoning (generation with input/output)")
    logger.info("  4. tool_calculator (if calculator was used)")

if __name__ == "__main__":
    asyncio.run(test_reasoning_output_trace())
