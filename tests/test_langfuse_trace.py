"""Test script to verify Langfuse tracing is working."""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.telemetry import initialize_telemetry
from linus.agents.agent import Agent, get_default_tools
from loguru import logger

async def main():
    """Test Langfuse integration with async agent."""

    logger.info("=" * 60)
    logger.info("Testing Langfuse Integration (Async)")
    logger.info("=" * 60)

    # Check environment variables
    langfuse_public = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not langfuse_public or not langfuse_secret:
        logger.error("❌ LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env")
        return

    logger.info(f"✓ Langfuse credentials found")
    logger.info(f"✓ Langfuse host: {langfuse_host}")

    # Initialize Langfuse tracer with session ID
    session_id = "test-session-123"
    logger.info(f"✓ Creating tracer with session_id: {session_id}")

    tracer = initialize_telemetry(
        service_name="langfuse-test-agent",
        exporter_type="langfuse",
        langfuse_public_key=langfuse_public,
        langfuse_secret_key=langfuse_secret,
        langfuse_host=langfuse_host,
        session_id=session_id,
        enabled=True
    )

    if not tracer or not tracer.enabled:
        logger.error("❌ Failed to initialize Langfuse tracer")
        return

    logger.info("✓ Langfuse tracer initialized")

    # Create async agent with tracer
    logger.info("✓ Creating async agent...")
    agent = Agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=0.7,
        tools=get_default_tools()[:2],  # Just use a couple tools for testing
        verbose=True,
        use_async=True  # Enable async mode
    )

    # Assign the tracer to the agent
    agent.tracer = tracer

    logger.info("✓ Async agent created with Langfuse tracer")

    # Run a simple test query
    test_query = "Calculate 42 * 17"
    logger.info(f"🚀 Running test query: {test_query}")
    logger.info("-" * 60)

    try:
        # Use await since agent.run() is now async
        response = await agent.run(test_query)
        logger.info("-" * 60)
        logger.info(f"✓ Response received: {response.result if hasattr(response, 'result') else response}")

        # Flush traces to Langfuse
        logger.info("✓ Flushing traces to Langfuse...")
        tracer.flush()

        logger.info("=" * 60)
        logger.info("✅ SUCCESS! Check your Langfuse dashboard:")
        logger.info(f"   {langfuse_host}")
        logger.info(f"   Session ID: {session_id}")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"❌ Error running agent: {e}")
        return

if __name__ == "__main__":
    asyncio.run(main())
