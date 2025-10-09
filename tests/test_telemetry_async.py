"""Test async telemetry implementation."""

import asyncio
import os
from dotenv import load_dotenv

from linus.agents.agent import Agent, get_default_tools
from linus.agents.telemetry import initialize_telemetry

# Load environment variables
load_dotenv()


async def test_langfuse_telemetry():
    """Test Langfuse telemetry with async agent."""
    print("=" * 60)
    print("Testing Langfuse Telemetry with Async Agent")
    print("=" * 60)

    # Initialize Langfuse tracer
    tracer = initialize_telemetry(
        service_name="test-async-agent",
        exporter_type="langfuse",
        session_id="test-session-123",
        enabled=True
    )

    print(f"\nTracer initialized: {type(tracer)}")
    print(f"Tracer enabled: {tracer.enabled}")

    # Create async agent
    agent = Agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=0.7,
        tools=get_default_tools(),
        verbose=True,
        use_async=True  # Use async client
    )

    # Override agent's tracer with our Langfuse tracer
    agent.tracer = tracer

    print("\nAgent created. Running test query...")

    # Run a simple query
    try:
        response = await agent.run("What is 10 + 20?", return_metrics=True)
        print("\n" + "=" * 60)
        print("Agent Response:")
        print("=" * 60)
        print(f"Result: {response.result}")
        print(f"\nMetrics:")
        for key, value in response.metrics.to_dict().items():
            print(f"  {key}: {value}")

        # Flush traces to Langfuse
        print("\nFlushing traces to Langfuse...")
        tracer.flush()
        print("✅ Traces flushed successfully!")

    except Exception as e:
        print(f"\n❌ Error during agent run: {e}")
        import traceback
        traceback.print_exc()


async def test_console_telemetry():
    """Test console telemetry with async agent."""
    print("\n\n" + "=" * 60)
    print("Testing Console (OpenTelemetry) Telemetry with Async Agent")
    print("=" * 60)

    # Initialize console tracer
    tracer = initialize_telemetry(
        service_name="test-async-agent-console",
        exporter_type="console",
        enabled=True
    )

    print(f"\nTracer initialized: {type(tracer)}")
    print(f"Tracer enabled: {tracer.enabled}")

    # Create async agent
    agent = Agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=0.7,
        tools=get_default_tools(),
        verbose=True,
        use_async=True  # Use async client
    )

    # Override agent's tracer with our console tracer
    agent.tracer = tracer

    print("\nAgent created. Running test query...")

    # Run a simple query
    try:
        response = await agent.run("Calculate 5 * 7", return_metrics=True)
        print("\n" + "=" * 60)
        print("Agent Response:")
        print("=" * 60)
        print(f"Result: {response.result}")
        print(f"\nMetrics:")
        for key, value in response.metrics.to_dict().items():
            print(f"  {key}: {value}")

        print("\n✅ Console telemetry test completed!")

    except Exception as e:
        print(f"\n❌ Error during agent run: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all telemetry tests."""
    # Test 1: Langfuse telemetry
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        await test_langfuse_telemetry()
    else:
        print("⚠️  Skipping Langfuse test - credentials not found in .env")

    # Test 2: Console telemetry
    await test_console_telemetry()

    print("\n" + "=" * 60)
    print("All telemetry tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
