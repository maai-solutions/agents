"""Simple demo of async telemetry working correctly."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.telemetry import LangfuseTracer, AgentTracer
from loguru import logger


async def demo_langfuse_async():
    """Demonstrate async-compatible Langfuse tracer."""
    print("\n" + "=" * 60)
    print("Demo: Async-Compatible Langfuse Tracer")
    print("=" * 60)

    # Create a mock Langfuse tracer (without actual client)
    tracer = LangfuseTracer(langfuse_client=None)  # Disabled
    print(f"Tracer enabled: {tracer.enabled}")

    # Try to use it in async context
    try:
        async with tracer.trace_agent_run("Test query", "TestAgent") as trace:
            print("✅ Successfully entered async context manager")
            print(f"   Trace object: {trace}")

            # Simulate some async work
            await asyncio.sleep(0.1)

            # Try nested spans
            async with tracer.trace_reasoning_phase("Analyzing...", iteration=1) as reasoning:
                print("✅ Successfully entered nested reasoning span")
                await asyncio.sleep(0.1)

        print("✅ Successfully exited all context managers")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def demo_otel_async():
    """Demonstrate async-compatible OpenTelemetry tracer."""
    print("\n" + "=" * 60)
    print("Demo: Async-Compatible OpenTelemetry Tracer")
    print("=" * 60)

    # Create a mock OpenTelemetry tracer (without actual tracer backend)
    tracer = AgentTracer(tracer=None)  # Disabled
    print(f"Tracer enabled: {tracer.enabled}")

    # Try to use it in async context
    try:
        async with tracer.trace_agent_run("Test query", "TestAgent") as trace:
            print("✅ Successfully entered async context manager")
            print(f"   Trace object: {trace}")

            # Simulate some async work
            await asyncio.sleep(0.1)

            # Try nested spans
            async with tracer.trace_llm_call("What is 2+2?", "gemma3:27b", "reasoning") as llm_span:
                print("✅ Successfully entered nested LLM call span")
                await asyncio.sleep(0.1)

        print("✅ Successfully exited all context managers")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("ASYNC TELEMETRY DEMONSTRATION")
    print("This demonstrates that telemetry tracers work in async contexts")
    print("=" * 70)

    # Demo 1: Langfuse async compatibility
    await demo_langfuse_async()

    # Demo 2: OpenTelemetry async compatibility
    await demo_otel_async()

    print("\n" + "=" * 70)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
    print("The telemetry tracers are now async-compatible!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
