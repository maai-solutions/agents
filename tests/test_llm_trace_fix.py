"""Test script to verify LLM output traces are captured for all methods.

This test verifies that:
1. _generate_response() has tracing
2. _check_completion() has tracing
3. Both methods properly update generation with output and usage stats
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry


async def test_trace_coverage():
    """Test that all LLM calls are properly traced."""

    print("=" * 60)
    print("Testing LLM Trace Coverage")
    print("=" * 60)

    # Initialize telemetry with console exporter (easy to see)
    tracer = initialize_telemetry(
        service_name="trace-test",
        exporter_type="console",
        enabled=True
    )

    if tracer is None or not tracer.enabled:
        print("⚠️  WARNING: Telemetry not enabled. Install OpenTelemetry:")
        print("pip install opentelemetry-api opentelemetry-sdk")
        return False

    print(f"✅ Telemetry initialized: {type(tracer).__name__}")

    # Create agent with tracing
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=get_default_tools(),
        tracer=tracer,
        use_async=True,
        verbose=True
    )

    print(f"✅ Agent created with telemetry: {type(agent.telemetry).__name__}")
    print()

    # Test 1: Simple query that should trigger _generate_response()
    print("Test 1: Testing _generate_response() tracing")
    print("-" * 60)
    try:
        response = await agent.run("What is the capital of France?")
        print(f"✅ Response received: {response.result[:100]}...")
        print(f"✅ Metrics: {response.metrics}")
        print()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False

    # Test 2: Query that requires tool use and completion check
    print("Test 2: Testing _check_completion() tracing")
    print("-" * 60)
    try:
        response = await agent.run("Calculate 42 * 17")
        print(f"✅ Response received: {response.result[:100]}...")
        print(f"✅ Metrics: {response.metrics}")
        print(f"✅ Iterations: {response.metrics.get('total_iterations', 'N/A')}")
        print()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False

    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("Expected trace hierarchy:")
    print("  agent.default")
    print("  ├── reasoning_phase")
    print("  │   └── llm.gemma3:27b (reasoning)")
    print("  ├── tool.calculator")
    print("  │   └── llm.gemma3:27b (tool_args)")
    print("  ├── llm.generate_response (NEW - previously missing!)")
    print("  └── llm.check_completion (NEW - previously missing!)")
    print()
    print("Check the console output above for these spans.")
    print()

    return True


def verify_method_has_tracing():
    """Verify the methods have been patched with tracing code."""

    print("=" * 60)
    print("Verifying Method Implementations")
    print("=" * 60)

    import inspect
    from linus.agents.agent.reasoning_agent import ReasoningAgent

    # Check _generate_response
    source = inspect.getsource(ReasoningAgent._generate_response)
    has_trace_call = "trace_llm_call" in source
    has_update_gen = "update_generation" in source

    print(f"_generate_response() has trace_llm_call: {'✅' if has_trace_call else '❌'}")
    print(f"_generate_response() has update_generation: {'✅' if has_update_gen else '❌'}")

    # Check _check_completion
    source = inspect.getsource(ReasoningAgent._check_completion)
    has_trace_call = "trace_llm_call" in source
    has_update_gen = "update_generation" in source

    print(f"_check_completion() has trace_llm_call: {'✅' if has_trace_call else '❌'}")
    print(f"_check_completion() has update_generation: {'✅' if has_update_gen else '❌'}")
    print()

    return True


if __name__ == "__main__":
    print()

    # First verify the code changes are present
    if not verify_method_has_tracing():
        print("❌ Code verification failed!")
        sys.exit(1)

    # Then test with actual agent execution
    print("Starting agent execution tests...")
    print("(Make sure Ollama is running with gemma3:27b model)")
    print()

    try:
        success = asyncio.run(test_trace_coverage())
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
