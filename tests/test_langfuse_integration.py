"""Integration test for Langfuse async telemetry."""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.telemetry import initialize_telemetry
from loguru import logger


async def test_real_langfuse():
    """Test with real Langfuse if credentials are available."""
    print("\n" + "=" * 60)
    print("Testing Real Langfuse Integration")
    print("=" * 60)

    # Check if Langfuse credentials are available
    langfuse_public = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY")
    
    if not langfuse_public or not langfuse_secret:
        print("❌ Langfuse credentials not found in environment")
        print("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to test with real Langfuse")
        return False

    print(f"✅ Found Langfuse credentials: {langfuse_public[:8]}...")

    try:
        # Initialize real Langfuse tracer
        tracer = initialize_telemetry(
            service_name="test-fixed-async-agent",
            exporter_type="langfuse",
            langfuse_public_key=langfuse_public,
            langfuse_secret_key=langfuse_secret,
            session_id="integration-test-session",
            enabled=True
        )

        print(f"Tracer initialized: {type(tracer)}")
        print(f"Tracer enabled: {tracer.enabled}")

        if not tracer.enabled:
            print("❌ Tracer is not enabled - check Langfuse installation")
            return False

        # Test the complete tracing workflow
        print("\n1. Creating agent run trace...")
        async with tracer.trace_agent_run("Integration test query", "TestAgent") as trace:
            print(f"✅ Agent trace created: {type(trace)}")
            
            print("\n2. Creating reasoning phase...")
            async with tracer.trace_reasoning_phase("Testing reasoning phase", iteration=1) as reasoning:
                print(f"✅ Reasoning span created: {type(reasoning)}")
                
                print("\n3. Creating LLM call...")
                async with tracer.trace_llm_call("Test prompt for reasoning", "test-model", "reasoning") as llm:
                    print(f"✅ LLM generation created: {type(llm)}")
                    
                    # Simulate updating the generation with output
                    if llm and hasattr(llm, 'end'):
                        llm.end(output="This is the LLM response for the test")
                
                print("\n4. Creating tool execution...")
                async with tracer.trace_tool_execution("test_tool", {"param": "test_value"}) as tool:
                    print(f"✅ Tool span created: {type(tool)}")
                    
                    # Simulate updating the tool span with output
                    if tool and hasattr(tool, 'end'):
                        tool.end(output="Tool execution completed successfully")

        print("\n5. Recording metrics...")
        tracer.record_metrics({
            "total_tokens": 150,
            "execution_time": 2.5,
            "tool_calls": 1,
            "llm_calls": 1
        })

        print("\n6. Flushing to Langfuse...")
        tracer.flush()

        print("\n✅ Integration test completed successfully!")
        print("Check your Langfuse dashboard for the 'integration-test-session' traces")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run integration test."""
    print("=" * 70)
    print("LANGFUSE ASYNC TELEMETRY INTEGRATION TEST")
    print("This tests the fixed implementation with real Langfuse")
    print("=" * 70)

    success = await test_real_langfuse()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 INTEGRATION TEST PASSED!")
        print("The async telemetry is now properly sending traces to Langfuse!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ INTEGRATION TEST FAILED")
        print("Check the error messages above for details")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())