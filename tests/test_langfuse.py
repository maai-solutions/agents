"""Test Langfuse integration."""

import os
from dotenv import load_dotenv
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

# Load environment variables
load_dotenv()

def test_langfuse_integration():
    """Test that Langfuse properly captures traces."""

    print("=== Testing Langfuse Integration ===\n")

    # Initialize Langfuse tracer
    print("1. Initializing Langfuse tracer...")
    tracer = initialize_telemetry(
        service_name="test-reasoning-agent",
        exporter_type="langfuse",
        session_id="test-session-001",
        enabled=True
    )

    if not tracer or not tracer.enabled:
        print("❌ Langfuse tracer not enabled! Check your .env file.")
        print(f"   LANGFUSE_PUBLIC_KEY: {os.getenv('LANGFUSE_PUBLIC_KEY')}")
        print(f"   LANGFUSE_SECRET_KEY: {os.getenv('LANGFUSE_SECRET_KEY')}")
        print(f"   LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST')}")
        return False

    print(f"✅ Langfuse tracer initialized: {type(tracer).__name__}")
    print(f"   Session ID: {tracer.session_id}")
    print(f"   Host: {os.getenv('LANGFUSE_HOST')}\n")

    # Create agent with Langfuse tracing
    print("2. Creating agent with tools...")
    agent = Agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=0.7,
        tools=get_default_tools(),
        verbose=True,
        tracer=tracer  # Pass the Langfuse tracer
    )

    print(f"✅ Agent created with tracer: {type(agent.tracer).__name__}\n")

    # Run a simple test query
    print("3. Running test query...")
    test_query = "What is 42 * 17?"

    try:
        response = agent.run(test_query, return_metrics=True)

        print(f"\n✅ Query completed successfully!")
        print(f"   Result: {response.result}")
        print(f"   Iterations: {response.metrics.total_iterations}")
        print(f"   LLM Calls: {response.metrics.llm_calls}")
        print(f"   Tool Executions: {response.metrics.tool_executions}")
        print(f"   Total Tokens: {response.metrics.total_tokens}")

        # Flush traces
        print("\n4. Flushing traces to Langfuse...")
        tracer.flush()
        print("✅ Traces flushed!")

        print(f"\n📊 Check your Langfuse dashboard at: {os.getenv('LANGFUSE_HOST')}")
        print(f"   Look for session: test-session-001")
        print(f"   Trace name: agent_run")

        return True

    except Exception as e:
        print(f"\n❌ Error during query execution: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_langfuse_integration()

    if success:
        print("\n✅ Test completed successfully!")
        print("   If you don't see traces in Langfuse, check:")
        print("   1. Langfuse server is running and accessible")
        print("   2. API keys are correct in .env")
        print("   3. Network connectivity to Langfuse host")
    else:
        print("\n❌ Test failed!")
        print("   Check the error messages above for details")
