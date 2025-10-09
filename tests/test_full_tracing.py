"""Test script to verify full tracing without truncation.

This script demonstrates that all inputs, outputs, and tool data
are captured completely without any truncation.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.telemetry import initialize_telemetry
from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools

# Load environment
load_dotenv()

def test_full_tracing():
    """Test that tracing captures complete data without truncation."""

    print("=" * 80)
    print("Testing Full Tracing (No Truncation)")
    print("=" * 80)

    # Initialize telemetry
    print("\n1. Initializing telemetry...")
    tracer = initialize_telemetry(
        service_name="test-full-tracing",
        exporter_type=os.getenv("TELEMETRY_EXPORTER", "langfuse"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        session_id="test-full-tracing-session",
        enabled=os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"
    )

    if tracer.enabled:
        print(f"✓ Telemetry enabled: {type(tracer).__name__}")
    else:
        print("⚠ Telemetry disabled")

    # Create agent
    print("\n2. Creating agent...")
    agent = create_gemma_agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=0.7,
        tools=get_default_tools(),
        tracer=tracer,
        verbose=True
    )
    print("✓ Agent created")

    # Test with a query that will produce long outputs
    print("\n3. Running test query...")
    print("-" * 80)

    # This query is designed to produce a long response
    test_query = """
    Please analyze the following scenario in detail:

    A company needs to optimize their software development workflow.
    They currently have 5 development teams, each working on different microservices.
    The teams face challenges with:
    1. Inconsistent testing practices across teams
    2. Long deployment times (avg 45 minutes per deploy)
    3. Frequent merge conflicts
    4. Lack of automated monitoring
    5. Manual rollback procedures

    Provide a comprehensive analysis with specific recommendations for each challenge.
    Include implementation steps, estimated timelines, and potential risks.
    """

    print(f"Query: {test_query[:100]}...")

    try:
        response = agent.run(test_query, return_metrics=True)

        print("\n" + "-" * 80)
        print("✓ Agent execution completed")

        # Display metrics
        print("\n4. Execution Metrics:")
        print(f"   - Total Iterations: {response.metrics.total_iterations}")
        print(f"   - Total Tokens: {response.metrics.total_tokens}")
        print(f"   - Execution Time: {response.metrics.execution_time_seconds:.2f}s")
        print(f"   - LLM Calls: {response.metrics.llm_calls}")
        print(f"   - Tool Executions: {response.metrics.tool_executions}")
        print(f"   - Task Completed: {response.metrics.task_completed}")

        # Show result length (should be complete, not truncated)
        result_str = str(response.result)
        print(f"\n5. Result Information:")
        print(f"   - Result length: {len(result_str)} characters")
        print(f"   - First 200 chars: {result_str[:200]}...")
        if len(result_str) > 500:
            print(f"   - Last 200 chars: ...{result_str[-200:]}")

        # Flush traces
        print("\n6. Flushing traces to backend...")
        if hasattr(tracer, 'flush'):
            tracer.flush()
            print("✓ Traces flushed")

            if hasattr(tracer, 'client') and tracer.enabled:
                print("\nℹ️  Check your Langfuse dashboard to verify complete traces:")
                langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                print(f"   {langfuse_host}")

        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        print("\n✓ All data should be captured without truncation")
        print("✓ Verify in your observability backend that:")
        print("  - Full prompt is visible (not truncated at 500/1000 chars)")
        print("  - Complete LLM response is captured")
        print("  - Tool arguments and results are complete")
        print("  - Execution history shows full task results")

    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_full_tracing()
