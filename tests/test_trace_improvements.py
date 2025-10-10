"""Test script to verify improved telemetry tracing with tool result feedback loop."""

import os
from dotenv import load_dotenv
from linus.agents.agent.reasoning_agent import ReasoningAgent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_trace_with_tool_feedback():
    """Test that tool results are properly traced in the feedback loop."""

    # Initialize Langfuse tracer
    tracer = initialize_telemetry(
        service_name="trace-improvement-test",
        exporter_type="langfuse",
        enabled=True,
        session_id="test-trace-feedback-loop"
    )

    print("\n" + "="*80)
    print("Testing Improved Telemetry with Tool Result Feedback Loop")
    print("="*80 + "\n")

    # Create OpenAI client for Ollama
    llm = OpenAI(
        base_url=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "not-needed")
    )

    # Create agent with tracer
    agent = ReasoningAgent(
        llm=llm,
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        tools=get_default_tools(),
        verbose=True,
        temperature=0.7,
        max_iterations=3
    )

    # Override the tracer
    agent.tracer = tracer

    # Test query that will use tools and iterate
    test_query = "What is 42 * 17?"

    print(f"Query: {test_query}\n")
    print("Watch for these new trace events in Langfuse:")
    print("  1. tool_execution_completed - after each tool runs")
    print("  2. context_updated_with_tool_result - when result is added to context")
    print("  3. reasoning_input_with_history - context passed to next reasoning call")
    print("  4. task_completion_checked - completion check results")
    print("  5. continuing_to_next_iteration - iteration loop continuation")
    print("\n" + "-"*80 + "\n")

    # Run the agent
    response = agent.run(test_query, return_metrics=True)

    print("\n" + "-"*80)
    print(f"\nResult: {response.result}")
    print(f"\nMetrics:")
    print(f"  - Iterations: {response.metrics.total_iterations}")
    print(f"  - Tool Executions: {response.metrics.tool_executions}")
    print(f"  - LLM Calls: {response.metrics.llm_calls}")
    print(f"  - Execution Time: {response.metrics.execution_time_seconds:.2f}s")

    print("\n" + "="*80)
    print("Flushing traces to Langfuse...")
    tracer.flush()

    print("\nTest complete!")
    print(f"Check Langfuse UI for session: test-trace-feedback-loop")
    print("Look for the new telemetry events listed above.")
    print("="*80 + "\n")

    return response


if __name__ == "__main__":
    test_trace_with_tool_feedback()
