"""Example of using the ReasoningAgent with Langfuse observability.

This example demonstrates:
1. Initializing telemetry with Langfuse
2. Running agent queries with automatic tracing
3. Viewing traces in Langfuse dashboard
"""

import os
from dotenv import load_dotenv
from linus.agents.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry, is_langfuse_available

# Load environment variables
load_dotenv()


def main():
    """Run the Langfuse example."""

    # Check if Langfuse is available
    if not is_langfuse_available():
        print("❌ Langfuse is not installed.")
        print("Install it with: pip install langfuse")
        return

    # Get Langfuse credentials from environment
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not langfuse_public_key or not langfuse_secret_key:
        print("❌ Langfuse credentials not found.")
        print("Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env file")
        print("\nYou can get your credentials from: https://cloud.langfuse.com")
        return

    print("🚀 Initializing telemetry with Langfuse...")
    print(f"   Host: {langfuse_host}")

    # Initialize telemetry with Langfuse
    tracer = initialize_telemetry(
        service_name="reasoning-agent-example",
        exporter_type="langfuse",
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=langfuse_host,
        enabled=True
    )

    if not tracer.enabled:
        print("❌ Failed to initialize Langfuse tracer")
        return

    print("✅ Langfuse tracer initialized successfully\n")

    # Create agent with default tools
    print("🤖 Creating ReasoningAgent...")
    agent = create_gemma_agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=0.7,
        tools=get_default_tools(),
        verbose=True,
        tracer=tracer  # Pass the Langfuse tracer to the agent
    )

    print("✅ Agent created\n")

    # Example queries
    queries = [
        "What is 42 * 17?",
        "Search for information about Python async programming",
        "Calculate the sum of 100, 200, and 300"
    ]

    print("📋 Running example queries...\n")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 60)

        try:
            # Run the agent - this will automatically create traces in Langfuse
            response = agent.run(query, return_metrics=True)

            print(f"\n✅ Result: {response.result}")
            print(f"\n📊 Metrics:")
            print(f"   - Total iterations: {response.metrics.get('total_iterations', 0)}")
            print(f"   - LLM calls: {response.metrics.get('llm_calls', 0)}")
            print(f"   - Tool executions: {response.metrics.get('tool_executions', 0)}")
            print(f"   - Execution time: {response.metrics.get('execution_time_seconds', 0):.2f}s")

        except Exception as e:
            print(f"\n❌ Error: {e}")

        print("=" * 60)

    # Flush traces to Langfuse
    print("\n📤 Flushing traces to Langfuse...")
    tracer.flush()

    print("\n✅ All queries completed!")
    print(f"\n📊 View your traces at: {langfuse_host}")
    print("   Navigate to your project dashboard to see the agent execution traces")
    print("\n💡 Tip: Each query creates a trace with nested spans for:")
    print("   - Agent execution")
    print("   - Reasoning phase")
    print("   - LLM calls")
    print("   - Tool executions")


if __name__ == "__main__":
    main()
