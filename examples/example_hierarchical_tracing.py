"""Example demonstrating hierarchical naming in traces.

This example shows how to use hierarchical naming for traces:
- agent.<name>: For agent execution traces
- llm.<name>: For LLM call traces
- tool.<name>: For tool execution traces

Run this example with:
    python example_hierarchical_tracing.py
"""

import asyncio
import os
from dotenv import load_dotenv

from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

# Load environment variables
load_dotenv()


async def main():
    """Demonstrate hierarchical tracing with multiple agents."""

    print("=" * 80)
    print("Hierarchical Tracing Example")
    print("=" * 80)
    print()

    # Initialize telemetry with Langfuse (or change to "console", "otlp", "jaeger")
    exporter_type = os.getenv("TELEMETRY_EXPORTER", "langfuse")
    telemetry_enabled = os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"

    print(f"Telemetry exporter: {exporter_type}")
    print(f"Telemetry enabled: {telemetry_enabled}")
    print()

    # Initialize telemetry tracer with a default agent name
    tracer = initialize_telemetry(
        service_name="hierarchical-tracing-demo",
        exporter_type=exporter_type,
        enabled=telemetry_enabled,
        agent_name="calculator_agent"  # Default agent name
    )

    # Example 1: Create an agent with a specific name for math calculations
    print("Example 1: Math Calculator Agent")
    print("-" * 80)

    calculator_agent = Agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        tools=get_default_tools(),
        verbose=True,
        use_async=True,
        tracer=tracer,
        agent_name="calculator"  # This creates traces with "agent.calculator"
    )

    # Run a calculation task
    # This will create traces like:
    # - agent.calculator (top-level trace)
    # - llm.reasoning (LLM call for reasoning)
    # - llm.tool_args (LLM call for generating tool arguments)
    # - tool.calculator (tool execution)
    response = await calculator_agent.run("What is 42 * 17 + 100?")

    print(f"\nResult: {response.result}")
    print(f"Metrics: {response.metrics.to_dict()}")
    print()

    # Example 2: Create an agent with a different name for search tasks
    print("Example 2: Search Agent")
    print("-" * 80)

    search_agent = Agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        tools=get_default_tools(),
        verbose=True,
        use_async=True,
        tracer=tracer,
        agent_name="search"  # This creates traces with "agent.search"
    )

    # Run a search task
    # This will create traces like:
    # - agent.search (top-level trace)
    # - llm.reasoning (LLM call for reasoning)
    # - llm.tool_args (LLM call for generating tool arguments)
    # - tool.search (tool execution)
    response = await search_agent.run("Search for information about Python programming")

    print(f"\nResult: {response.result}")
    print(f"Metrics: {response.metrics.to_dict()}")
    print()

    # Example 3: Create an agent for file operations
    print("Example 3: File Operations Agent")
    print("-" * 80)

    file_agent = Agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        tools=get_default_tools(),
        verbose=True,
        use_async=True,
        tracer=tracer,
        agent_name="file_ops"  # This creates traces with "agent.file_ops"
    )

    # Run a file operation task
    # This will create traces like:
    # - agent.file_ops (top-level trace)
    # - llm.reasoning (LLM call for reasoning)
    # - llm.tool_args (LLM call for generating tool arguments)
    # - tool.read_file (tool execution)
    response = await file_agent.run("Read the contents of README.md")

    print(f"\nResult: {response.result}")
    print(f"Metrics: {response.metrics.to_dict()}")
    print()

    # Flush traces to ensure they're sent
    if hasattr(tracer, 'flush'):
        tracer.flush()
        print("✓ Traces flushed to telemetry backend")

    print()
    print("=" * 80)
    print("Hierarchical Tracing Complete")
    print("=" * 80)
    print()
    print("Trace Naming Convention:")
    print("  • agent.<name>  - Top-level agent execution (e.g., agent.calculator)")
    print("  • llm.<name>    - LLM calls (e.g., llm.reasoning, llm.tool_args)")
    print("  • tool.<name>   - Tool executions (e.g., tool.calculator, tool.search)")
    print()

    if exporter_type == "langfuse":
        print("View your traces in Langfuse:")
        langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        print(f"  {langfuse_host}")
    elif exporter_type == "console":
        print("Traces have been printed to console above")

    print()


if __name__ == "__main__":
    asyncio.run(main())
