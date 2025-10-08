"""Example: Using OpenTelemetry tracing with the ReasoningAgent.

This example demonstrates how to enable and configure telemetry to monitor:
- Agent execution flow
- LLM calls (reasoning, argument generation)
- Tool executions with arguments and results
- Performance metrics

Prerequisites:
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

Optional backends:
    - Jaeger: docker run -d -p 16686:16686 -p 6831:6831/udp jaegertracing/all-in-one
    - OTLP Collector: Configure OpenTelemetry Collector
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent import Agent, get_default_tools
from linus.agents.telemetry import initialize_telemetry, is_telemetry_available


def example_console_exporter():
    """Example 1: Console exporter (for testing/development)."""

    print("="*60)
    print("Example 1: Console Exporter")
    print("="*60)

    if not is_telemetry_available():
        print("❌ OpenTelemetry not installed. Install with:")
        print("   pip install opentelemetry-api opentelemetry-sdk")
        return

    # Initialize telemetry with console exporter
    initialize_telemetry(
        service_name="reasoning-agent-example",
        exporter_type="console",
        enabled=True
    )
    print("✅ Telemetry initialized with console exporter")

    # Create agent
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        verbose=True
    )

    # Run a query - traces will be printed to console
    print("\n🤖 Running agent query...")
    result = agent.run("What is 42 * 17?")
    print(f"\n📝 Result: {result.result}")
    print(f"📊 Metrics: {result.metrics.to_dict()}")


def example_jaeger_exporter():
    """Example 2: Jaeger exporter (for production/monitoring)."""

    print("\n" + "="*60)
    print("Example 2: Jaeger Exporter")
    print("="*60)
    print("Prerequisite: Jaeger must be running")
    print("  docker run -d -p 16686:16686 -p 6831:6831/udp jaegertracing/all-in-one")
    print("  Access UI at: http://localhost:16686")
    print("="*60)

    if not is_telemetry_available():
        print("❌ OpenTelemetry not installed")
        return

    # Initialize telemetry with Jaeger exporter
    initialize_telemetry(
        service_name="reasoning-agent-jaeger",
        exporter_type="jaeger",
        jaeger_endpoint="localhost",
        enabled=True
    )
    print("✅ Telemetry initialized with Jaeger exporter")

    # Create agent
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        verbose=False  # Less verbose since we have tracing
    )

    # Run multiple queries to generate traces
    queries = [
        "Calculate 100 + 200",
        "What is the current time?",
        "Search for information about OpenTelemetry",
    ]

    print("\n🤖 Running multiple queries (check Jaeger UI)...")
    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.run(query)
        print(f"Result: {result.result}")

    print("\n✅ Traces sent to Jaeger at http://localhost:16686")


def example_otlp_exporter():
    """Example 3: OTLP exporter (for OpenTelemetry Collector)."""

    print("\n" + "="*60)
    print("Example 3: OTLP Exporter")
    print("="*60)
    print("Prerequisite: OTLP Collector must be running on localhost:4317")
    print("="*60)

    if not is_telemetry_available():
        print("❌ OpenTelemetry not installed")
        return

    # Initialize telemetry with OTLP exporter
    initialize_telemetry(
        service_name="reasoning-agent-otlp",
        exporter_type="otlp",
        otlp_endpoint="http://localhost:4317",
        enabled=True
    )
    print("✅ Telemetry initialized with OTLP exporter")

    # Create agent
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
    )

    # Run a query
    print("\n🤖 Running agent query...")
    result = agent.run("Calculate 42 * 17 and tell me the result")
    print(f"\n📝 Result: {result.result}")


def example_fastapi_integration():
    """Example 4: Using telemetry with FastAPI app."""

    print("\n" + "="*60)
    print("Example 4: FastAPI Integration")
    print("="*60)
    print("To enable telemetry in the FastAPI app, set environment variables:")
    print()
    print("  export TELEMETRY_ENABLED=true")
    print("  export TELEMETRY_EXPORTER=console  # or jaeger, otlp")
    print("  export TELEMETRY_OTLP_ENDPOINT=http://localhost:4317")
    print("  export TELEMETRY_JAEGER_ENDPOINT=localhost")
    print()
    print("Then start the app:")
    print("  python src/app.py")
    print()
    print("All API requests will be traced automatically!")
    print("="*60)


if __name__ == "__main__":
    # Run console example
    example_console_exporter()

    # Uncomment to run other examples:
    # example_jaeger_exporter()
    # example_otlp_exporter()
    # example_fastapi_integration()
