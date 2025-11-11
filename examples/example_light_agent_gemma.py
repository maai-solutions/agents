"""Example: Using LightAgent with Gemma3:27b (Ollama)

This example demonstrates how LightAgent now works seamlessly with
models that don't support native function calling, like Gemma3:27b.

The agent will automatically detect that Gemma doesn't support native
tools and switch to manual mode (parsing tool calls from text).
"""

import asyncio
from openai import AsyncOpenAI

from linus.agents.agent.light_agent import LightAgent
from linus.agents.agent.tools import get_default_tools


async def main():
    """Run examples with Gemma3:27b."""

    print("=" * 80)
    print("LightAgent with Gemma3:27b Example")
    print("=" * 80)

    # Create OpenAI client for Ollama
    llm = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed"  # Ollama doesn't need an API key
    )

    # Create LightAgent - will auto-detect MANUAL mode for Gemma
    agent = LightAgent(
        llm=llm,
        model="gemma3:27b",
        tools=get_default_tools(),
        instructions="You are a helpful AI assistant that can use tools to help users.",
        temperature=0.7,
        max_tool_iterations=5,
        verbose=False,  # Set to True to see detailed logs
        agent_name="gemma_assistant"
    )

    print(f"\n✓ Agent created")
    print(f"  - Model: {agent.model}")
    print(f"  - Mode: {agent.active_mode.value} (auto-detected)")
    print(f"  - Tools: {len(agent.tools)} available")

    # Example 1: Simple calculation
    print("\n" + "-" * 80)
    print("Example 1: Simple Calculation")
    print("-" * 80)

    query1 = "What is 156 multiplied by 37?"
    print(f"\n📝 Query: {query1}")

    response1 = await agent.run(query1, return_metrics=True)

    print(f"\n✅ Response: {response1.result}")
    print(f"\n📊 Metrics:")
    print(f"   - Iterations: {response1.metrics.total_iterations}")
    print(f"   - Tools used: {response1.metrics.tool_executions}")
    print(f"   - Execution time: {response1.metrics.execution_time_seconds:.2f}s")

    # Example 2: Multi-step calculation
    print("\n" + "-" * 80)
    print("Example 2: Multi-Step Calculation")
    print("-" * 80)

    query2 = "Calculate (25 + 75) * 2, then add 50 to the result"
    print(f"\n📝 Query: {query2}")

    response2 = await agent.run(query2, return_metrics=True)

    print(f"\n✅ Response: {response2.result}")
    print(f"\n📊 Metrics:")
    print(f"   - Iterations: {response2.metrics.total_iterations}")
    print(f"   - Tools used: {response2.metrics.tool_executions}")
    print(f"   - Execution time: {response2.metrics.execution_time_seconds:.2f}s")

    # Example 3: Complex expression
    print("\n" + "-" * 80)
    print("Example 3: Complex Expression")
    print("-" * 80)

    query3 = "What is the result of (100 + 200) / 3?"
    print(f"\n📝 Query: {query3}")

    response3 = await agent.run(query3, return_metrics=True)

    print(f"\n✅ Response: {response3.result}")
    print(f"\n📊 Metrics:")
    print(f"   - Iterations: {response3.metrics.total_iterations}")
    print(f"   - Tools used: {response3.metrics.tool_executions}")
    print(f"   - Execution time: {response3.metrics.execution_time_seconds:.2f}s")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\n✨ LightAgent successfully worked with Gemma3:27b using MANUAL mode!")
    print("\nKey Points:")
    print("  • No code changes needed - auto-detection 'just works'")
    print("  • Tools are called by parsing JSON from model output")
    print("  • Same API as native function calling models")
    print("  • Works with any OpenAI-compatible endpoint")


if __name__ == "__main__":
    print("\n🚀 Starting LightAgent + Gemma3:27b examples...")
    print("   (Make sure Ollama is running with gemma3:27b model)")
    print()

    asyncio.run(main())

    print("\n✅ All examples completed successfully!")
