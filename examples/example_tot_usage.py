"""Example usage of TreeOfThoughtAgent.

This script demonstrates how to use the Tree of Thought agent for complex reasoning tasks.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linus.agents.agent.factory import TreeOfThought
from linus.agents.agent.tools import get_default_tools


async def example_basic_tot():
    """Example 1: Basic Tree of Thought agent."""
    print("=" * 80)
    print("Example 1: Basic Tree of Thought Agent")
    print("=" * 80)

    # Create a ToT agent with default tools
    tot_agent = TreeOfThought(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=get_default_tools(),
        verbose=True,
        temperature=0.7,
        enable_reflection=True,
        enable_tool_filtering=True
    )

    # Run a complex query
    query = "Calculate the result of 42 * 17, then search for information about the number you got."

    print(f"\nQuery: {query}\n")

    response = await tot_agent.run(query)

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(response.result)

    print("\n" + "=" * 80)
    print("Metrics:")
    print("=" * 80)
    print(f"Total Iterations: {response.metrics.total_iterations}")
    print(f"Execution Time: {response.metrics.execution_time_seconds:.2f}s")
    print(f"Tool Executions: {response.metrics.tool_executions}")
    print(f"LLM Calls: {response.metrics.llm_calls}")
    print(f"Total Tokens: {response.metrics.total_tokens}")


async def example_tot_with_separate_reasoning_model():
    """Example 2: ToT agent with separate reasoning model."""
    print("\n\n" + "=" * 80)
    print("Example 2: ToT Agent with Separate Reasoning Model")
    print("=" * 80)

    # Create a ToT agent using a different model for reasoning
    # This is useful when you want to use a more powerful reasoning model
    # like DeepSeek-R1 or Qwen2.5 for planning, while using a faster model
    # like Gemma3 for execution
    tot_agent = TreeOfThought(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",  # Fast execution model
        reasoning_model="qwen2.5:7b",  # More powerful reasoning model
        api_key="not-needed",
        tools=get_default_tools(),
        verbose=True,
        temperature=0.5,  # Lower temperature for execution
        reasoning_temperature=0.9,  # Higher temperature for creative reasoning
        enable_reflection=True,
        enable_tool_filtering=True
    )

    query = "What is the current date and time? Then calculate how many days until the end of the year."

    print(f"\nQuery: {query}\n")

    response = await tot_agent.run(query)

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(response.result)

    print("\n" + "=" * 80)
    print("Execution History:")
    print("=" * 80)
    for item in response.execution_history:
        print(f"\nStep {item.get('step', '?')}: {item.get('description', 'N/A')}")
        print(f"  Tool: {item.get('tool', 'None')}")
        print(f"  Status: {item.get('status', 'unknown')}")
        print(f"  Result: {str(item.get('result', ''))[:100]}...")


async def example_tot_without_reflection():
    """Example 3: ToT agent without reflection (faster but less refined)."""
    print("\n\n" + "=" * 80)
    print("Example 3: ToT Agent Without Reflection")
    print("=" * 80)

    # Create a ToT agent with reflection disabled
    # This is faster but may produce less refined plans
    tot_agent = TreeOfThought(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=get_default_tools(),
        verbose=True,
        temperature=0.7,
        enable_reflection=False,  # Disable reflection for speed
        enable_tool_filtering=True
    )

    query = "Calculate 100 + 200 and then multiply the result by 3."

    print(f"\nQuery: {query}\n")

    response = await tot_agent.run(query)

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(response.result)

    print("\n" + "=" * 80)
    print("Metrics:")
    print("=" * 80)
    print(f"Execution Time: {response.metrics.execution_time_seconds:.2f}s")
    print(f"Tool Executions: {response.metrics.tool_executions}")


async def example_tot_without_tool_filtering():
    """Example 4: ToT agent without tool filtering (uses all tools)."""
    print("\n\n" + "=" * 80)
    print("Example 4: ToT Agent Without Tool Filtering")
    print("=" * 80)

    # Create a ToT agent without tool filtering
    # This makes all tools available for every task
    tot_agent = TreeOfThought(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=get_default_tools(),
        verbose=True,
        temperature=0.7,
        enable_reflection=True,
        enable_tool_filtering=False  # Disable tool filtering
    )

    query = "Read the README.md file if it exists."

    print(f"\nQuery: {query}\n")

    response = await tot_agent.run(query)

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(response.result)


async def example_tot_comparison():
    """Example 5: Compare ToT agent with and without features."""
    print("\n\n" + "=" * 80)
    print("Example 5: Feature Comparison")
    print("=" * 80)

    query = "Calculate the square root of 144 and then multiply it by 7."

    # Configuration 1: Full ToT features
    print("\n--- Configuration 1: Full ToT (Reflection + Tool Filtering) ---")
    tot_full = TreeOfThought(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        verbose=False,
        enable_reflection=True,
        enable_tool_filtering=True
    )

    result1 = await tot_full.run(query)
    print(f"Time: {result1.metrics.execution_time_seconds:.2f}s")
    print(f"Tool calls: {result1.metrics.tool_executions}")
    print(f"Result: {result1.result[:200]}...")

    # Configuration 2: No reflection
    print("\n--- Configuration 2: No Reflection (Faster) ---")
    tot_no_reflect = TreeOfThought(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        verbose=False,
        enable_reflection=False,
        enable_tool_filtering=True
    )

    result2 = await tot_no_reflect.run(query)
    print(f"Time: {result2.metrics.execution_time_seconds:.2f}s")
    print(f"Tool calls: {result2.metrics.tool_executions}")
    print(f"Result: {result2.result[:200]}...")

    # Configuration 3: No features
    print("\n--- Configuration 3: Minimal ToT (No Reflection, No Filtering) ---")
    tot_minimal = TreeOfThought(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        verbose=False,
        enable_reflection=False,
        enable_tool_filtering=False
    )

    result3 = await tot_minimal.run(query)
    print(f"Time: {result3.metrics.execution_time_seconds:.2f}s")
    print(f"Tool calls: {result3.metrics.tool_executions}")
    print(f"Result: {result3.result[:200]}...")


async def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("TreeOfThoughtAgent Examples")
    print("=" * 80)
    print("\nThese examples demonstrate the Tree of Thought agent's capabilities:")
    print("- Multi-phase reasoning with initial thought and reflection")
    print("- Intelligent tool selection and filtering")
    print("- Support for separate reasoning and execution models")
    print("- Configurable reflection and tool filtering")
    print("\n")

    # Run examples
    try:
        await example_basic_tot()
        await example_tot_with_separate_reasoning_model()
        await example_tot_without_reflection()
        await example_tot_without_tool_filtering()
        await example_tot_comparison()

        print("\n\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
