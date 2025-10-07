"""Test the ReasoningAgent's iterative loop with task completion validation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/agent_loop_test.log", rotation="10 MB", level="DEBUG")


def test_simple_task():
    """Test a simple task that should complete in one iteration."""
    print("\n" + "="*80)
    print("TEST 1: Simple Calculation (should complete in 1 iteration)")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=True,
        max_iterations=5
    )

    result = agent.run("Calculate 42 * 17")
    print(f"\nFinal Result:\n{result}\n")


def test_multi_step_task():
    """Test a multi-step task requiring sequential tool use."""
    print("\n" + "="*80)
    print("TEST 2: Multi-step Task (may require multiple iterations)")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=True,
        max_iterations=5
    )

    result = agent.run(
        "First, get the current time. Then, calculate 100 + 50. "
        "Finally, search for information about Python programming."
    )
    print(f"\nFinal Result:\n{result}\n")


def test_complex_task():
    """Test a complex task that might need validation and multiple iterations."""
    print("\n" + "="*80)
    print("TEST 3: Complex Task with Validation")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=True,
        max_iterations=10
    )

    result = agent.run(
        "I need you to: "
        "1. Calculate the result of 25 * 4, "
        "2. Then search for information about that number's significance in mathematics, "
        "3. Finally, get the current time to timestamp this research."
    )
    print(f"\nFinal Result:\n{result}\n")


def test_iterative_refinement():
    """Test a task that might need iterative refinement."""
    print("\n" + "="*80)
    print("TEST 4: Task Requiring Iterative Refinement")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=True,
        max_iterations=8
    )

    result = agent.run(
        "Create a comprehensive analysis: "
        "Calculate 365 * 24 to get hours in a year, "
        "then calculate that number * 60 for minutes, "
        "and search for interesting facts about time measurement."
    )
    print(f"\nFinal Result:\n{result}\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("REASONING AGENT - ITERATIVE LOOP TESTS")
    print("Testing agent loop with task completion validation")
    print("="*80)

    # Run tests
    try:
        test_simple_task()
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")

    try:
        test_multi_step_task()
    except Exception as e:
        logger.error(f"Test 2 failed: {e}")

    try:
        test_complex_task()
    except Exception as e:
        logger.error(f"Test 3 failed: {e}")

    try:
        test_iterative_refinement()
    except Exception as e:
        logger.error(f"Test 4 failed: {e}")

    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
