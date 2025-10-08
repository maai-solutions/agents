#!/usr/bin/env python3
"""Test script to verify agent iterates and tries alternative tools when one fails."""

import os
from dotenv import load_dotenv
from linus.agents.agent.factory import Agent
from linus.agents.agent.tool_base import BaseTool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class EmptySearchArgs(BaseModel):
    """Arguments for empty search tool."""
    query: str = Field(description="The search query")


class AlternativeSearchArgs(BaseModel):
    """Arguments for alternative search tool."""
    query: str = Field(description="The search query")


class EmptySearchTool(BaseTool):
    """A mock search tool that always returns empty results."""

    name: str = "empty_search"
    description: str = "Search the database for information (may return empty results)"

    def _run(self, query: str) -> str:
        """Run the empty search."""
        print(f"[EmptySearchTool] Searching for: {query}")
        return "No results found."

    @property
    def args_schema(self):
        return EmptySearchArgs


class AlternativeSearchTool(BaseTool):
    """An alternative search tool that returns results."""

    name: str = "alternative_search"
    description: str = "Alternative search method using different data sources"

    def _run(self, query: str) -> str:
        """Run the alternative search."""
        print(f"[AlternativeSearchTool] Searching for: {query}")
        return f"Found relevant information about '{query}': This is a comprehensive answer with useful data from alternative sources."

    @property
    def args_schema(self):
        return AlternativeSearchArgs


def test_agent_tool_iteration():
    """Test that agent tries alternative tools when first one returns no results."""

    print("\n" + "="*80)
    print("Testing Agent Tool Iteration with Empty Results")
    print("="*80 + "\n")

    # Create tools
    tools = [
        EmptySearchTool(),
        AlternativeSearchTool()
    ]

    # Create agent with multiple iteration capacity
    agent = Agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=0.7,
        max_iterations=5,  # Allow multiple iterations
        tools=tools,
        verbose=True
    )

    # Test query that should trigger multiple tool attempts
    query = "What is quantum computing?"

    print(f"\n{'='*80}")
    print(f"User Query: {query}")
    print(f"{'='*80}\n")
    print("Expected behavior:")
    print("1. Agent tries 'empty_search' first")
    print("2. Gets no results")
    print("3. Agent recognizes task is not complete")
    print("4. Agent plans to try 'alternative_search' in next iteration")
    print("5. Gets useful results from alternative tool")
    print("6. Task completes successfully")
    print(f"{'='*80}\n")

    # Run the agent
    response = agent.run(query, return_metrics=True)

    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nFinal Answer:\n{response.result}\n")

    print("\nExecution History:")
    for i, step in enumerate(response.execution_history, 1):
        print(f"\n{i}. Iteration {step['iteration']}")
        print(f"   Task: {step['task']}")
        print(f"   Tool: {step['tool']}")
        print(f"   Status: {step['status']}")
        print(f"   Result: {step['result'][:100]}...")

    print("\nMetrics:")
    metrics = response.metrics.to_dict()
    print(f"  - Total Iterations: {metrics['total_iterations']}")
    print(f"  - Task Completed: {metrics['task_completed']}")
    print(f"  - Tool Executions: {metrics['tool_executions']}")
    print(f"  - Successful Tool Calls: {metrics['successful_tool_calls']}")
    print(f"  - Execution Time: {metrics['execution_time_seconds']:.2f}s")

    print("\nCompletion Status:")
    print(f"  - Is Complete: {response.completion_status['is_complete']}")
    print(f"  - Reasoning: {response.completion_status['reasoning']}")

    # Verify expectations
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    success = True

    # Check if multiple iterations occurred
    if metrics['total_iterations'] < 2:
        print("❌ FAIL: Agent should have performed at least 2 iterations")
        success = False
    else:
        print(f"✅ PASS: Agent performed {metrics['total_iterations']} iterations")

    # Check if both tools were used
    tools_used = set(step['tool'] for step in response.execution_history)
    if 'empty_search' in tools_used and 'alternative_search' in tools_used:
        print("✅ PASS: Agent tried both tools")
    else:
        print(f"❌ FAIL: Agent only used tools: {tools_used}")
        success = False

    # Check if task completed
    if metrics['task_completed']:
        print("✅ PASS: Task completed successfully")
    else:
        print("❌ FAIL: Task did not complete")
        success = False

    # Check if alternative tool was tried after empty one
    if len(response.execution_history) >= 2:
        first_tool = response.execution_history[0]['tool']
        second_tool = response.execution_history[1]['tool']
        if first_tool == 'empty_search' and second_tool == 'alternative_search':
            print("✅ PASS: Agent correctly tried alternative tool after empty result")
        else:
            print(f"❌ FAIL: Tool execution order was: {first_tool} -> {second_tool}")
            success = False

    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")

    return success


if __name__ == "__main__":
    test_agent_tool_iteration()
