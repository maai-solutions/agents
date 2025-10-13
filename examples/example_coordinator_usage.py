"""Example usage of CoordinatorAgent orchestrating multiple specialized subagents."""

import asyncio
from linus.agents.agent.factory import Agent, Coordinator
from linus.agents.agent.coordinator_agent import SubAgent
from linus.agents.agent.tools import (
    SearchTool,
    CalculatorTool,
    FileReaderTool,
    get_default_tools
)
from linus.agents.telemetry import initialize_telemetry


async def main():
    """Example of using CoordinatorAgent to orchestrate multiple subagents."""

    # Initialize telemetry (optional)
    tracer = initialize_telemetry(
        service_name="coordinator-demo",
        exporter_type="console",  # Change to "langfuse" if you have it configured
        enabled=True
    )

    print("\n" + "="*80)
    print("CoordinatorAgent Demo - Orchestrating Multiple Specialized Subagents")
    print("="*80 + "\n")

    # Create specialized subagents
    print("Creating specialized subagents...")

    # Research agent with search capabilities
    research_agent = Agent(
        model="gemma3:27b",
        tools=[SearchTool()],
        verbose=False,
        agent_name="researcher",
        tracer=tracer
    )

    # Calculator agent for math operations
    calculator_agent = Agent(
        model="gemma3:27b",
        tools=[CalculatorTool()],
        verbose=False,
        agent_name="calculator",
        tracer=tracer
    )

    # File operations agent
    file_agent = Agent(
        model="gemma3:27b",
        tools=[FileReaderTool()],
        verbose=False,
        agent_name="file_operator",
        tracer=tracer
    )

    # General assistant agent with all tools
    general_agent = Agent(
        model="gemma3:27b",
        tools=get_default_tools(),
        verbose=False,
        agent_name="general_assistant",
        tracer=tracer
    )

    # Wrap agents as SubAgents with descriptions
    subagents = [
        SubAgent(
            agent=research_agent,
            name="researcher",
            description="Searches for information using web search capabilities",
            capabilities=["search", "web_research", "information_retrieval"]
        ),
        SubAgent(
            agent=calculator_agent,
            name="calculator",
            description="Performs mathematical calculations and evaluates expressions",
            capabilities=["math", "calculator", "arithmetic", "expressions"]
        ),
        SubAgent(
            agent=file_agent,
            name="file_operator",
            description="Reads and analyzes file contents",
            capabilities=["file_reading", "file_analysis", "text_extraction"]
        ),
        SubAgent(
            agent=general_agent,
            name="general_assistant",
            description="General purpose assistant with multiple tools for various tasks",
            capabilities=["general", "multi_purpose", "flexible"]
        )
    ]

    # Create coordinator
    print("Creating coordinator agent...")
    coordinator = Coordinator(
        model="gemma3:27b",
        subagents=subagents,
        verbose=True,
        max_iterations=10,
        agent_name="coordinator",
        tracer=tracer
    )

    print("\n" + "="*80)
    print("Available Subagents:")
    for sa in subagents:
        print(f"  - {sa.name}: {sa.description}")
        print(f"    Capabilities: {', '.join(sa.capabilities)}")
    print("="*80 + "\n")

    # Example 1: Complex multi-step task
    print("\n" + "="*80)
    print("Example 1: Complex Task Requiring Multiple Agents")
    print("="*80 + "\n")

    query1 = """
    I need help with the following task:
    1. First, search for information about the Fibonacci sequence
    2. Then, calculate the 10th Fibonacci number
    3. Finally, explain the result in simple terms
    """

    print(f"Query: {query1}\n")
    response1 = await coordinator.run(query1, return_metrics=True)

    print("\n" + "-"*80)
    print("Final Result:")
    print("-"*80)
    print(response1.result)
    print("\n")

    if response1.execution_history:
        print("-"*80)
        print("Execution History:")
        print("-"*80)
        for item in response1.execution_history:
            status_emoji = "✅" if item["status"] == "completed" else "❌"
            print(f"{status_emoji} Step {item['step_number']}: {item.get('description', 'N/A')}")
            print(f"   Subagent: {item['subagent']}")
            print(f"   Result: {item['result'][:150]}...")
            print()

    # Example 2: Task requiring replanning
    print("\n" + "="*80)
    print("Example 2: Task That May Require Adaptive Planning")
    print("="*80 + "\n")

    query2 = """
    I need to analyze some data:
    1. Calculate what 15% of 850 is
    2. Then multiply that result by 3
    3. Search for why this calculation might be useful in retail pricing
    """

    print(f"Query: {query2}\n")
    response2 = await coordinator.run(query2, return_metrics=True)

    print("\n" + "-"*80)
    print("Final Result:")
    print("-"*80)
    print(response2.result)
    print("\n")

    # Example 3: Simple task
    print("\n" + "="*80)
    print("Example 3: Simple Task")
    print("="*80 + "\n")

    query3 = "What is 42 multiplied by 17?"

    print(f"Query: {query3}\n")
    response3 = await coordinator.run(query3, return_metrics=True)

    print("\n" + "-"*80)
    print("Final Result:")
    print("-"*80)
    print(response3.result)
    print("\n")

    # Flush telemetry
    if hasattr(tracer, 'flush'):
        tracer.flush()

    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80 + "\n")


async def example_with_file_operations():
    """Example demonstrating file operations coordination."""
    print("\n" + "="*80)
    print("File Operations Coordination Example")
    print("="*80 + "\n")

    # Create specialized agents
    file_agent = Agent(
        model="gemma3:27b",
        tools=[FileReaderTool()],
        verbose=False,
        agent_name="file_reader"
    )

    calculator_agent = Agent(
        model="gemma3:27b",
        tools=[CalculatorTool()],
        verbose=False,
        agent_name="calculator"
    )

    subagents = [
        SubAgent(
            agent=file_agent,
            name="file_reader",
            description="Reads and extracts information from files",
            capabilities=["file_reading", "text_extraction"]
        ),
        SubAgent(
            agent=calculator_agent,
            name="calculator",
            description="Performs calculations",
            capabilities=["math", "calculator"]
        )
    ]

    coordinator = Coordinator(
        model="gemma3:27b",
        subagents=subagents,
        verbose=True,
        agent_name="file_coordinator"
    )

    query = """
    Read the README.md file and count how many times the word 'agent' appears,
    then calculate what percentage that is of the total number of words in the file.
    """

    print(f"Query: {query}\n")
    response = await coordinator.run(query, return_metrics=True)

    print("\n" + "-"*80)
    print("Final Result:")
    print("-"*80)
    print(response.result)
    print("\n")


async def example_plan_recalculation():
    """Example demonstrating plan recalculation on failures."""
    print("\n" + "="*80)
    print("Plan Recalculation Example")
    print("="*80 + "\n")

    search_agent = Agent(
        model="gemma3:27b",
        tools=[SearchTool()],
        verbose=False,
        agent_name="searcher"
    )

    calc_agent = Agent(
        model="gemma3:27b",
        tools=[CalculatorTool()],
        verbose=False,
        agent_name="calculator"
    )

    subagents = [
        SubAgent(
            agent=search_agent,
            name="searcher",
            description="Searches for information",
            capabilities=["search", "information_retrieval"]
        ),
        SubAgent(
            agent=calc_agent,
            name="calculator",
            description="Performs calculations",
            capabilities=["math", "calculator"]
        )
    ]

    coordinator = Coordinator(
        model="gemma3:27b",
        subagents=subagents,
        verbose=True,
        max_iterations=15,
        agent_name="adaptive_coordinator"
    )

    # Query that might require trying different approaches
    query = """
    I need comprehensive information about Python:
    1. Search for the latest Python version
    2. Calculate how many years it's been since Python 1.0 (released in 1994)
    3. If the first search doesn't work, try a different search approach
    """

    print(f"Query: {query}\n")
    response = await coordinator.run(query, return_metrics=True)

    print("\n" + "-"*80)
    print("Final Result:")
    print("-"*80)
    print(response.result)
    print("\n")

    print("-"*80)
    print("Metrics:")
    print("-"*80)
    print(f"Total iterations: {response.metrics.total_iterations}")
    print(f"Execution time: {response.metrics.execution_time_seconds:.2f}s")
    print(f"Task completed: {response.metrics.task_completed}")
    print()


if __name__ == "__main__":
    # Run the main demo
    asyncio.run(main())

    # Uncomment to run additional examples:
    # asyncio.run(example_with_file_operations())
    # asyncio.run(example_plan_recalculation())
