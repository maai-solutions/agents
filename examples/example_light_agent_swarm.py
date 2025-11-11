"""Example usage of LightAgent and Swarm.

This script demonstrates how to use LightAgent for simple tasks and
Swarm for multi-agent coordination.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linus.agents.agent.factory import Light
from linus.agents.agent.swarm import Swarm
from linus.agents.agent.tools import get_default_tools, CalculatorTool, SearchTool


async def example_basic_light_agent():
    """Example 1: Basic LightAgent usage."""
    print("=" * 80)
    print("Example 1: Basic LightAgent")
    print("=" * 80)

    # Create a simple LightAgent
    agent = Light(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=get_default_tools(),
        instructions="You are a helpful assistant that can search and calculate.",
        verbose=True
    )

    query = "Calculate 42 * 17"

    print(f"\nQuery: {query}\n")

    response = await agent.run(query)

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(response.result)

    print("\n" + "=" * 80)
    print("Metrics:")
    print("=" * 80)
    print(f"Execution Time: {response.metrics.execution_time_seconds:.2f}s")
    print(f"Tool Executions: {response.metrics.tool_executions}")
    print(f"LLM Calls: {response.metrics.llm_calls}")


async def example_light_agent_with_role():
    """Example 2: LightAgent with specific role."""
    print("\n\n" + "=" * 80)
    print("Example 2: LightAgent with Role")
    print("=" * 80)

    # Create an agent with a specific role
    agent = Light(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=[CalculatorTool()],
        instructions="You are a math tutor helping students learn.",
        role="Math Tutor",
        agent_name="math_tutor",
        verbose=False
    )

    query = "What is the square root of 144?"

    print(f"\nQuery: {query}\n")

    response = await agent.run(query)

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(response.result)


async def example_basic_swarm():
    """Example 3: Basic Swarm with specialized agents."""
    print("\n\n" + "=" * 80)
    print("Example 3: Basic Swarm")
    print("=" * 80)

    # Create specialized agents
    researcher = Light(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=[SearchTool()],
        instructions="You are a research specialist who finds information.",
        agent_name="researcher",
        verbose=False
    )

    calculator = Light(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=[CalculatorTool()],
        instructions="You are a math specialist who performs calculations.",
        agent_name="calculator",
        verbose=False
    )

    # Create swarm
    swarm = Swarm()
    swarm.register(researcher, calculator)

    print(f"\nRegistered agents: {swarm.list_agents()}\n")

    # Test 1: Math query (should route to calculator)
    query1 = "Calculate 100 + 50 * 2"
    print(f"Query 1: {query1}")
    result1 = await swarm.run(query1)
    print(f"Result: {result1.result}\n")

    # Test 2: Research query (should route to researcher)
    query2 = "Search for information about Python programming"
    print(f"Query 2: {query2}")
    result2 = await swarm.run(query2)
    print(f"Result: {result2.result}\n")


async def example_swarm_with_handoff():
    """Example 4: Swarm with explicit task handoff."""
    print("\n\n" + "=" * 80)
    print("Example 4: Swarm with Task Handoff")
    print("=" * 80)

    # Create agents
    analyst = Light(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=[SearchTool()],
        instructions="You analyze information and identify what calculations are needed.",
        agent_name="analyst",
        verbose=False
    )

    calculator = Light(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=[CalculatorTool()],
        instructions="You perform mathematical calculations.",
        agent_name="calculator",
        verbose=False
    )

    # Create swarm
    swarm = Swarm()
    swarm.register(analyst, calculator)

    # Demonstrate explicit handoff
    initial_query = "I need to calculate the ROI on a $10,000 investment that returned $12,500"

    print(f"\nInitial query to analyst: {initial_query}\n")

    # Analyst processes first
    analyst_result = await swarm.run(initial_query, agent_name="analyst")
    print(f"Analyst result: {analyst_result.result}\n")

    # Hand off to calculator for the actual calculation
    handoff_context = "Calculate ROI: (12500 - 10000) / 10000 * 100"
    print(f"Handing off to calculator: {handoff_context}\n")

    calculator_result = await swarm.handoff(
        from_agent="analyst",
        to_agent="calculator",
        context=handoff_context,
        reason="Need to perform ROI calculation"
    )

    print(f"Calculator result: {calculator_result.result}\n")

    # Show handoff history
    print("Handoff History:")
    for handoff in swarm.get_handoff_history():
        print(f"  {handoff.from_agent} -> {handoff.to_agent}: {handoff.reason}")


async def example_swarm_capabilities():
    """Example 5: Inspecting swarm capabilities."""
    print("\n\n" + "=" * 80)
    print("Example 5: Swarm Capabilities")
    print("=" * 80)

    # Create diverse agents
    agents = [
        Light(
            model="gemma3:27b",
            tools=[SearchTool()],
            instructions="Research specialist",
            agent_name="researcher"
        ),
        Light(
            model="gemma3:27b",
            tools=[CalculatorTool()],
            instructions="Math specialist",
            agent_name="calculator"
        ),
        Light(
            model="gemma3:27b",
            tools=[],
            instructions="General purpose assistant",
            role="Assistant",
            agent_name="assistant"
        )
    ]

    swarm = Swarm()
    swarm.register(*agents)

    # Get capabilities
    capabilities = swarm.get_agent_capabilities()

    print("\nSwarm Capabilities:")
    print("=" * 80)

    for name, info in capabilities.items():
        print(f"\nAgent: {name}")
        print(f"  Instructions: {info['instructions']}")
        print(f"  Role: {info['role'] or 'None'}")
        print(f"  Tools: {', '.join(info['tools']) if info['tools'] else 'None'}")
        print(f"  Model: {info['model']}")

    print(f"\nTotal agents: {len(swarm)}")
    print(f"Agent names: {swarm.list_agents()}")


async def example_light_agent_with_history():
    """Example 6: LightAgent with conversation history."""
    print("\n\n" + "=" * 80)
    print("Example 6: LightAgent with Conversation History")
    print("=" * 80)

    agent = Light(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=[CalculatorTool()],
        instructions="You are a helpful math assistant.",
        agent_name="math_assistant",
        verbose=False
    )

    # First query
    query1 = "Calculate 10 + 5"
    print(f"\nQuery 1: {query1}")

    result1 = await agent.run(query1)
    print(f"Response: {result1.result}")

    # Get history
    history = agent.get_history()

    # Second query with history (context-aware)
    query2 = "Now multiply that by 3"
    print(f"\nQuery 2: {query2}")

    result2 = await agent.run(query2, history=history)
    print(f"Response: {result2.result}")

    # Show full conversation
    print("\nFull Conversation History:")
    for msg in agent.get_history():
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"  {role}: {content[:100]}...")


async def example_swarm_shared_state():
    """Example 7: Swarm with shared state."""
    print("\n\n" + "=" * 80)
    print("Example 7: Swarm with Shared State")
    print("=" * 80)

    from linus.agents.graph.state import SharedState

    # Create shared state
    shared_state = SharedState()

    # Create swarm with shared state
    swarm = Swarm(state=shared_state)

    # Create agents that will share state
    agent1 = Light(
        model="gemma3:27b",
        tools=[CalculatorTool()],
        instructions="You calculate and store results in state.",
        agent_name="calculator",
        state=shared_state
    )

    agent2 = Light(
        model="gemma3:27b",
        tools=[],
        instructions="You retrieve and summarize results from state.",
        agent_name="summarizer",
        state=shared_state
    )

    swarm.register(agent1, agent2)

    # Agent 1 calculates and stores
    print("\nStep 1: Calculator agent calculates 42 * 17")
    result1 = await swarm.run("Calculate 42 * 17 and store it", agent_name="calculator")
    print(f"Result: {result1.result}")

    # Manually add to shared state for demonstration
    shared_state.set("calculation_result", "714", source="calculator")

    # Agent 2 retrieves from shared state
    print("\nStep 2: Summarizer checks shared state")
    state_context = shared_state.get_context()
    print(f"Shared state: {state_context}")

    # Show state stats
    stats = shared_state.get_state_stats()
    print(f"\nState stats: {stats}")


async def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("LightAgent and Swarm Examples")
    print("=" * 80)
    print("\nThese examples demonstrate:")
    print("- LightAgent for direct, simple agent tasks")
    print("- Swarm for multi-agent coordination")
    print("- Task handoff between specialized agents")
    print("- Intent detection and routing")
    print("- Shared state across agents")
    print("\n")

    try:
        await example_basic_light_agent()
        await example_light_agent_with_role()
        await example_basic_swarm()
        await example_swarm_with_handoff()
        await example_swarm_capabilities()
        await example_light_agent_with_history()
        await example_swarm_shared_state()

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
