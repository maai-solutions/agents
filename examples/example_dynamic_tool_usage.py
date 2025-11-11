"""
Minimal example showing dynamic tool loading with the ReasoningAgent.

This demonstrates the simplest way to use dynamic tool loading.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import load_tools_from_directory, get_default_tools


async def main():
    """Run agent with dynamically loaded tools."""

    # Option 1: Load all tools from a directory
    print("=" * 60)
    print("Option 1: Load all tools from custom_tools directory")
    print("=" * 60)

    custom_tools = load_tools_from_directory("examples/custom_tools")
    print(f"\nLoaded {len(custom_tools)} custom tools:")
    for tool in custom_tools:
        print(f"  - {tool.name}: {tool.description}")

    agent1 = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=custom_tools,
        use_async=True
    )

    print("\nQuerying agent with custom tools...")
    response = await agent1.run("What's the weather in Tokyo?")
    print(f"Response: {response.result}\n")

    # Option 2: Mix default and custom tools
    print("=" * 60)
    print("Option 2: Mix default and custom tools")
    print("=" * 60)

    default_tools = get_default_tools()
    custom_tools = load_tools_from_directory("examples/custom_tools", ["weather"])
    all_tools = default_tools + custom_tools

    print(f"\nLoaded {len(all_tools)} total tools:")
    for tool in all_tools:
        print(f"  - {tool.name}: {tool.description}")

    agent2 = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=all_tools,
        use_async=True
    )

    print("\nQuerying agent with mixed tools...")
    response = await agent2.run("Calculate 42 * 17 and tell me the weather in Paris")
    print(f"Response: {response.result}\n")

    # Option 3: Use ToolRegistry for advanced control
    print("=" * 60)
    print("Option 3: Use ToolRegistry for advanced control")
    print("=" * 60)

    from linus.agents.agent.tools import create_tool_registry, SearchTool

    registry = create_tool_registry(
        tools=[
            SearchTool(),   # Pre-defined tool
            "weather",      # Load from file
            "translator"    # Load from file
        ],
        tools_directory="examples/custom_tools"
    )

    print(f"\nRegistry contains {len(registry.tool_map)} tools:")
    for tool_name, tool in registry.tool_map.items():
        print(f"  - {tool_name}: {tool.description}")

    # Get OpenAI-compatible schemas
    print("\nOpenAI-compatible schemas:")
    schemas = registry.get_tools()
    for schema in schemas:
        print(f"  - {schema['function']['name']}")

    # Use tools from registry
    tools_list = list(registry.tool_map.values())
    agent3 = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=tools_list,
        use_async=True
    )

    print("\nQuerying agent with registry tools...")
    response = await agent3.run("Search for Python tutorials")
    print(f"Response: {response.result}\n")


if __name__ == "__main__":
    print("\n╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Dynamic Tool Loading Examples" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝\n")

    asyncio.run(main())

    print("\n╔" + "=" * 58 + "╗")
    print("║" + " " * 23 + "Done!" + " " * 30 + "║")
    print("╚" + "=" * 58 + "╝\n")
