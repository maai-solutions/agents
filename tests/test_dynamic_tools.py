"""Test script for dynamic tool loading functionality."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.tools import (
    load_tools_from_directory,
    create_tool_registry,
    get_default_tools,
    ToolRegistry,
    ToolLoader
)


async def test_tool_loader():
    """Test the ToolLoader class."""
    print("=" * 60)
    print("Test 1: ToolLoader - Load specific tools")
    print("=" * 60)

    loader = ToolLoader("examples/custom_tools")

    # Discover available tools
    print("\nDiscovering tools in 'custom_tools' directory...")
    available_tools = loader.discover_tools()
    print(f"Found tools: {available_tools}")

    # Load specific tool
    print("\nLoading 'weather' tool...")
    weather_tool = loader.load_tool("weather")
    print(f"Loaded: {weather_tool.name}")
    print(f"Description: {weather_tool.description}")

    # Test the tool
    print("\nTesting weather tool...")
    result = await weather_tool.arun({"city": "London", "units": "celsius"})
    print(f"Result: {result}")

    print("\n" + "=" * 60 + "\n")


async def test_tool_registry():
    """Test the ToolRegistry class."""
    print("=" * 60)
    print("Test 2: ToolRegistry - Register and manage tools")
    print("=" * 60)

    registry = ToolRegistry()

    # Load tools from directory
    loader = ToolLoader("examples/custom_tools")
    weather_tool = loader.load_tool("weather")
    translator_tool = loader.load_tool("translator")

    # Register tools
    print("\nRegistering tools...")
    registry.register_tool(weather_tool)
    registry.register_tool(translator_tool)

    print(f"Registered {len(registry.tool_map)} tools")

    # Get tool schemas
    print("\nTool schemas (OpenAI format):")
    schemas = registry.get_tools()
    for schema in schemas:
        print(f"  - {schema['function']['name']}: {schema['function']['description']}")

    # Get tool by name
    print("\nRetrieving 'weather' tool from registry...")
    tool = registry.get_tool("weather")
    print(f"Retrieved: {tool.name}")

    # Test tool from registry
    print("\nTesting tool from registry...")
    result = await tool.arun({"city": "Paris", "units": "fahrenheit"})
    print(f"Result: {result}")

    print("\n" + "=" * 60 + "\n")


async def test_load_tools_from_directory():
    """Test the load_tools_from_directory helper function."""
    print("=" * 60)
    print("Test 3: load_tools_from_directory - Load all tools at once")
    print("=" * 60)

    # Load all tools from directory
    print("\nLoading all tools from 'examples/custom_tools'...")
    tools = load_tools_from_directory("examples/custom_tools")
    print(f"Loaded {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # Load specific tools only
    print("\nLoading only 'weather' tool...")
    tools = load_tools_from_directory("examples/custom_tools", ["weather"])
    print(f"Loaded {len(tools)} tool(s):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    print("\n" + "=" * 60 + "\n")


async def test_create_tool_registry():
    """Test the create_tool_registry helper function."""
    print("=" * 60)
    print("Test 4: create_tool_registry - Mix of pre-defined and dynamic tools")
    print("=" * 60)

    # Get default tools
    default_tools = get_default_tools()
    print(f"\nDefault tools available: {len(default_tools)}")

    # Create registry with mix of pre-defined and file-based tools
    print("\nCreating registry with mixed tools...")
    registry = create_tool_registry(
        tools=[
            default_tools[0],  # SearchTool
            default_tools[1],  # CalculatorTool
            "weather",         # Load from file
            "translator"       # Load from file
        ],
        tools_directory="examples/custom_tools"
    )

    print(f"\nRegistry contains {len(registry.tool_map)} tools:")
    schemas = registry.get_tools()
    for schema in schemas:
        print(f"  - {schema['function']['name']}: {schema['function']['description']}")

    # Test a dynamically loaded tool
    print("\nTesting dynamically loaded 'translator' tool...")
    translator = registry.get_tool("translator")
    result = await translator.arun({
        "text": "Hello, world!",
        "target_language": "es"
    })
    print(f"Result: {result}")

    print("\n" + "=" * 60 + "\n")


async def test_tool_schemas():
    """Test OpenAI schema generation."""
    print("=" * 60)
    print("Test 5: OpenAI Schema Generation")
    print("=" * 60)

    registry = create_tool_registry(["weather"], "examples/custom_tools")

    print("\nGenerated OpenAI schema:")
    schema_json = registry.get_tools_str()
    print(schema_json)

    print("\n" + "=" * 60 + "\n")


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "DYNAMIC TOOL LOADING TESTS" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    await test_tool_loader()
    await test_tool_registry()
    await test_load_tools_from_directory()
    await test_create_tool_registry()
    await test_tool_schemas()

    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 18 + "ALL TESTS PASSED" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
