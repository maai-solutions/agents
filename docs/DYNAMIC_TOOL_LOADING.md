# Dynamic Tool Loading

This document explains how to use the dynamic tool loading feature in the ReasoningAgent framework.

## Overview

Dynamic tool loading allows you to:
- Load tools from Python files at runtime
- Organize tools in separate files for better modularity
- Mix pre-defined tools with dynamically loaded tools
- Automatically discover and load all tools in a directory

This feature is inspired by the lightagent implementation and provides a flexible way to manage tools without hardcoding them in your agent.

## Core Components

### 1. ToolRegistry

The `ToolRegistry` class manages tool registration and converts tools to OpenAI-compatible format.

```python
from linus.agents.agent.tools import ToolRegistry

# Create a registry
registry = ToolRegistry()

# Register tools
registry.register_tool(my_tool)
registry.register_tools([tool1, tool2, tool3])

# Get tool by name
tool = registry.get_tool("weather")

# Get all tool schemas (OpenAI format)
schemas = registry.get_tools()

# Get schemas as JSON string
schemas_json = registry.get_tools_str()
```

### 2. ToolLoader

The `ToolLoader` class dynamically loads tools from Python files.

```python
from linus.agents.agent.tools import ToolLoader

# Create a loader
loader = ToolLoader("custom_tools")

# Discover available tools
available_tools = loader.discover_tools()
# Returns: ["weather", "translator", "calculator"]

# Load a specific tool
weather_tool = loader.load_tool("weather")

# Load multiple tools
tools_dict = loader.load_tools(["weather", "translator"])
```

### 3. Helper Functions

Two convenience functions simplify common use cases:

#### `load_tools_from_directory()`

```python
from linus.agents.agent.tools import load_tools_from_directory

# Load all tools from a directory
tools = load_tools_from_directory("custom_tools")

# Load specific tools only
tools = load_tools_from_directory("custom_tools", ["weather", "calculator"])
```

#### `create_tool_registry()`

```python
from linus.agents.agent.tools import create_tool_registry, SearchTool

# Create empty registry
registry = create_tool_registry()

# Create registry with pre-defined tools
registry = create_tool_registry([SearchTool(), CalculatorTool()])

# Create registry with tools loaded from files
registry = create_tool_registry(["weather", "calculator"], "custom_tools")

# Mix pre-defined and file-based tools
registry = create_tool_registry(
    [SearchTool(), "weather"],
    "custom_tools"
)
```

## Creating Custom Tools

To create a tool that can be dynamically loaded, create a Python file with a class that inherits from `BaseTool`:

### Example: Weather Tool

File: `examples/custom_tools/weather.py`

```python
from pydantic import BaseModel, Field
from linus.agents.agent.tool_base import BaseTool


class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    city: str = Field(description="City name to get weather for")
    units: str = Field(default="celsius", description="Temperature units")


class WeatherTool(BaseTool):
    """Tool for getting weather information."""

    name: str = "weather"
    description: str = "Get current weather for a specified city"
    args_schema = WeatherInput

    async def _arun(self, city: str, units: str = "celsius") -> str:
        """Get weather for a city."""
        # Your implementation here
        return f"Weather in {city}: 22°C, sunny"
```

### Tool File Requirements

1. **Class Name**: Must inherit from `BaseTool`
2. **Name Attribute**: Set the `name` attribute (used for tool identification)
3. **Description Attribute**: Set the `description` attribute (shown to LLM)
4. **Args Schema**: Optional `args_schema` Pydantic model for input validation
5. **Implementation**: Implement the `_arun()` async method

## Usage Examples

### Example 1: Load All Tools from Directory

```python
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import load_tools_from_directory

# Load all custom tools
tools = load_tools_from_directory("custom_tools")

# Create agent with loaded tools
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=tools,
    use_async=True
)

# Run the agent
response = await agent.run("What's the weather in London?")
```

### Example 2: Mix Pre-defined and Dynamic Tools

```python
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools, load_tools_from_directory

# Get default tools (search, calculator, etc.)
default_tools = get_default_tools()

# Load custom tools
custom_tools = load_tools_from_directory("custom_tools")

# Combine them
all_tools = default_tools + custom_tools

# Create agent
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=all_tools,
    use_async=True
)
```

### Example 3: Selective Tool Loading

```python
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import create_tool_registry, SearchTool

# Load only specific tools
registry = create_tool_registry(
    tools=[
        SearchTool(),      # Pre-defined tool
        "weather",         # Load from custom_tools/weather.py
        "translator"       # Load from custom_tools/translator.py
    ],
    tools_directory="custom_tools"
)

# Get tools from registry
tools = list(registry.tool_map.values())

# Create agent
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=tools,
    use_async=True
)
```

### Example 4: Using ToolRegistry Directly

```python
from linus.agents.agent.tools import ToolRegistry, ToolLoader

# Create registry and loader
registry = ToolRegistry()
loader = ToolLoader("custom_tools")

# Discover and load tools
available_tools = loader.discover_tools()
print(f"Found tools: {available_tools}")

for tool_name in available_tools:
    tool = loader.load_tool(tool_name)
    registry.register_tool(tool)

# Use the registry
print(f"Registered {len(registry.tool_map)} tools")
schemas = registry.get_tools()

# Get OpenAI-format schemas for API calls
tools_for_api = registry.get_tools()
```

## Directory Structure

Organize your tools in a dedicated directory:

```
my_project/
├── src/
│   └── linus/
│       └── agents/
│           └── agent/
│               ├── tool_base.py
│               └── tools.py
├── examples/
│   ├── custom_tools/
│   │   ├── weather.py
│   │   ├── translator.py
│   │   ├── database.py
│   │   └── email.py
│   └── example_dynamic_tool_usage.py
├── tests/
│   └── test_dynamic_tools.py
└── main.py
```

## Best Practices

1. **One Tool Per File**: Keep each tool in its own file for better organization

2. **Clear Naming**: Use descriptive names for both files and tool classes
   - File: `weather.py`
   - Class: `WeatherTool`

3. **Input Validation**: Always define an `args_schema` Pydantic model for type safety

4. **Descriptive Documentation**: Write clear descriptions that help the LLM understand when to use the tool

5. **Error Handling**: Implement proper error handling in your `_arun()` method

6. **Async Implementation**: Always implement `_arun()` as an async method, even if the underlying operation is synchronous

## Testing Tools

Test your dynamically loaded tools before using them with the agent:

```python
import asyncio
from linus.agents.agent.tools import ToolLoader

async def test_tool():
    loader = ToolLoader("custom_tools")
    weather_tool = loader.load_tool("weather")

    # Test the tool
    result = await weather_tool.arun({
        "city": "London",
        "units": "celsius"
    })
    print(f"Result: {result}")

asyncio.run(test_tool())
```

## OpenAI Schema Generation

The `ToolRegistry` automatically generates OpenAI-compatible schemas from your tools:

```python
from linus.agents.agent.tools import create_tool_registry

registry = create_tool_registry(["weather"], "custom_tools")

# Get JSON schema
schema = registry.get_tools_str()
print(schema)
```

Output:
```json
[
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get current weather for a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "description": "City name to get weather for",
                        "type": "string"
                    },
                    "units": {
                        "default": "celsius",
                        "description": "Temperature units",
                        "type": "string"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

## Troubleshooting

### Tool Not Found

```
FileNotFoundError: Tool 'weather' not found at custom_tools/weather.py
```

**Solution**: Ensure the file exists and the path is correct

### No BaseTool Subclass Found

```
AttributeError: No BaseTool subclass found in custom_tools/weather.py
```

**Solution**: Ensure your tool class inherits from `BaseTool`:
```python
class WeatherTool(BaseTool):  # ✓ Correct
    ...
```

### Import Errors

If you get import errors when loading tools, ensure your PYTHONPATH includes the src directory:

```bash
export PYTHONPATH=/path/to/project/src:$PYTHONPATH
```

Or in Python:
```python
import sys
sys.path.insert(0, '/path/to/project/src')
```

## Performance Considerations

1. **Caching**: Tools are cached after first load, so repeated loads are fast
2. **Lazy Loading**: Tools are only loaded when requested, not all at once
3. **Memory**: Each tool instance is kept in memory once loaded

## See Also

- [CLAUDE.md](../CLAUDE.md) - Project documentation
- [tool_base.py](../src/linus/agents/agent/tool_base.py) - Base tool classes
- [tools.py](../src/linus/agents/agent/tools.py) - Tool implementations and helpers
- [test_dynamic_tools.py](../tests/test_dynamic_tools.py) - Comprehensive test suite
- [example_dynamic_tool_usage.py](../examples/example_dynamic_tool_usage.py) - Quick start examples
