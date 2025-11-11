# Examples

This directory contains example code demonstrating how to use the ReasoningAgent framework.

## Contents

### Tree of Thought Agent

- **[example_tot_usage.py](example_tot_usage.py)** - Comprehensive examples of TreeOfThoughtAgent usage
  - Basic ToT agent with reflection and tool filtering
  - Using separate reasoning models for planning vs execution
  - Configuring ToT features for different use cases
  - Performance comparisons between configurations

### Dynamic Tool Loading

- **[example_dynamic_tool_usage.py](example_dynamic_tool_usage.py)** - Demonstrates how to use dynamic tool loading with the agent
- **[custom_tools/](custom_tools/)** - Example custom tools that can be dynamically loaded
  - [weather.py](custom_tools/weather.py) - Weather lookup tool
  - [translator.py](custom_tools/translator.py) - Text translation tool

## Running Examples

### Prerequisites

Ensure you have the project dependencies installed and Ollama running locally.

### Tree of Thought Agent Example

```bash
# From the project root
PYTHONPATH=src python examples/example_tot_usage.py
```

This example demonstrates:
1. Basic ToT agent with reflection and tool filtering
2. Using separate reasoning models (e.g., DeepSeek-R1 for reasoning, Gemma3 for execution)
3. Configuring ToT features (reflection, tool filtering)
4. Performance comparisons between different configurations

See [docs/TREE_OF_THOUGHT.md](../docs/TREE_OF_THOUGHT.md) for detailed documentation.

### Dynamic Tool Loading Example

```bash
# From the project root
PYTHONPATH=src python examples/example_dynamic_tool_usage.py
```

This example shows three different approaches:
1. Loading all tools from a directory
2. Mixing default and custom tools
3. Using ToolRegistry for advanced control

## Creating Your Own Tools

To create custom tools for dynamic loading:

1. Create a new Python file in `custom_tools/` (e.g., `my_tool.py`)
2. Define a class that inherits from `BaseTool`
3. Implement the required attributes and methods
4. Load it using the `ToolLoader` or helper functions

See the [Dynamic Tool Loading Documentation](../docs/DYNAMIC_TOOL_LOADING.md) for detailed instructions.

## Example Structure

```
examples/
├── README.md                          # This file
├── example_tot_usage.py               # Tree of Thought agent examples
├── example_dynamic_tool_usage.py      # Dynamic tool loading examples
└── custom_tools/                      # Example custom tools
    ├── weather.py                     # Weather tool
    └── translator.py                  # Translation tool
```

## More Examples

For more examples and detailed documentation, see:
- [CLAUDE.md](../CLAUDE.md) - Project overview and usage
- [docs/TREE_OF_THOUGHT.md](../docs/TREE_OF_THOUGHT.md) - Tree of Thought agent guide
- [docs/DYNAMIC_TOOL_LOADING.md](../docs/DYNAMIC_TOOL_LOADING.md) - Tool loading guide
- [tests/](../tests/) - Test files showing various usage patterns
