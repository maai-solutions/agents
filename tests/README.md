# Tests

This directory contains test files for the ReasoningAgent framework.

## Contents

### Dynamic Tool Loading Tests

- **[test_dynamic_tools.py](test_dynamic_tools.py)** - Comprehensive test suite for dynamic tool loading functionality

## Running Tests

### Run All Dynamic Tool Loading Tests

```bash
# From the project root
PYTHONPATH=src python tests/test_dynamic_tools.py
```

This will run all tests including:
1. ToolLoader - Loading specific tools from files
2. ToolRegistry - Tool registration and management
3. load_tools_from_directory - Batch loading tools
4. create_tool_registry - Creating registries with mixed tools
5. OpenAI Schema Generation - Validating schema output

### Expected Output

All tests should pass with output showing:
- Tools discovered and loaded successfully
- Tool schemas generated correctly
- Tools executing properly
- OpenAI-compatible schemas validated

## Test Coverage

The test suite covers:
- ✓ Dynamic tool loading from Python files
- ✓ Tool discovery in directories
- ✓ Tool registration and lookup
- ✓ OpenAI schema generation
- ✓ Mixing pre-defined and dynamically loaded tools
- ✓ Tool caching and performance
- ✓ Tool execution with various input types

## Writing New Tests

To add new tests:

1. Create a new test file or add to existing ones
2. Follow the naming convention: `test_*.py`
3. Use async test functions for agent-related tests
4. Include clear test documentation and assertions

Example:
```python
async def test_my_feature():
    """Test description."""
    # Setup
    loader = ToolLoader("examples/custom_tools")

    # Execute
    tool = loader.load_tool("weather")
    result = await tool.arun({"city": "London"})

    # Assert
    assert "London" in result
    print(f"✓ Test passed: {result}")
```

## Test Structure

```
tests/
├── README.md                 # This file
└── test_dynamic_tools.py     # Dynamic tool loading tests
```

## Continuous Integration

These tests should be run:
- Before committing changes
- In CI/CD pipelines
- After adding new features
- When modifying tool-related code

## Troubleshooting

If tests fail:

1. **Import Errors**: Ensure `PYTHONPATH` includes the `src` directory
   ```bash
   export PYTHONPATH=src:$PYTHONPATH
   ```

2. **File Not Found**: Check that `examples/custom_tools/` contains the example tools

3. **Tool Loading Errors**: Verify that tool files have proper `BaseTool` subclasses

4. **Async Errors**: Make sure to use `await` with async tool methods

## See Also

- [examples/](../examples/) - Example code and usage
- [docs/DYNAMIC_TOOL_LOADING.md](../docs/DYNAMIC_TOOL_LOADING.md) - Detailed documentation
- [CLAUDE.md](../CLAUDE.md) - Project overview
