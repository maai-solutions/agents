# Pydantic v2 Migration Summary

## Changes Made

The codebase has been updated to use **Pydantic v2 exclusively**, removing support for Pydantic v1 deprecated methods.

### 1. MCP Tool Schema Update

**File**: `src/linus/agents/agent/mcp_client.py:76-82`

**Before**:
```python
class MCPToolSchema:
    @staticmethod
    def schema():  # Pydantic v1 method (deprecated)
        return input_schema or {}
```

**After**:
```python
class MCPToolSchema:
    @staticmethod
    def model_json_schema():  # Pydantic v2 method
        """Return JSON schema compatible with Pydantic v2."""
        return input_schema or {}
```

### 2. Tool Argument Generation Update

**File**: `src/linus/agents/agent/reasoning_agent.py:739-748`

**Before**: Complex fallback logic supporting both Pydantic v1 `schema()` and v2 `model_json_schema()`

**After**: Simple, clean Pydantic v2 only:
```python
# Get tool schema using Pydantic v2 model_json_schema()
tool_schema = {}
if hasattr(tool, 'args_schema') and tool.args_schema:
    if hasattr(tool.args_schema, 'model_json_schema'):
        tool_schema = tool.args_schema.model_json_schema()
    else:
        logger.warning(f"[TOOL-ARGS] Tool {tool.name} has args_schema without model_json_schema()")
        # Fallback to input_schema_dict if available
        if hasattr(tool, 'input_schema_dict'):
            tool_schema = tool.input_schema_dict
```

## Benefits

1. **Cleaner code**: Removed complex fallback logic
2. **No deprecation warnings**: Uses only Pydantic v2 APIs
3. **Future-proof**: Ready for Pydantic v3
4. **Consistent API**: All tools use `model_json_schema()`

## Tool Schema Requirements

All custom tools must provide one of:
1. **Pydantic BaseModel** with `model_json_schema()` method (recommended)
2. **Custom schema class** with `model_json_schema()` static method
3. **Fallback**: `input_schema_dict` attribute with raw JSON schema

## Testing

```bash
# Test MCP tool schema access
PYTHONPATH=/Users/udg/Projects/Git/agents/src python -c "
from linus.agents.agent.mcp_client import MCPTool

class MockSession:
    pass

tool = MCPTool(
    name='test_tool',
    description='Test tool',
    mcp_tool_name='test',
    session=MockSession(),
    input_schema={'type': 'object', 'properties': {'arg1': {'type': 'string'}}}
)

# Should work with Pydantic v2
schema = tool.args_schema.model_json_schema()
print(f'Schema: {schema}')
"
```

## Migration Checklist

- [x] Update MCPToolSchema to use `model_json_schema()`
- [x] Simplify tool argument generation logic
- [x] Remove Pydantic v1 `schema()` method calls
- [x] Test MCP tools work correctly
- [x] Verify standard Pydantic tools still work
- [x] Document changes

## Related Issues Fixed

This migration also fixed the error:
```
AttributeError: type object 'MCPToolSchema' has no attribute 'model_json_schema'
```

The error occurred because MCP tools were using the old `schema()` method while the agent code was trying to call `model_json_schema()`.
