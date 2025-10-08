# Import Path Fix

## Issues
After refactoring `agent.py` into multiple files, the application failed to start with multiple errors:

1. `NameError: name 'AsyncOpenAI' is not defined`
2. `AttributeError: type object 'StructuredTool' has no attribute 'from_function'`

## Root Causes
1. **Missing OpenAI imports in `reasoning_agent.py`**: The file used `AsyncOpenAI` and `OpenAI` types but didn't import them
2. **Incorrect import paths in `app.py`**: Still referenced the old `linus.agents.agent.agent` module that no longer exists
3. **Missing `from_function` class method**: `StructuredTool` was missing the `from_function()` factory method used by `create_custom_tool()`

## Files Fixed

### 1. `src/linus/agents/agent/reasoning_agent.py`
**Added missing import:**
```python
from openai import OpenAI, AsyncOpenAI
```

### 2. `src/app.py`
**Fixed import paths (4 locations):**

**Before:**
```python
from linus.agents.agent.agent import ReasoningAgent, Agent
from linus.agents.agent.tools import get_default_tools, create_custom_tool
from linus.agents.agent.agent import AgentResponse as AgentResponseData  # 3 occurrences
```

**After:**
```python
from linus.agents.agent import ReasoningAgent, Agent, get_default_tools, create_custom_tool
from linus.agents.agent import AgentResponse as AgentResponseData  # 3 occurrences
```

### 3. `src/linus/agents/agent/tool_base.py`
**Added `from_function` class method to `StructuredTool`:**

```python
@classmethod
def from_function(
    cls,
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None
) -> 'StructuredTool':
    """Create a StructuredTool from a function."""
    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or f"Tool {tool_name}"
    return cls(
        name=tool_name,
        description=tool_description,
        func=func,
        args_schema=args_schema
    )
```

This method is used by `create_custom_tool()` in `tools.py` to create tools from functions.

## Verification

All imports now work correctly:
```bash
PYTHONPATH=/Users/udg/Projects/ai/agents/src python -c "from linus.agents.agent import ReasoningAgent, Agent, get_default_tools, create_custom_tool; print('✅ All imports working')"
```

FastAPI app loads successfully:
```bash
python src/app.py
```

## Module Structure

The correct import structure after refactoring:

```
linus.agents.agent/
├── __init__.py          # Exports all public classes
├── models.py            # ReasoningResult, AgentMetrics, AgentResponse
├── base.py              # Agent base class
├── reasoning_agent.py   # ReasoningAgent implementation
├── factory.py           # Agent()
├── tool_base.py         # BaseTool, StructuredTool, @tool
├── tools.py             # Tool implementations
├── memory.py            # MemoryManager
└── mcp_client.py        # MCP integration (optional)
```

**Public API (all imported from `linus.agents.agent`):**
- `ReasoningAgent`, `Agent`
- `Agent` (factory function)
- `BaseTool`, `StructuredTool`, `tool`
- `get_default_tools`, `create_custom_tool`
- `ReasoningResult`, `AgentMetrics`, `AgentResponse`
- `MemoryManager`, `create_memory_manager`
- `MCPClientManager`, `MCPServerConfig`, `connect_mcp_servers` (if MCP installed)

---

**Fixed**: 2025-10-07
**Status**: ✅ Resolved - All imports working correctly
