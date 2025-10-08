# Unified State Management with SharedState

## Overview

The agent system now uses **SharedState exclusively** for all state management across both the agent module and DAG orchestration. This provides a consistent, powerful state management system with built-in features like metadata tracking, history, and multi-agent coordination.

## What Changed

**Before**: Agents supported both dict and SharedState with a compatibility layer (StateWrapper).

**After**: Agents use SharedState only, simplifying the codebase and ensuring consistent behavior.

### Key Changes

1. **Removed dict state support** from Agent and ReasoningAgent
2. **Removed StateWrapper class** - no longer needed
3. **SharedState is now required** - automatically created if not provided
4. **Simplified API** - single state type across the entire system

## Benefits

### 1. Consistency
- Same state management everywhere (agents, DAG, etc.)
- No confusion about which state type to use
- Predictable behavior across all components

### 2. Built-in Features
- **Metadata tracking**: Every state change includes source and metadata
- **History tracking**: Full audit trail of all state modifications
- **Timestamps**: Automatic timestamping of changes
- **Thread-safe**: Safe for concurrent access

### 3. Multi-Agent Coordination
- Multiple agents naturally share the same SharedState
- No wrapper or translation layer needed
- Direct access to all SharedState features

### 4. Cleaner Code
- Removed ~70 lines of StateWrapper code
- Simpler Agent initialization
- Less cognitive load for developers

## Usage

### Basic Agent with State

```python
from linus.agents.agent import Agent
from linus.agents.graph.state import SharedState

# Create shared state
state = SharedState()
state.set("user_id", 123, source="init")

# Create agent - state is now a SharedState instance
agent = Agent(tools=tools, state=state)

# Access state using SharedState methods
value = state.get("user_id")  # 123
all_data = state.get_all()    # {'user_id': 123}
```

### Agent Without Explicit State

```python
# If no state provided, SharedState is created automatically
agent = Agent(tools=tools)

# Agent has a SharedState instance
agent.state.set("key", "value", source="agent")
value = agent.state.get("key")  # "value"
```

### Multiple Agents Sharing State

```python
from linus.agents.graph.state import SharedState

# Create one SharedState instance
shared_state = SharedState()

# All agents share the same state
agent1 = Agent(tools=tools, state=shared_state)
agent2 = Agent(tools=tools, state=shared_state)
agent3 = Agent(tools=tools, state=shared_state)

# Agent 1 sets data
shared_state.set("step1_complete", True, source="agent1")

# Agent 2 can see it
value = shared_state.get("step1_complete")  # True
```

### DAG Orchestration

```python
from linus.agents.graph import AgentDAG, AgentNode, DAGExecutor, SharedState

# Create shared state
state = SharedState()

# Create agents with shared state
analyzer = Agent(tools=tools, state=state)
processor = Agent(tools=tools, state=state)
reporter = Agent(tools=tools, state=state)

# Build DAG
dag = AgentDAG("Pipeline")
dag.add_node(AgentNode(name="analyze", agent=analyzer, output_key="analysis"))
dag.add_node(AgentNode(name="process", agent=processor, output_key="processed"))
dag.add_node(AgentNode(name="report", agent=reporter, output_key="final_report"))

dag.add_edge("analyze", "process")
dag.add_edge("process", "report")

# Execute - all agents use the same SharedState
executor = DAGExecutor(dag, state=state)
result = await executor.aexecute()

# Access final state
print(result.final_state["final_report"])
```

## SharedState API

### Setting Values

```python
# Basic set
state.set("key", "value")

# With source tracking
state.set("key", "value", source="agent1")

# With metadata
state.set("key", "value", source="agent1", metadata={"priority": "high"})
```

### Getting Values

```python
# Get single value
value = state.get("key")

# Get with default
value = state.get("missing_key", default="default_value")

# Get all values
all_data = state.get_all()  # Returns dict of all key-value pairs

# Get state entry with metadata
entry = state.get_entry("key")  # Returns StateEntry with metadata
```

### History Tracking

```python
# Get history for specific key
history = state.get_history("key")  # List of all changes to this key

# Get full history
full_history = state.get_history()  # List of all state changes

# Iterate through history
for entry in history:
    print(f"Value: {entry.value}")
    print(f"Source: {entry.source}")
    print(f"Timestamp: {entry.timestamp}")
    print(f"Metadata: {entry.metadata}")
```

### Other Operations

```python
# Check if key exists
exists = state.has(key)  # Returns bool

# Delete key
state.delete(key)

# Clear all state
state._state.clear()

# Export to JSON
json_data = state.to_json()

# Import from JSON
state.from_json(json_data)
```

## Migration from Dict State

If you have existing code using dict state:

### Before (Old Code with Dict)

```python
# This no longer works
state = {"key": "value"}
agent = Agent(tools=tools, state=state)
```

### After (Using SharedState)

```python
# Use SharedState instead
from linus.agents.graph.state import SharedState

state = SharedState()
state.set("key", "value", source="init")
agent = Agent(tools=tools, state=state)
```

### Accessing State

```python
# Before (dict-style)
value = agent.state["key"]          # ❌ No longer works
agent.state["key"] = "new_value"    # ❌ No longer works

# After (SharedState methods)
value = agent.state.get("key")                      # ✅ Works
agent.state.set("key", "new_value", source="agent") # ✅ Works
```

## Code Changes

### Agent Class

```python
# Before
def __init__(
    self,
    state: Optional[Union[Dict[str, Any], SharedState]] = None,
    ...
):
    if state is None:
        self.state = {}
    elif isinstance(state, SharedState):
        self.state = StateWrapper(state)
    else:
        self.state = state

# After
def __init__(
    self,
    state: Optional[SharedState] = None,
    ...
):
    self.state = state or SharedState()
```

### Usage in Code

```python
# Before (dict access)
if self.state:
    state_context = {k: str(v) for k, v in self.state.items()}

# After (SharedState access)
state_data = self.state.get_all()
if state_data:
    state_context = {k: str(v) for k, v in state_data.items()}
```

## Examples

### Example 1: Simple Agent

```python
from linus.agents.agent import Agent
from linus.agents.agent.tools import get_default_tools

tools = get_default_tools()
agent = Agent(tools=tools)

# Agent has SharedState automatically
agent.state.set("task", "process_data", source="user")

result = await agent.arun("Your task here")

# Access state after execution
all_state = agent.state.get_all()
print(f"Final state: {all_state}")
```

### Example 2: Multi-Agent Pipeline

```python
from linus.agents.graph.state import SharedState

# Create shared state
state = SharedState()
state.set("input_data", "Sales figures Q4 2024", source="system")

# Create agents sharing state
agent1 = Agent(tools=tools, state=state)
agent2 = Agent(tools=tools, state=state)

# Agent 1 processes
result1 = await agent1.arun("Analyze the input data")
state.set("analysis_complete", True, source="agent1")

# Agent 2 sees agent1's work
if state.get("analysis_complete"):
    result2 = await agent2.arun("Generate report from analysis")
```

### Example 3: State History

```python
state = SharedState()

# Make several changes
state.set("counter", 0, source="init")
state.set("counter", 1, source="agent1")
state.set("counter", 2, source="agent2")
state.set("counter", 3, source="agent3")

# Review history
history = state.get_history("counter")
print(f"Total changes: {len(history)}")

for i, entry in enumerate(history):
    print(f"Change {i}: value={entry.value}, source={entry.source}, time={entry.timestamp}")

# Output:
# Change 0: value=0, source=init, time=2024-10-07 21:00:00
# Change 1: value=1, source=agent1, time=2024-10-07 21:00:01
# Change 2: value=2, source=agent2, time=2024-10-07 21:00:02
# Change 3: value=3, source=agent3, time=2024-10-07 21:00:03
```

## Summary

✅ **Unified state management** - SharedState everywhere
✅ **Simplified API** - no dict/SharedState duality
✅ **Removed StateWrapper** - ~70 lines of code eliminated
✅ **Automatic SharedState creation** - `state or SharedState()`
✅ **Full feature access** - metadata, history, timestamps
✅ **Multi-agent ready** - built for coordination
✅ **Cleaner codebase** - easier to maintain and understand

The agent system now has a consistent, powerful state management foundation! 🚀
