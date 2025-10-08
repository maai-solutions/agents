# DAG (Directed Acyclic Graph) Orchestration

## Overview

The DAG module provides a powerful framework for orchestrating multiple agents in complex workflows. It allows you to:

- **Define workflows** as directed acyclic graphs
- **Execute agents** in parallel or sequential order
- **Share state** between agents
- **Handle errors** gracefully with recovery strategies
- **Add conditions** to control execution flow
- **Visualize workflows** for debugging and documentation

## Core Concepts

### 1. AgentNode

A node represents a single agent in the workflow.

```python
from linus.agents.graph import AgentNode

node = AgentNode(
    name="data_processor",
    agent=agent,
    description="Process incoming data",
    input_mapping={"data": "raw_data"},  # Map state keys to agent input
    output_key="processed_data",         # Where to store output
    retry_count=0,
    timeout_seconds=None,
    on_error="fail"  # "fail", "skip", or "continue"
)
```

### 2. AgentDAG

The DAG structure that contains nodes and edges.

```python
from linus.agents.graph import AgentDAG

dag = AgentDAG(name="MyWorkflow")

# Add nodes
dag.add_node(node1)
dag.add_node(node2)

# Add edges
dag.add_edge("node1", "node2")  # node1 -> node2
```

### 3. SharedState

State shared between all agents in the DAG.

```python
from linus.agents.graph import SharedState

state = SharedState()
state.set("key", "value", source="node_name")
value = state.get("key")
```

### 4. DAGExecutor

Executes the DAG and manages the workflow.

```python
from linus.agents.graph import DAGExecutor

executor = DAGExecutor(dag, state=shared_state)
result = executor.execute(initial_state={"input": "data"})
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      AgentDAG                            │
│                                                          │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐        │
│  │  Node A  │────▶│  Node B  │────▶│  Node C  │        │
│  └──────────┘     └──────────┘     └──────────┘        │
│       │                                   ▲              │
│       │                                   │              │
│       └───────────────────────────────────┘              │
│                                                          │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
            ┌────────────────────┐
            │   DAGExecutor      │
            │                    │
            │  - Validate DAG    │
            │  - Execute nodes   │
            │  - Manage state    │
            │  - Collect results │
            └────────────────────┘
                        │
                        ▼
            ┌────────────────────┐
            │   SharedState      │
            │                    │
            │  key1: value1      │
            │  key2: value2      │
            │  ...               │
            └────────────────────┘
```

## Usage Examples

### Example 1: Simple Linear Workflow

```python
from linus.agents.graph import AgentDAG, AgentNode, DAGExecutor
from linus.agents.agent import Agent
from linus.agents.agent import get_default_tools

tools = get_default_tools()

# Create agents
agent1 = Agent(tools=tools)
agent2 = Agent(tools=tools)
agent3 = Agent(tools=tools)

# Build DAG
dag = AgentDAG(name="LinearFlow")

dag.add_node(AgentNode(
    name="step1",
    agent=agent1,
    description="First step",
    output_key="step1_result"
))

dag.add_node(AgentNode(
    name="step2",
    agent=agent2,
    description="Second step",
    input_mapping={"data": "step1_result"},
    output_key="step2_result"
))

dag.add_node(AgentNode(
    name="step3",
    agent=agent3,
    description="Final step",
    input_mapping={"data": "step2_result"},
    output_key="final_result"
))

# Connect nodes
dag.add_edge("step1", "step2")
dag.add_edge("step2", "step3")

# Execute
executor = DAGExecutor(dag)
result = executor.execute(initial_state={"input": "Start here"})

print(f"Status: {result.status}")
print(f"Final result: {result.final_state['final_result']}")
```

### Example 2: Parallel Execution

```python
# Build DAG with parallel branches
dag = AgentDAG(name="ParallelFlow")

# Start node
dag.add_node(AgentNode(
    name="start",
    agent=start_agent,
    output_key="initial_data"
))

# Parallel branches
dag.add_node(AgentNode(
    name="branch_a",
    agent=agent_a,
    input_mapping={"data": "initial_data"},
    output_key="result_a"
))

dag.add_node(AgentNode(
    name="branch_b",
    agent=agent_b,
    input_mapping={"data": "initial_data"},
    output_key="result_b"
))

dag.add_node(AgentNode(
    name="branch_c",
    agent=agent_c,
    input_mapping={"data": "initial_data"},
    output_key="result_c"
))

# Merge node (waits for all branches)
dag.add_node(AgentNode(
    name="merge",
    agent=merge_agent,
    output_key="merged_result"
))

# Edges: start -> branches -> merge
dag.add_edge("start", "branch_a")
dag.add_edge("start", "branch_b")
dag.add_edge("start", "branch_c")
dag.add_edge("branch_a", "merge")
dag.add_edge("branch_b", "merge")
dag.add_edge("branch_c", "merge")

# Execute (branches run in parallel levels)
executor = DAGExecutor(dag)
result = executor.execute()
```

### Example 3: Conditional Workflow

```python
from linus.agents.graph import SharedState

# Define condition functions
def check_high_value(state: SharedState) -> bool:
    value = state.get("value", 0)
    return value > 100

def check_low_value(state: SharedState) -> bool:
    return not check_high_value(state)

# Build DAG
dag = AgentDAG(name="ConditionalFlow")

dag.add_node(AgentNode(name="analyzer", agent=analyzer_agent, output_key="value"))
dag.add_node(AgentNode(name="high_path", agent=high_agent, output_key="result"))
dag.add_node(AgentNode(name="low_path", agent=low_agent, output_key="result"))
dag.add_node(AgentNode(name="finalizer", agent=final_agent, output_key="final"))

# Conditional edges
dag.add_edge("analyzer", "high_path", condition=check_high_value)
dag.add_edge("analyzer", "low_path", condition=check_low_value)
dag.add_edge("high_path", "finalizer")
dag.add_edge("low_path", "finalizer")

# Execute
executor = DAGExecutor(dag)
result = executor.execute(initial_state={"input_value": 150})
```

### Example 4: Error Handling

```python
# Configure error handling per node
dag = AgentDAG(name="ErrorHandling")

# Critical node - fail entire workflow on error
dag.add_node(AgentNode(
    name="critical_task",
    agent=critical_agent,
    on_error="fail",  # Default behavior
    output_key="critical_result"
))

# Optional node - skip on error and continue
dag.add_node(AgentNode(
    name="optional_task",
    agent=optional_agent,
    on_error="skip",
    output_key="optional_result"
))

# Resilient node - continue despite errors
dag.add_node(AgentNode(
    name="resilient_task",
    agent=resilient_agent,
    on_error="continue",
    output_key="resilient_result"
))

dag.add_edge("critical_task", "optional_task")
dag.add_edge("optional_task", "resilient_task")

executor = DAGExecutor(dag)
result = executor.execute()

# Check what happened
print(f"Completed: {result.completed_nodes}")
print(f"Failed: {result.failed_nodes}")
print(f"Skipped: {result.skipped_nodes}")
print(f"Errors: {result.errors}")
```

## API Reference

### AgentNode

```python
AgentNode(
    name: str,                              # Unique node name
    agent: Any,                            # Agent instance
    description: str = "",                 # Description
    input_mapping: Optional[Dict[str, str]] = None,  # State key mapping
    output_key: Optional[str] = None,      # Output state key
    retry_count: int = 0,                  # Retry attempts
    timeout_seconds: Optional[float] = None,  # Timeout
    on_error: str = "fail"                 # Error handling
)
```

**Input Mapping:**
- Maps shared state keys to agent input
- Example: `{"agent_param": "state_key"}`
- If None, passes all state to agent

**Error Handling:**
- `"fail"`: Fail workflow on error (default)
- `"skip"`: Skip node and continue
- `"continue"`: Mark as completed despite error

### AgentDAG

```python
dag = AgentDAG(name="WorkflowName")

# Add nodes
dag.add_node(node)

# Add edges
dag.add_edge(
    from_node: str,
    to_node: str,
    condition: Optional[Callable[[SharedState], bool]] = None
)

# Validation
dag.validate()  # Raises ValueError if invalid

# Visualization
print(dag.visualize())

# Inspection
start_nodes = dag.get_start_nodes()
end_nodes = dag.get_end_nodes()
deps = dag.get_dependencies("node_name")
execution_order = dag.get_execution_order()

# Reset
dag.reset()  # Reset all nodes to pending
```

### SharedState

```python
state = SharedState()

# Set values
state.set(
    key: str,
    value: Any,
    source: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
)

# Get values
value = state.get(key: str, default: Any = None)
exists = state.has(key: str)
entry = state.get_entry(key: str)  # Get StateEntry with metadata

# Get all
all_values = state.get_all()  # Dict[str, Any]

# History
history = state.get_history(key: Optional[str] = None)

# Management
state.delete(key: str)
state.clear()

# Export/import
data = state.to_dict()
```

### DAGExecutor

```python
executor = DAGExecutor(
    dag: AgentDAG,
    state: Optional[SharedState] = None
)

# Execute
result = executor.execute(
    initial_state: Optional[Dict[str, Any]] = None,
    parallel: bool = False  # Not implemented yet
)
```

### ExecutionResult

```python
result = executor.execute()

# Result attributes
result.dag_name: str
result.status: str  # "success", "partial", "failed"
result.total_nodes: int
result.completed_nodes: int
result.failed_nodes: int
result.skipped_nodes: int
result.execution_time_seconds: float
result.node_results: Dict[str, Any]
result.errors: Dict[str, str]
result.final_state: Dict[str, Any]

# Convert to dict
data = result.to_dict()
```

## Advanced Patterns

### Pattern 1: Map-Reduce

```python
# Map phase: Process items in parallel
dag = AgentDAG(name="MapReduce")

dag.add_node(AgentNode(name="splitter", agent=splitter, output_key="items"))

# Create mapper nodes dynamically
for i in range(num_mappers):
    dag.add_node(AgentNode(
        name=f"map_{i}",
        agent=mapper_agent,
        output_key=f"mapped_{i}"
    ))
    dag.add_edge("splitter", f"map_{i}")

# Reduce phase
dag.add_node(AgentNode(name="reducer", agent=reducer, output_key="result"))
for i in range(num_mappers):
    dag.add_edge(f"map_{i}", "reducer")
```

### Pattern 2: Pipeline with Validation

```python
# Each step validates before proceeding
dag = AgentDAG(name="ValidatedPipeline")

steps = ["ingest", "clean", "transform", "validate", "export"]
for i, step in enumerate(steps):
    dag.add_node(AgentNode(
        name=step,
        agent=agents[step],
        output_key=f"{step}_result"
    ))

    if i > 0:
        # Add validation condition
        def validate_prev_step(state, prev=steps[i-1]):
            result = state.get(f"{prev}_result")
            return result is not None and "error" not in str(result).lower()

        dag.add_edge(steps[i-1], step, condition=validate_prev_step)
```

### Pattern 3: Retry Logic

```python
# Implement retry with multiple attempts
dag = AgentDAG(name="RetryPattern")

# Try primary approach
dag.add_node(AgentNode(
    name="primary",
    agent=primary_agent,
    on_error="skip",
    output_key="result"
))

# Fallback approach if primary fails
dag.add_node(AgentNode(
    name="fallback",
    agent=fallback_agent,
    output_key="result"
))

# Check if primary succeeded
def primary_failed(state):
    return not state.has("result") or state.get("result") is None

dag.add_edge("primary", "fallback", condition=primary_failed)
```

## Best Practices

### 1. Node Naming

```python
# ✅ Good: Descriptive names
dag.add_node(AgentNode(name="validate_customer_data", ...))
dag.add_node(AgentNode(name="calculate_risk_score", ...))

# ❌ Bad: Generic names
dag.add_node(AgentNode(name="node1", ...))
dag.add_node(AgentNode(name="process", ...))
```

### 2. Input/Output Mapping

```python
# ✅ Good: Explicit mapping
dag.add_node(AgentNode(
    name="analyzer",
    agent=agent,
    input_mapping={
        "customer_data": "validated_data",
        "config": "analysis_config"
    },
    output_key="analysis_results"
))

# ❌ Bad: No mapping (passes all state)
dag.add_node(AgentNode(name="analyzer", agent=agent))
```

### 3. Error Handling

```python
# ✅ Good: Appropriate error handling
dag.add_node(AgentNode(
    name="critical_validation",
    agent=validator,
    on_error="fail"  # Must succeed
))

dag.add_node(AgentNode(
    name="optional_enrichment",
    agent=enricher,
    on_error="skip"  # Can fail gracefully
))

# ❌ Bad: Same handling for all
# All nodes with on_error="fail" - too strict
# All nodes with on_error="continue" - masks errors
```

### 4. State Management

```python
# ✅ Good: Clear state keys
state.set("customer_risk_score", 0.75, source="risk_analyzer")
state.set("validation_passed", True, source="validator")

# ❌ Bad: Unclear keys
state.set("result", some_value)
state.set("data", other_value)
```

### 5. Validation

```python
# ✅ Good: Validate before execution
dag.validate()  # Catches cycles, missing nodes, etc.
print(dag.visualize())  # Review structure

executor = DAGExecutor(dag)
result = executor.execute()

# ❌ Bad: Skip validation
executor = DAGExecutor(dag)
result = executor.execute()  # Might fail unexpectedly
```

## Debugging

### Visualize DAG

```python
dag = AgentDAG(name="MyWorkflow")
# ... add nodes and edges ...

# Print structure
print(dag.visualize())

# Output:
# DAG: MyWorkflow
# ==================================================
#
# Nodes:
#   ⏸ node1: Description of node1
#   ⏸ node2: Description of node2
#
# Edges:
#   node1 -> node2
#
# Execution Order:
#   Level 1: node1
#   Level 2: node2
```

### Inspect State

```python
# During execution, check state
def debug_condition(state: SharedState) -> bool:
    print(f"Current state: {state.get_all()}")
    print(f"History: {state.get_history()}")
    return True

dag.add_edge("node1", "node2", condition=debug_condition)
```

### Check Execution Results

```python
result = executor.execute()

# Detailed inspection
print(f"Status: {result.status}")
print(f"Completed: {result.completed_nodes}/{result.total_nodes}")

# Check each node
for node_name, node in dag.nodes.items():
    print(f"\n{node_name}:")
    print(f"  Status: {node.status}")
    print(f"  Result: {node.result}")
    print(f"  Error: {node.error}")
    print(f"  Execution time: {node.end_time - node.start_time if node.end_time else 'N/A'}")
```

## Performance Considerations

### Parallelization

Currently, nodes at the same level in the execution order can conceptually run in parallel, but execution is sequential. Future versions will support true parallel execution.

```python
# Execution order shows potential parallelism
order = dag.get_execution_order()
# [[node1], [node2, node3, node4], [node5]]
# Level 1: node1 (must run first)
# Level 2: node2, node3, node4 (can run in parallel)
# Level 3: node5 (waits for level 2)
```

### Memory Management

```python
# For large workflows, clear state periodically
dag.add_node(AgentNode(
    name="cleanup",
    agent=cleanup_agent,
    description="Clean up intermediate results"
))

# In cleanup agent
def cleanup(state: SharedState):
    # Remove large intermediate results
    for key in list(state.get_all().keys()):
        if "intermediate_" in key:
            state.delete(key)
```

## Limitations

1. **No True Parallelism**: Nodes execute sequentially (parallel execution planned)
2. **No Dynamic DAG**: DAG structure is static once created
3. **No Cycle Detection in Conditions**: Conditional edges could create runtime cycles
4. **No Timeout Enforcement**: Timeout parameter exists but not enforced yet

## Future Enhancements

- [ ] True parallel execution for independent nodes
- [ ] Dynamic DAG modification during execution
- [ ] Distributed execution across multiple machines
- [ ] DAG persistence and resumption
- [ ] Visual DAG editor
- [ ] Real-time execution monitoring
- [ ] Sub-DAG support (nested workflows)
- [ ] Event-driven triggers

## Examples

See:
- `examples/dag_example.py` - Complete examples
- `tests/test_dag.py` - Test cases and patterns
