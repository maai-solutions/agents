# Async/Await Support

## Overview

The agent system now supports **async/await** for non-blocking execution and **parallel processing** of DAG nodes, providing significant performance improvements for I/O-bound operations like LLM calls and tool executions.

## Benefits

### 1. Non-Blocking Execution
- Agent operations don't block the event loop
- Perfect for web servers (FastAPI, etc.)
- Better resource utilization

### 2. Parallel Processing
- DAG nodes at the same level can run in parallel
- Reduces total execution time significantly
- Optimal for independent tasks

### 3. Performance
- **Typical speedup**: 2-3x for DAG workflows with parallel nodes
- **LLM calls**: Non-blocking, can run concurrently
- **Tool execution**: Runs in thread pool when tools are synchronous

## Agent Async API

### Using `arun()` Method

The `ReasoningAgent` class now has an async `arun()` method alongside the sync `run()` method.

#### Basic Usage

```python
import asyncio
from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools

async def main():
    tools = get_default_tools()
    agent = create_gemma_agent(tools=tools)

    # Use async arun instead of sync run
    result = await agent.arun("Calculate 42 * 17", return_metrics=True)

    print(f"Result: {result.result}")
    print(f"Metrics: {result.metrics.to_dict()}")

# Run async code
asyncio.run(main())
```

#### Parallel Agent Execution

```python
async def parallel_agents():
    tools = get_default_tools()

    # Create multiple agents
    agent1 = create_gemma_agent(tools=tools)
    agent2 = create_gemma_agent(tools=tools)
    agent3 = create_gemma_agent(tools=tools)

    # Execute all agents in parallel
    results = await asyncio.gather(
        agent1.arun("Task 1", return_metrics=False),
        agent2.arun("Task 2", return_metrics=False),
        agent3.arun("Task 3", return_metrics=False)
    )

    print(f"All results: {results}")
```

### API Comparison

| Feature | `run()` (Sync) | `arun()` (Async) |
|---------|----------------|------------------|
| Execution | Blocking | Non-blocking |
| Concurrency | Sequential | Can run in parallel |
| Use case | Simple scripts | Web servers, parallel tasks |
| Performance | Baseline | Up to 3x faster with parallelism |
| Integration | Synchronous code | Async/await code |

## DAG Async API

### Using `aexecute()` Method

The `DAGExecutor` class now has an async `aexecute()` method with built-in parallel execution support.

#### Sequential Execution

```python
async def sequential_dag():
    from linus.agents.graph import AgentDAG, AgentNode, DAGExecutor

    # Create DAG
    dag = AgentDAG("Pipeline")
    dag.add_node(AgentNode(name="step1", agent=agent1, output_key="result1"))
    dag.add_node(AgentNode(name="step2", agent=agent2, output_key="result2"))
    dag.add_edge("step1", "step2")

    # Execute asynchronously but sequentially
    executor = DAGExecutor(dag)
    result = await executor.aexecute(
        initial_state={"input": "data"},
        parallel=False  # Sequential execution
    )

    print(f"Status: {result.status}")
```

#### Parallel Execution (Default)

```python
async def parallel_dag():
    # Create DAG with parallel branches
    #          ┌→ stats ┐
    # ingest ──┼→ patterns ┼→ report
    #          └→ analysis ┘

    dag = AgentDAG("ParallelPipeline")
    dag.add_node(AgentNode(name="ingest", agent=ingest_agent))
    dag.add_node(AgentNode(name="stats", agent=stats_agent))
    dag.add_node(AgentNode(name="patterns", agent=pattern_agent))
    dag.add_node(AgentNode(name="analysis", agent=analysis_agent))
    dag.add_node(AgentNode(name="report", agent=report_agent))

    dag.add_edge("ingest", "stats")
    dag.add_edge("ingest", "patterns")
    dag.add_edge("ingest", "analysis")
    dag.add_edge("stats", "report")
    dag.add_edge("patterns", "report")
    dag.add_edge("analysis", "report")

    # Execute with parallel processing (default)
    executor = DAGExecutor(dag)
    result = await executor.aexecute(
        initial_state={"data": "input"},
        parallel=True  # Parallel execution (default)
    )

    # Execution order:
    # Level 1: [ingest] - runs alone
    # Level 2: [stats, patterns, analysis] - run in PARALLEL
    # Level 3: [report] - runs after all level 2 complete

    print(f"Completed in: {result.execution_time_seconds:.2f}s")
```

### DAG Parallel Execution

The DAG executor automatically identifies which nodes can run in parallel based on the topological sort:

```
Execution Levels:
  Level 1: [node1]              # Runs first
  Level 2: [node2, node3, node4] # Run in PARALLEL
  Level 3: [node5]              # Runs after level 2 completes
```

**Key Points**:
- Nodes in the same level have no dependencies on each other
- All nodes in a level run **concurrently** when `parallel=True`
- Each level waits for the previous level to complete
- Automatic fallback to sync execution if agent doesn't support async

## Performance Examples

### Example 1: Single Agent

```python
import time
import asyncio

async def compare_agent_performance():
    agent = create_gemma_agent(tools=tools)
    task = "Complex calculation task"

    # Sync version
    start = time.time()
    result_sync = agent.run(task, return_metrics=False)
    time_sync = time.time() - start

    # Async version (single task, similar performance)
    start = time.time()
    result_async = await agent.arun(task, return_metrics=False)
    time_async = time.time() - start

    print(f"Sync: {time_sync:.2f}s")
    print(f"Async: {time_async:.2f}s")
    # Similar times for single task
```

### Example 2: Multiple Agents

```python
async def compare_multiple_agents():
    agents = [create_gemma_agent(tools=tools) for _ in range(3)]
    tasks = ["Task 1", "Task 2", "Task 3"]

    # Sync version (sequential)
    start = time.time()
    results_sync = [agent.run(task, return_metrics=False)
                    for agent, task in zip(agents, tasks)]
    time_sync = time.time() - start

    # Async version (parallel)
    start = time.time()
    results_async = await asyncio.gather(*[
        agent.arun(task, return_metrics=False)
        for agent, task in zip(agents, tasks)
    ])
    time_async = time.time() - start

    print(f"Sync (sequential): {time_sync:.2f}s")
    print(f"Async (parallel): {time_async:.2f}s")
    print(f"Speedup: {time_sync/time_async:.2f}x")
    # Typical speedup: 2-3x
```

### Example 3: DAG with Parallel Nodes

```python
async def compare_dag_execution():
    # DAG with 3 parallel nodes in level 2
    dag = create_parallel_dag()  # 1 -> [2,3,4] -> 5
    executor = DAGExecutor(dag)

    # Sequential execution
    start = time.time()
    result_seq = await executor.aexecute(parallel=False)
    time_seq = time.time() - start

    # Parallel execution
    start = time.time()
    result_par = await executor.aexecute(parallel=True)
    time_par = time.time() - start

    print(f"Sequential: {time_seq:.2f}s")
    print(f"Parallel: {time_par:.2f}s")
    print(f"Speedup: {time_seq/time_par:.2f}x")
    # Typical speedup: 2-3x for 3 parallel nodes
```

## Integration with FastAPI

Perfect for web applications with FastAPI:

```python
from fastapi import FastAPI
from linus.agents.agent.agent import create_gemma_agent

app = FastAPI()
agent = create_gemma_agent(tools=tools)

@app.post("/agent/query")
async def query_agent(query: str):
    """Non-blocking agent endpoint"""
    # Use async arun - doesn't block the event loop
    result = await agent.arun(query, return_metrics=True)

    return {
        "result": result.result,
        "metrics": result.metrics.to_dict()
    }

@app.post("/dag/execute")
async def execute_dag(data: dict):
    """Non-blocking DAG execution"""
    executor = DAGExecutor(dag)

    # Parallel execution in background
    result = await executor.aexecute(
        initial_state=data,
        parallel=True
    )

    return {
        "status": result.status,
        "results": result.node_results
    }
```

## Implementation Details

### Agent Async Methods

The following methods have async versions:

- `arun()` - Main entry point (async version of `run()`)
- `_areasoning_call()` - Async reasoning phase
- `_aexecute_task_with_tool()` - Async tool execution
- `_agenerate_tool_arguments()` - Async arg generation
- `_agenerate_response()` - Async response generation
- `_acheck_completion()` - Async completion check

All async methods use `await self.llm.ainvoke()` instead of `self.llm.invoke()`.

### DAG Async Methods

- `aexecute()` - Main execution method
- `_execute_level_parallel()` - Parallel execution of a DAG level
- `_execute_node_async()` - Execute single node asynchronously
- `_execute_node_with_arun()` - Execute node using agent's `arun()`

### Backward Compatibility

- ✅ Sync methods (`run()`, `execute()`) still work identically
- ✅ No breaking changes to existing code
- ✅ Async methods fallback to sync when needed
- ✅ Tools run in thread pool if they're synchronous

### Tool Execution

Since most tools are synchronous, they run in a thread pool:

```python
# Inside _aexecute_task_with_tool
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, tool.run, tool_args)
```

This ensures synchronous tools don't block the event loop.

## Best Practices

### 1. Use Async for I/O-Bound Tasks

```python
# ✅ Good: I/O-bound (LLM calls, API requests)
result = await agent.arun(query)

# ❌ Not ideal: CPU-bound tasks (use sync or ProcessPoolExecutor)
result = compute_heavy_math()  # Better to use sync
```

### 2. Parallel DAG Execution

```python
# ✅ Good: Enable parallel for independent nodes
result = await executor.aexecute(parallel=True)

# ⚠️  Optional: Use sequential if order matters
result = await executor.aexecute(parallel=False)
```

### 3. Error Handling

```python
try:
    result = await agent.arun(query)
except Exception as e:
    logger.error(f"Agent failed: {e}")
    # Handle error
```

### 4. Concurrent Requests

```python
# ✅ Good: Use asyncio.gather for multiple concurrent requests
results = await asyncio.gather(
    agent1.arun(task1),
    agent2.arun(task2),
    agent3.arun(task3),
    return_exceptions=True  # Continue even if one fails
)
```

### 5. Mixed Sync/Async Code

```python
# Running sync code from async
result_sync = agent.run(query)  # Still works

# Running async code from sync
result_async = asyncio.run(agent.arun(query))

# In async context
result = await agent.arun(query)
```

## Migration Guide

### From Sync to Async

```python
# Before (sync)
def process_data(data):
    agent = create_gemma_agent(tools=tools)
    result = agent.run(data)
    return result

# After (async)
async def process_data(data):
    agent = create_gemma_agent(tools=tools)
    result = await agent.arun(data)
    return result
```

### From Sequential DAG to Parallel DAG

```python
# Before (sync, sequential)
executor = DAGExecutor(dag)
result = executor.execute(initial_state=data)

# After (async, parallel)
executor = DAGExecutor(dag)
result = await executor.aexecute(
    initial_state=data,
    parallel=True  # Enable parallelism
)
```

## Examples

See `examples/async_example.py` for comprehensive examples:

1. **Async agent execution** - Basic `arun()` usage
2. **Parallel agents** - Multiple agents running concurrently
3. **Async DAG sequential** - Async execution without parallelism
4. **Async DAG parallel** - Full parallel DAG execution
5. **Performance comparison** - Sync vs async vs parallel benchmarks

Run examples:

```bash
python examples/async_example.py
```

## Performance Tips

### 1. Maximize Parallelism

Design DAGs to have maximum parallelism:

```python
# ✅ Good: Wide parallelism
#     ┌→ task1 ┐
# start ┼→ task2 ┼→ end
#     └→ task3 ┘

# ⚠️  Suboptimal: Narrow pipeline
# start → task1 → task2 → task3 → end
```

### 2. Use Appropriate Batch Sizes

```python
# ✅ Good: Reasonable batch size
tasks = [agent.arun(q) for q in queries[:10]]
results = await asyncio.gather(*tasks)

# ❌ Bad: Too many concurrent tasks
tasks = [agent.arun(q) for q in queries[:1000]]  # May overwhelm system
```

### 3. Monitor Resource Usage

- Each concurrent LLM call uses memory
- Limit concurrent requests based on available resources
- Use semaphores if needed:

```python
semaphore = asyncio.Semaphore(5)  # Max 5 concurrent

async def limited_run(query):
    async with semaphore:
        return await agent.arun(query)
```

## Summary

✅ **Async agent execution** with `arun()` method
✅ **Parallel DAG execution** with `aexecute(parallel=True)`
✅ **Backward compatible** - sync methods still work
✅ **FastAPI integration** - non-blocking endpoints
✅ **2-3x performance improvement** for parallel workflows
✅ **Automatic parallelization** based on DAG topology
✅ **Thread pool execution** for synchronous tools
✅ **Production-ready** with comprehensive examples and docs

Perfect for building high-performance, scalable agent systems! 🚀
