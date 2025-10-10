# Telemetry Improvements: Tool Result Feedback Loop

## Problem

Previously, the telemetry did not clearly show how tool execution results flow back into the agent's reasoning process. The trace showed:

1. Tool execution with result
2. Next reasoning call

But it was unclear:
- How the tool result was incorporated into the agent's context
- What context was being passed to the next reasoning iteration
- Whether the agent was using previous results to inform next steps

## Solution

Added explicit trace events at key points in the feedback loop to show the complete flow:

### New Trace Events

#### 1. `tool_execution_completed`
**When**: Immediately after a tool successfully executes
**Location**: `_execute_task_with_tool()`, `_aexecute_task_with_tool()`
**Attributes**:
- `tool_name`: Name of the executed tool
- `result_length`: Length of the result string
- `status`: "success" or "error"

**Purpose**: Marks the completion of tool execution with metadata about the result.

```python
self.tracer.add_event("tool_execution_completed", {
    "tool_name": task.tool_name,
    "result_length": len(str(result)),
    "status": "success"
})
```

---

#### 2. `context_updated_with_tool_result`
**When**: After tool result is appended to the agent's context
**Location**: Main run loop after task execution
**Attributes**:
- `tool_name`: Tool that produced the result
- `result_preview`: First 200 chars of the result
- `context_length`: Total length of updated context

**Purpose**: Shows exactly what information is being fed back into the agent's reasoning.

```python
self.tracer.add_event("context_updated_with_tool_result", {
    "tool_name": task.tool_name,
    "result_preview": str(task_result)[:200],
    "context_length": len(context)
})
```

---

#### 3. `context_updated_with_llm_response`
**When**: After direct LLM response (no tool) is added to context
**Location**: Main run loop for non-tool tasks
**Attributes**:
- `response_preview`: First 200 chars of response
- `context_length`: Total context length

**Purpose**: Tracks context updates from LLM-only responses.

```python
self.tracer.add_event("context_updated_with_llm_response", {
    "response_preview": response[:200],
    "context_length": len(context)
})
```

---

#### 4. `reasoning_input_with_history`
**When**: Before reasoning call when execution history exists
**Location**: Beginning of each iteration (when history is available)
**Attributes**:
- `iteration`: Current iteration number
- `has_execution_history`: Always `true` for this event
- `history_items_count`: Number of previous executions
- `context_preview`: Last 500 chars of context
- `context_length`: Total context length

**Purpose**: Shows the accumulated context being passed into the next reasoning phase, including previous tool results.

```python
self.tracer.add_event("reasoning_input_with_history", {
    "iteration": iteration,
    "has_execution_history": True,
    "history_items_count": len(execution_history),
    "context_preview": context[-500:] if len(context) > 500 else context,
    "context_length": len(context)
})
```

---

#### 5. `task_completion_checked`
**When**: After completion check is performed
**Location**: After `_check_completion()` call
**Attributes**:
- `is_complete`: Boolean completion status
- `iteration`: Current iteration
- `reasoning`: First 200 chars of completion reasoning

**Purpose**: Shows the agent's assessment of task completion.

```python
self.tracer.add_event("task_completion_checked", {
    "is_complete": is_complete,
    "iteration": iteration,
    "reasoning": completion_status['reasoning'][:200]
})
```

---

#### 6. `continuing_to_next_iteration`
**When**: When task is incomplete and more iterations available
**Location**: After completion check, before loop continues
**Attributes**:
- `next_iteration`: Iteration number about to start
- `next_action`: What the agent plans to do next
- `missing_steps`: List of steps still needed
- `accumulated_context_length`: Size of context for next iteration

**Purpose**: Shows iteration continuation logic and accumulated context.

```python
self.tracer.add_event("continuing_to_next_iteration", {
    "next_iteration": iteration + 1,
    "next_action": completion_status['next_action'],
    "missing_steps": completion_status['missing_steps'],
    "accumulated_context_length": len(context)
})
```

---

## Trace Flow Example

For a multi-step task like "What is 42 * 17?", the trace now shows:

```
agent_run (span)
├── reasoning_phase (span)
│   └── llm_reasoning (generation)
│       ├── Input: "What is 42 * 17?"
│       └── Output: {tasks: [{tool: "calculator", ...}]}
│
├── tool_calculator (span)
│   ├── Input: {expression: "42 * 17"}
│   ├── Output: "714"
│   └── EVENT: tool_execution_completed
│       - tool_name: calculator
│       - result_length: 3
│       - status: success
│
├── EVENT: context_updated_with_tool_result
│   - tool_name: calculator
│   - result_preview: "714"
│   - context_length: 245
│
├── EVENT: task_completion_checked
│   - is_complete: true
│   - iteration: 1
│   - reasoning: "Task completed successfully"
│
└── Final output: "714"
```

For incomplete tasks requiring multiple iterations (replanning):

```
agent_run (span)
├── Iteration 1
│   ├── reasoning_phase (span) - Iteration 1
│   │   ├── llm_reasoning (generation)
│   │   │   ├── Input: "Search for X and calculate Y"
│   │   │   └── Output: {tasks: [{tool: "search", ...}]}
│   │   └── Output: {tasks_planned: 1}
│   │
│   ├── tool_search (span)
│   │   ├── Input: {query: "X"}
│   │   ├── Output: "No results found"
│   │   └── EVENT: tool_execution_completed
│   │
│   ├── EVENT: context_updated_with_tool_result
│   │   - tool_name: search
│   │   - result_preview: "No results found"
│   │   - context_length: 450
│   │
│   ├── EVENT: task_completion_checked
│   │   - is_complete: false
│   │   - reasoning: "Need to try alternative approach"
│   │
│   └── EVENT: continuing_to_next_iteration
│       - next_iteration: 2
│       - next_action: "Try calculator directly"
│       - accumulated_context_length: 450
│
└── Iteration 2 (REPLANNING)
    ├── EVENT: reasoning_input_with_history ← Shows feedback loop
    │   - history_items_count: 1
    │   - context_preview: "...Latest result: No results found"
    │   - context_length: 450
    │
    ├── reasoning_phase (span) - Iteration 2 ← NEW PLAN based on previous results
    │   ├── llm_reasoning (generation)
    │   │   ├── Input: [context with search failure]
    │   │   └── Output: {tasks: [{tool: "calculator", ...}]}
    │   └── Output: {tasks_planned: 1}
    │
    ├── tool_calculator (span)
    │   ├── Input: {expression: "Y"}
    │   ├── Output: "42"
    │   └── EVENT: tool_execution_completed
    │
    ├── EVENT: context_updated_with_tool_result
    │   - result_preview: "42"
    │
    ├── EVENT: task_completion_checked
    │   - is_complete: true
    │
    └── Final output: "42"
```

## Benefits

1. **Visibility**: Clear view of how tool results feed back into reasoning
2. **Debugging**: Easy to identify where context grows or breaks
3. **Performance**: Track context size growth across iterations
4. **Understanding**: See the complete agent decision loop
5. **Monitoring**: Detect when agents are stuck in loops or missing context

## Testing

Run the test script to see these events in Langfuse:

```bash
python test_trace_improvements.py
```

This will execute a simple calculation task and flush all trace events to Langfuse with session ID `test-trace-feedback-loop`.

## Implementation Notes

- Events are added using `tracer.add_event()` which works for both `AgentTracer` (OpenTelemetry) and `LangfuseTracer`
- Events include preview data (truncated to 200 chars) to avoid overwhelming traces
- Context length tracking helps identify memory growth issues
- Both sync (`run`) and async (`arun`) paths include the same events

## Future Enhancements

Potential additions:
- Event for memory context injection
- Event for shared state updates
- Metrics on context growth rate per iteration
- Warnings when context exceeds thresholds
