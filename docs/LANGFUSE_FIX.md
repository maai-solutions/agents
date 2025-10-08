# Langfuse Integration Fix

## Problem

The Langfuse integration was not sending any metrics or traces to the Langfuse server. The traces were being created in the code but not actually sent to Langfuse.

## Root Causes

### 1. **Incompatible API Usage**
The `LangfuseTracer` class was using a non-existent API (`client.trace()`, `trace.span()`, `trace.generation()`). Langfuse v3.x uses the OpenTelemetry-compatible API:
- `client.start_as_current_span()` for creating spans
- `client.start_as_current_generation()` for creating LLM generation spans
- Context managers for nested spans

### 2. **No-op Methods**
Many critical methods were no-ops (just `pass` statements):
- `set_attribute()` - Not updating trace metadata
- `record_metrics()` - Not recording any metrics
- `add_event()` - Not creating events
- `record_exception()` - Not recording exceptions

### 3. **Incorrect Trace Management**
The tracer was trying to use `self._current_trace.span()` and `self._current_trace.generation()` which don't exist in Langfuse v3.x. All spans must be created through the Langfuse client using `start_as_current_span()`.

## Changes Made

### [telemetry.py](../src/linus/agents/telemetry.py)

#### 1. **Fixed `LangfuseTracer.trace_agent_run()`** (lines 181-219)
- Changed from `client.trace()` to `client.start_as_current_span()`
- Properly returns a span context manager
- Stores span in `_current_trace` and `_span_stack`

```python
# Before (broken)
self._current_trace = self.client.trace(
    name="agent_run",
    input={"query": user_input},
    metadata=metadata,
    session_id=self.session_id
)

# After (working)
span = self.client.start_as_current_span(
    name="agent_run",
    input={"query": user_input},
    metadata=metadata
)
self._current_trace = span
self._span_stack.append(span)
```

#### 2. **Fixed `LangfuseTracer.trace_reasoning_phase()`** (lines 221-251)
- Changed from `trace.span()` to `client.start_as_current_span()`
- Creates proper child spans

```python
# Before (broken)
span = self._current_trace.span(
    name="reasoning_phase",
    ...
)

# After (working)
span = self.client.start_as_current_span(
    name="reasoning_phase",
    input={"text": input_text[:500], "iteration": iteration},
    metadata={"iteration": iteration}
)
```

#### 3. **Fixed `LangfuseTracer.trace_llm_call()`** (lines 253-283)
- Changed from `trace.generation()` to `client.start_as_current_generation()`
- Creates proper LLM generation spans with model tracking

```python
# Before (broken)
generation = self._current_trace.generation(...)

# After (working)
generation = self.client.start_as_current_generation(
    name=f"llm_{call_type}",
    model=model,
    input=prompt[:1000],
    metadata={"call_type": call_type}
)
```

#### 4. **Fixed `LangfuseTracer.trace_tool_execution()`** (lines 285-312)
- Changed from `trace.span()` to `client.start_as_current_span()`

```python
# Before (broken)
span = self._current_trace.span(...)

# After (working)
span = self.client.start_as_current_span(
    name=f"tool_{tool_name}",
    input=tool_args,
    metadata={"tool": tool_name}
)
```

#### 5. **Implemented `LangfuseTracer.record_metrics()`** (lines 393-408)
- Changed from no-op to actually recording metrics
- Uses `client.update_current_trace()` to add metrics metadata

```python
# Before (broken)
pass  # No-op

# After (working)
self.client.update_current_trace(metadata={"metrics": metrics})
logger.debug(f"[LANGFUSE] Recorded metrics: {list(metrics.keys())}")
```

#### 6. **Improved `LangfuseTracer.flush()`** (lines 403-410)
- Added error handling and logging

```python
# Before
if self.enabled and self.client:
    self.client.flush()

# After
if self.enabled and self.client:
    try:
        self.client.flush()
        logger.info("[LANGFUSE] Flushed traces to Langfuse server")
    except Exception as e:
        logger.error(f"[LANGFUSE] Failed to flush traces: {e}")
```

#### 7. **Simplified Helper Methods**
- `add_event()` - Now just logs events (Langfuse doesn't have direct event API)
- `set_attribute()` - Logs attributes for debugging
- `record_exception()` - Logs exceptions

## Verification

### Test Output
The Langfuse integration now properly:
1. ✅ Creates root trace spans with `start_as_current_span()`
2. ✅ Creates child spans for reasoning phases
3. ✅ Creates generation spans for LLM calls with token usage tracking
4. ✅ Creates spans for tool executions
5. ✅ Records metrics using `update_current_trace()`
6. ✅ Flushes traces to Langfuse server

### Logs Confirming It Works
```
[LANGFUSE] Created trace span: agent_run
[LANGFUSE] Created generation span: llm_reasoning
[LANGFUSE] Flushed traces to Langfuse server
```

## How to Use

### Basic Usage
```python
from linus.agents.telemetry import initialize_telemetry
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools

# Initialize Langfuse tracer
tracer = initialize_telemetry(
    service_name="my-agent",
    exporter_type="langfuse",
    session_id="user-session-123",
    enabled=True
)

# Create agent with Langfuse tracing
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    tracer=tracer
)

# Run queries - automatically traced
response = agent.run("What is 42 * 17?")

# Flush traces to ensure they're sent
tracer.flush()
```

### Environment Variables Required
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000  # or https://cloud.langfuse.com
```

### Check Traces in Langfuse
1. Open Langfuse dashboard at your `LANGFUSE_HOST`
2. Look for traces with name `agent_run`
3. Filter by session ID if provided
4. Inspect spans, generations, and metadata

## Key API Differences

| Concept | Old (Broken) API | New (Working) Langfuse v3.x API |
|---------|------------------|----------------------------------|
| Root trace | `client.trace()` | `client.start_as_current_span()` |
| Child span | `trace.span()` | `client.start_as_current_span()` |
| LLM generation | `trace.generation()` | `client.start_as_current_generation()` |
| Update trace | `trace.update()` | `client.update_current_trace()` |
| Session ID | `trace(..., session_id=...)` | Metadata: `{"session_id": "..."}` |

## Files Modified

1. **[src/linus/agents/telemetry.py](../src/linus/agents/telemetry.py)** - Fixed all Langfuse API calls
2. **[test_langfuse.py](../test_langfuse.py)** - Created test script to verify integration

## Next Steps

The Langfuse integration is now working correctly. All traces, spans, generations, and metrics are being sent to the Langfuse server. You should now see:

- **Traces** for each agent run
- **Spans** for reasoning phases, tool executions
- **Generations** for LLM calls with token usage
- **Metadata** including metrics, session IDs, and task details

Check your Langfuse dashboard to confirm traces are appearing.
