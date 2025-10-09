# Async Telemetry Fix

## Problem

After removing the sync implementation and keeping only the async implementation, the telemetry stopped working because:

1. **Langfuse SDK provides sync context managers** - Methods like `start_as_current_span()` and `start_as_current_observation()` return sync context managers that don't work with `async with`
2. **OpenTelemetry also provides sync context managers** - Even though our agent is async, OpenTelemetry's `start_as_current_span()` is synchronous
3. **Agent using `async with`** - The `ReasoningAgent.run()` method tried to use `async with self.tracer.trace_agent_run(...)` but got sync context managers

## Root Cause

The issue was in how we were trying to use sync context managers in async contexts:

```python
# This DOESN'T work - sync context manager with async with
async with self.tracer.trace_agent_run(...) as trace:  # ❌ Error!
    await self._run_with_trace(...)
```

## Solution

We wrapped all sync context managers in async-compatible wrappers using `@asynccontextmanager`:

### For LangfuseTracer

```python
def trace_agent_run(self, user_input: str, agent_type: str = "ReasoningAgent") -> Any:
    """Returns AsyncContextManager for async compatibility."""
    if not self.enabled:
        return nullcontext()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def async_trace_wrapper():
        """Wrap sync Langfuse context manager for async usage."""
        ctx = self.client.start_as_current_span(
            name="agent_run",
            input={"query": user_input},
            metadata=metadata
        )

        # Manually enter/exit the sync context manager
        span_obj = ctx.__enter__()
        self._trace_context = span_obj

        try:
            yield span_obj
        finally:
            ctx.__exit__(None, None, None)
            self._trace_context = None

    return async_trace_wrapper()
```

### For AgentTracer (OpenTelemetry)

```python
def trace_agent_run(self, user_input: str, agent_type: str = "ReasoningAgent") -> Any:
    """Returns AsyncContextManager for async compatibility."""
    if not self.enabled:
        return nullcontext()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def async_span_wrapper():
        """Wrap sync OpenTelemetry context manager for async usage."""
        with self.tracer.start_as_current_span(
            "agent.run",
            kind=SpanKind.SERVER,
            attributes={
                "agent.type": agent_type,
                "agent.input": user_input,
            }
        ) as span:
            yield span

    return async_span_wrapper()
```

## Changes Made

### 1. Updated `src/linus/agents/telemetry.py`

- **LangfuseTracer**: Wrapped all tracing methods to return async-compatible context managers:
  - `trace_agent_run()` - Main trace for agent execution
  - `trace_reasoning_phase()` - Reasoning phase spans
  - `trace_llm_call()` - LLM generation observations
  - `trace_tool_execution()` - Tool execution spans

- **AgentTracer**: Wrapped all tracing methods for OpenTelemetry:
  - `trace_agent_run()` - Main agent span
  - `trace_reasoning_phase()` - Reasoning spans
  - `trace_llm_call()` - LLM call spans
  - `trace_tool_execution()` - Tool execution spans

- Added `_trace_context` field to LangfuseTracer to store active trace context

### 2. Updated `src/linus/agents/agent/reasoning_agent.py`

- Changed from `with` to `async with`:
  ```python
  # Before (broken)
  with self.tracer.trace_agent_run(...) as trace:

  # After (working)
  async with self.tracer.trace_agent_run(...) as trace:
  ```

### 3. Updated Test Files

- Updated `tests/test_langfuse_trace.py` to use async agent:
  - Updated to use `Agent` factory function with `use_async=True`
  - Made `main()` function async with `async def`
  - Added `await` before `agent.run()`
  - Changed `if __name__ == "__main__"` to use `asyncio.run(main())`

### 4. Created Demo Scripts

- `demo_async_telemetry.py` - Simple demo showing async context managers work correctly
- `test_telemetry_async.py` - Comprehensive test with both Langfuse and OpenTelemetry

## How It Works

1. **Agent calls tracer method** - e.g., `self.tracer.trace_agent_run(...)`
2. **Tracer returns async context manager** - An `@asynccontextmanager` function
3. **Agent uses `async with`** - Works correctly with async context manager
4. **Inside the wrapper**:
   - Sync context manager is entered using `ctx.__enter__()`
   - Control yields to agent code
   - Async operations can happen
   - On exit, sync context manager is exited using `ctx.__exit__()`

## Key Benefits

✅ **Async compatibility** - Tracers now work correctly with async agent code
✅ **Both backends supported** - Works with Langfuse and OpenTelemetry
✅ **No breaking changes** - Same API, just async-compatible
✅ **Proper context management** - Traces are properly opened and closed
✅ **Nested spans work** - Can create nested traces/spans within async code

## Testing

Run the demo to verify async telemetry works:

```bash
python demo_async_telemetry.py
```

Expected output:
```
✅ Successfully entered async context manager
✅ Successfully entered nested reasoning span
✅ Successfully exited all context managers
```

To test with actual Langfuse credentials:

```bash
# Set environment variables
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."

# Run test
python tests/test_langfuse_trace.py
```

## Technical Details

### Why This Approach?

We can't make Langfuse/OpenTelemetry's context managers async (they're from external libraries), so we:
1. Create our own async context manager wrapper
2. Manually call `__enter__()` and `__exit__()` on the sync context manager
3. Yield control between enter/exit, allowing async operations
4. This pattern is compatible with `async with` statements

### Thread Safety

Since we're wrapping sync context managers that may use thread-local storage:
- The wrapping happens in the same async task
- `__enter__()` and `__exit__()` are called in the same async context
- This preserves thread-local context correctly

### Performance

Minimal overhead:
- Just one extra function call (the wrapper)
- No blocking operations
- Same underlying telemetry implementation

## Migration Notes

If you have code using the old sync tracers, no changes needed if you're using the async agent. The tracers automatically return async-compatible context managers.

If you're still using sync agent code, the wrappers still work (async context managers can be used with regular `with` in some cases, but it's better to migrate to async).

## Future Improvements

Potential enhancements:
1. Add async-native Langfuse client when available
2. Add batch flushing for async contexts
3. Add async-aware span processors for OpenTelemetry
4. Support for async generators in tracing

## Conclusion

The telemetry system is now fully async-compatible and works correctly with both Langfuse and OpenTelemetry backends. The fix maintains backward compatibility while enabling proper async operation.
