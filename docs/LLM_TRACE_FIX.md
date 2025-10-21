# LLM Output Trace Fix - Complete Implementation

## Problem Summary

Two LLM calls in the ReasoningAgent were **not being traced**, causing missing observability data:

1. **`_generate_response()`** - Line 850: Generates direct responses without tools
2. **`_check_completion()`** - Line 880: Checks if task is completed

These methods made LLM calls **without wrapping them in `trace_llm_call()`**, so:
- No traces appeared in Langfuse/OpenTelemetry
- No input/output data was captured
- No token usage was recorded
- Debugging was difficult

## Root Cause

### Missing Trace Wrappers

**Before (❌ Not Traced)**:
```python
async def _generate_response(self, task_description: str, context: str) -> str:
    messages = [...]

    # Direct LLM call - NOT TRACED
    response = await self.llm.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content
```

**After (✅ Properly Traced)**:
```python
async def _generate_response(self, task_description: str, context: str) -> str:
    messages = [...]

    # Wrapped in trace context
    async with self.telemetry.trace_llm_call(
        prompt=messages[1]["content"],
        model=self.model,
        call_type="generate",
        llm_name="generate_response"
    ):
        response = await self.llm.chat.completions.create(...)
        response_text = response.choices[0].message.content

        # Extract usage
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        # Update generation with output
        if hasattr(self.telemetry, 'update_generation'):
            self.telemetry.update_generation(
                output={"response": response_text},
                usage=usage
            )
```

## Solution Details

### File Modified
- **`src/linus/agents/agent/reasoning_agent.py`**

### Changes Made

#### 1. `_generate_response()` Method (Lines 850-903)

**What was added**:
- ✅ Wrapped LLM call in `trace_llm_call()` context manager
- ✅ Added usage extraction from response
- ✅ Added `update_generation()` call with output and usage
- ✅ Moved token tracking after context closes

**Trace hierarchy created**:
```
llm.generate_response (generation)
├── Input: Context + task description
├── Output: {"response": "generated text"}
└── Usage: {prompt_tokens, completion_tokens, total_tokens}
```

#### 2. `_check_completion()` Method (Lines 905-1011)

**What was added**:
- ✅ Wrapped LLM call in `trace_llm_call()` context manager
- ✅ Moved JSON parsing **inside** the context manager
- ✅ Added usage extraction from response
- ✅ Added `update_generation()` call with parsed output and usage
- ✅ Added error handling with generation updates
- ✅ Moved metrics tracking after context closes

**Trace hierarchy created**:
```
llm.check_completion (generation)
├── Input: Original request + execution history
├── Output: {is_complete, reasoning, next_action, missing_steps}
└── Usage: {prompt_tokens, completion_tokens, total_tokens}
```

## Complete Trace Hierarchy (After Fix)

### Before Fix
```
agent.default (trace)
├── reasoning_phase (span)
│   └── llm.gemma3:27b (generation) ✅
└── tool.calculator (span)
    └── llm.gemma3:27b (generation) ✅
```

**Missing**: 2 LLM calls were invisible!

### After Fix
```
agent.default (trace)
├── reasoning_phase (span)
│   └── llm.gemma3:27b (generation) ✅
├── tool.calculator (span)
│   └── llm.gemma3:27b (generation) ✅
├── llm.generate_response (generation) ✅ NEW!
└── llm.check_completion (generation) ✅ NEW!
```

**Result**: 100% LLM call coverage!

## Key Implementation Details

### 1. Context Manager Pattern

All LLM calls **must** follow this pattern:

```python
async with self.telemetry.trace_llm_call(
    prompt=prompt_text,
    model=self.model,
    call_type="type_name",
    llm_name="hierarchical_name"
):
    # 1. Make LLM call
    response = await self.llm.chat.completions.create(...)

    # 2. Extract data
    response_text = response.choices[0].message.content
    usage = extract_usage(response)

    # 3. Parse/process (INSIDE context)
    result = parse_json(response_text)

    # 4. Update generation (INSIDE context, BEFORE exit)
    if hasattr(self.telemetry, 'update_generation'):
        self.telemetry.update_generation(output=result, usage=usage)

# 5. Post-processing (OUTSIDE context)
self._update_token_usage(response)
```

### 2. Critical Timing

⚠️ **IMPORTANT**: `update_generation()` must be called **INSIDE** the context manager:

```python
# ✅ CORRECT
async with self.telemetry.trace_llm_call(...):
    response = await llm.create(...)
    self.telemetry.update_generation(output=result, usage=usage)  # Inside!

# ❌ WRONG - TOO LATE!
async with self.telemetry.trace_llm_call(...):
    response = await llm.create(...)

self.telemetry.update_generation(output=result, usage=usage)  # Outside - won't work!
```

### 3. Error Handling

Both success and error cases must update the generation:

```python
async with self.telemetry.trace_llm_call(...):
    response = await llm.create(...)
    response_text = response.choices[0].message.content

    try:
        result = json.loads(response_text)
        # Success case
        self.telemetry.update_generation(output=result, usage=usage)
    except json.JSONDecodeError as e:
        # Error case - still update!
        error_output = {
            "error": str(e),
            "failed_to_parse": response_text[:500]
        }
        self.telemetry.update_generation(output=error_output, usage=usage)
```

## Testing

### Automated Test

Run the verification test:

```bash
python tests/test_llm_trace_fix.py
```

**Expected output**:
```
============================================================
Verifying Method Implementations
============================================================
_generate_response() has trace_llm_call: ✅
_generate_response() has update_generation: ✅
_check_completion() has trace_llm_call: ✅
_check_completion() has update_generation: ✅

============================================================
Testing LLM Trace Coverage
============================================================
✅ Telemetry initialized: AgentTracer
✅ Agent created with telemetry: AgentTracer

Test 1: Testing _generate_response() tracing
------------------------------------------------------------
✅ Response received: ...
✅ Metrics: {...}

Test 2: Testing _check_completion() tracing
------------------------------------------------------------
✅ Response received: ...
✅ Metrics: {...}

============================================================
✅ ALL TESTS PASSED!
============================================================
```

### Manual Testing with Langfuse

1. **Set environment variables**:
   ```bash
   export LANGFUSE_PUBLIC_KEY="pk-..."
   export LANGFUSE_SECRET_KEY="sk-..."
   export LANGFUSE_HOST="https://cloud.langfuse.com"
   export TELEMETRY_ENABLED=true
   export TELEMETRY_EXPORTER=langfuse
   ```

2. **Run agent with tracing**:
   ```python
   from linus.agents.agent.factory import Agent
   from linus.agents.agent.tools import get_default_tools
   from linus.agents.telemetry import initialize_telemetry

   tracer = initialize_telemetry(
       service_name="my-agent",
       exporter_type="langfuse",
       enabled=True
   )

   agent = Agent(
       model="gemma3:27b",
       tools=get_default_tools(),
       tracer=tracer,
       use_async=True
   )

   # Test generate_response path
   response = await agent.run("What is the capital of France?")

   # Test check_completion path
   response = await agent.run("Calculate 42 * 17")

   # Flush traces
   tracer.flush()
   ```

3. **Check Langfuse Dashboard**:
   - Navigate to your project in Langfuse
   - You should now see:
     - `llm.generate_response` generations with input/output
     - `llm.check_completion` generations with input/output
     - Token usage stats for all calls

## Debugging

### Check if telemetry is enabled

```python
print(f"Telemetry enabled: {agent.telemetry.enabled}")
print(f"Telemetry type: {type(agent.telemetry)}")
```

### Check logs for trace updates

```bash
grep "\[LANGFUSE\]" your_log_file.log | grep "generation"
```

Expected:
```
[LANGFUSE] Creating generation span: llm.generate_response
[LANGFUSE] Updated generation with output
[LANGFUSE] Creating generation span: llm.check_completion
[LANGFUSE] Updated generation with parsed output
```

### Verify methods have tracing code

```python
import inspect
from linus.agents.agent.reasoning_agent import ReasoningAgent

source = inspect.getsource(ReasoningAgent._generate_response)
assert "trace_llm_call" in source
assert "update_generation" in source

source = inspect.getsource(ReasoningAgent._check_completion)
assert "trace_llm_call" in source
assert "update_generation" in source
```

## Benefits

1. **Complete Observability**: All LLM calls (100%) are now traced
2. **Structured Output**: Both raw responses and parsed results are captured
3. **Token Tracking**: Every LLM call includes token usage statistics
4. **Error Visibility**: Parsing failures are logged with error details
5. **Better Debugging**: Can see exactly what the LLM returned at each step
6. **Cost Tracking**: Full token usage for accurate cost calculation

## Summary of All Traced LLM Calls

| Method | Call Type | Trace Name | Status |
|--------|-----------|------------|--------|
| `_reasoning_call()` | Reasoning | `llm.{model}` (reasoning) | ✅ Already traced |
| `_generate_tool_arguments()` | Tool args | `llm.{model}` (tool_args) | ✅ Already traced |
| **`_generate_response()`** | Direct response | **`llm.generate_response`** | ✅ **FIXED** |
| **`_check_completion()`** | Completion check | **`llm.check_completion`** | ✅ **FIXED** |

## Files Modified

- **`src/linus/agents/agent/reasoning_agent.py`**:
  - `_generate_response()` method (lines 850-903)
  - `_check_completion()` method (lines 905-1011)

## Test Files Created

- **`tests/test_llm_trace_fix.py`**: Automated verification test

## Status

✅ **COMPLETE** - All LLM calls are now properly traced with output capture

## Related Documentation

- [TRACING_OUTPUT_FIX.md](./TRACING_OUTPUT_FIX.md) - Previous fix for reasoning/tool_args
- [TELEMETRY.md](./TELEMETRY.md) - Telemetry configuration guide
- [LANGFUSE_INTEGRATION.md](./LANGFUSE_INTEGRATION.md) - Langfuse setup
