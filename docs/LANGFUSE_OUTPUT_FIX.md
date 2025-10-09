# Langfuse Output Capture Fix

## Problem
The LLM reasoning and tool argument outputs were showing as empty in Langfuse traces.

## Root Cause
1. **Wrong parameter name**: `update_current_generation()` expects `usage_details` not `usage`
2. **Output not captured**: The `update_generation()` method wasn't being called correctly with the parsed outputs
3. **Context timing**: Updates need to happen inside the trace context managers
4. **Span vs Generation**: Both the generation (LLM call) and the parent span (reasoning_phase) needed to be updated

## Changes Made

### 1. Fixed `telemetry.py` - `LangfuseTracer.update_generation()`
**File**: `src/linus/agents/telemetry.py:428-453`

**Before**:
```python
def update_generation(self, output: str, usage: Optional[Dict[str, int]] = None):
    # ...
    if usage:
        self.client.update_current_generation(output=output, usage=usage)  # ❌ Wrong param
    else:
        self.client.update_current_generation(output=output)
```

**After**:
```python
def update_generation(self, output: Any, usage: Optional[Dict[str, int]] = None):
    # ...
    update_kwargs = {"output": output}
    if usage:
        update_kwargs["usage_details"] = usage  # ✅ Correct param name

    self.client.update_current_generation(**update_kwargs)
    logger.info(f"[LANGFUSE] Successfully updated generation with output and usage")
```

### 2. Fixed Reasoning Phase Output Capture
**File**: `src/linus/agents/agent/reasoning_agent.py:560-664`

**Key changes**:
- Parse the response INSIDE the `trace_llm_call` context
- Call `update_generation()` with parsed output while still inside the LLM context
- Additionally update the parent `reasoning_phase` span with output after LLM context closes

```python
async with self.tracer.trace_reasoning_phase(input_text, iteration):
    # ...
    async with self.tracer.trace_llm_call(...):
        response = await self.llm.chat.completions.create(...)
        response_text = response.choices[0].message.content
        usage = {...}  # Extract usage stats

        # Parse response INSIDE context
        result = ReasoningResult(...)

        # Update generation INSIDE context
        if hasattr(self.tracer, 'update_generation'):
            output_data = {
                "has_sufficient_info": result.has_sufficient_info,
                "reasoning": result.reasoning,
                "tasks_count": len(result.tasks),
                "tasks": result.tasks
            }
            self.tracer.update_generation(output=output_data, usage=usage)

    # Update parent span AFTER LLM context but INSIDE reasoning_phase context
    if hasattr(self.tracer, 'client') and self.tracer.enabled:
        self.tracer.client.update_current_span(output=span_output)
```

### 3. Fixed Tool Arguments Output Capture
**File**: `src/linus/agents/agent/reasoning_agent.py:747-811`

**Key changes**:
- Parse tool arguments INSIDE the `trace_llm_call` context
- Call `update_generation()` with parsed args while still inside context
- Handle both success and error cases

```python
async with self.tracer.trace_llm_call(prompt, self.model, "tool_args"):
    response = await self.llm.chat.completions.create(...)
    response_text = response.choices[0].message.content
    usage = {...}

    try:
        # Parse arguments INSIDE context
        args = json.loads(...)

        # Update generation INSIDE context
        if hasattr(self.tracer, 'update_generation'):
            self.tracer.update_generation(output=args, usage=usage)

    except (json.JSONDecodeError, KeyError) as e:
        # Update with error INSIDE context
        if hasattr(self.tracer, 'update_generation'):
            error_output = {"error": str(e), "failed_to_parse": response_text[:500]}
            self.tracer.update_generation(output=error_output, usage=usage)
        args = None

# Track metrics AFTER context closes
self._update_token_usage(response)
```

## What Gets Captured in Langfuse Now

### 1. **agent_run** (Trace)
- Input: User query
- Output: Final agent response
- Metadata: Completion status, iterations, execution time

### 2. **reasoning_phase** (Span)
- Input: Current context and iteration
- **Output**: ✅ Parsed reasoning result with:
  - `has_sufficient_info`
  - `reasoning` text
  - `tasks_count`
  - `tasks` array

### 3. **llm_reasoning** (Generation)
- Input: Reasoning prompt
- Model: gemma3:27b (or configured model)
- **Output**: ✅ Parsed JSON object with reasoning and tasks
- **Usage**: ✅ prompt_tokens, completion_tokens, total_tokens

### 4. **llm_tool_args** (Generation)
- Input: Tool argument generation prompt
- Model: gemma3:27b (or configured model)
- **Output**: ✅ Parsed tool arguments JSON
- **Usage**: ✅ Token counts

### 5. **tool_{name}** (Span)
- Input: Tool arguments
- Output: Tool execution result

## Testing

Run with Langfuse credentials:

```bash
# Set environment variables in .env
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Run test
PYTHONPATH=/Users/udg/Projects/Git/agents/src python test_langfuse_output.py
```

## Verification Checklist

In Langfuse UI, verify:
- [ ] `llm_reasoning` generation has non-empty output with parsed JSON
- [ ] `llm_reasoning` generation has usage_details with token counts
- [ ] `llm_tool_args` generation has non-empty output with tool arguments
- [ ] `llm_tool_args` generation has usage_details with token counts
- [ ] `reasoning_phase` span has output with reasoning result
- [ ] All traces are properly nested

## Key Learnings

1. **Langfuse API expects `usage_details`, not `usage`** for token counts
2. **Update calls must happen INSIDE context managers** - Langfuse tracks the "current" generation/span
3. **Parse before updating** - Don't update with raw LLM text, parse it first
4. **Both spans and generations** can have outputs - update both when appropriate
5. **Structured output is better** - Pass dicts/objects instead of strings for better Langfuse UI display
