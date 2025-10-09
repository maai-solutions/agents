# Async/Await Fix for ReasoningAgent

## Problem
The error occurred when using `AsyncOpenAI` client with the `ReasoningAgent`:

```
AttributeError: 'coroutine' object has no attribute 'choices'
```

This happened because some methods were calling `self.llm.chat.completions.create()` without `await`, even though `self.llm` was an `AsyncOpenAI` instance. When you call an async method without `await`, you get a coroutine object instead of the actual result.

## Root Cause
The `ReasoningAgent` supports both `OpenAI` and `AsyncOpenAI` clients (line 42 in `reasoning_agent.py`), but two methods were not properly handling the async case:

1. `_format_final_response_with_history()` (line 477)
2. `_format_final_response()` (line 434) - legacy method

These methods were:
- Not declared as `async`
- Calling `self.llm.chat.completions.create()` without `await`

## Solution
Made both methods async and added `await` to their LLM calls:

### Changes Made

#### 1. `_format_final_response_with_history()` (lines 477-547)
**Before:**
```python
@trace_method("agent.final_formatting")
def _format_final_response_with_history(self, input_text: str, ...) -> str:
    # ...
    response = self.llm.chat.completions.create(  # ❌ Missing await
        messages=messages,
        **self._get_generation_kwargs()
    )
```

**After:**
```python
@trace_method("agent.final_formatting")
async def _format_final_response_with_history(self, input_text: str, ...) -> str:
    # ...
    response = await self.llm.chat.completions.create(  # ✅ Added await
        messages=messages,
        **self._get_generation_kwargs()
    )
```

**Updated call site (line 374):**
```python
# Before
final_result = self._format_final_response_with_history(input_text, execution_history, completion_status)

# After
final_result = await self._format_final_response_with_history(input_text, execution_history, completion_status)
```

#### 2. `_format_final_response()` (lines 434-474)
**Before:**
```python
@trace_method("agent.reasoning")
def _format_final_response(self, original_request: str, results: List[str]) -> str:
    # ...
    response = self.llm.chat.completions.create(  # ❌ Missing await
        messages=messages,
        **self._get_generation_kwargs()
    )
```

**After:**
```python
@trace_method("agent.reasoning")
async def _format_final_response(self, original_request: str, results: List[str]) -> str:
    # ...
    response = await self.llm.chat.completions.create(  # ✅ Added await
        messages=messages,
        **self._get_generation_kwargs()
    )
```

## Files Modified
- [src/linus/agents/agent/reasoning_agent.py](src/linus/agents/agent/reasoning_agent.py)
  - Line 374: Added `await` to `_format_final_response_with_history()` call
  - Line 434: Made `_format_final_response()` async
  - Line 464: Added `await` to LLM call in `_format_final_response()`
  - Line 477: Made `_format_final_response_with_history()` async
  - Line 532: Added `await` to LLM call in `_format_final_response_with_history()`

## Verification
All other LLM calls in the file were checked and confirmed to be properly awaited:
- ✅ Line 582 - `_reasoning_call()` - Async method with `await`
- ✅ Line 728 - `_generate_tool_arguments()` - Async method with `await`
- ✅ Line 791 - `_generate_response()` - Async method with `await`
- ✅ Line 831 - `_check_completion()` - Async method with `await`

## Impact
This fix ensures that the `ReasoningAgent` works correctly with both:
- **Synchronous** `OpenAI` client (returns results directly)
- **Asynchronous** `AsyncOpenAI` client (returns coroutines that must be awaited)

The error `'coroutine' object has no attribute 'choices'` should no longer occur when using `AsyncOpenAI`.

## Testing
To verify the fix works:
```bash
# Start the FastAPI server (uses AsyncOpenAI)
python src/app.py

# In another terminal, test the endpoint
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate 2 multiplied by 10"}'
```

The agent should now properly handle async LLM calls and return results without errors.
