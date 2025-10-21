# LLM Trace Fix - Quick Summary

## What Was Fixed

Two LLM calls were **not being traced**:
1. ❌ `_generate_response()` - Direct response generation
2. ❌ `_check_completion()` - Task completion checking

## Why Traces Were Missing

These methods made LLM calls **without the tracing wrapper**:

```python
# ❌ NOT TRACED
response = await self.llm.chat.completions.create(...)
```

Should have been:

```python
# ✅ PROPERLY TRACED
async with self.telemetry.trace_llm_call(...):
    response = await self.llm.chat.completions.create(...)
    self.telemetry.update_generation(output=result, usage=usage)
```

## Changes Made

### File: `src/linus/agents/agent/reasoning_agent.py`

#### 1. `_generate_response()` (Lines 850-903)
- ✅ Added `trace_llm_call()` wrapper
- ✅ Added `update_generation()` with output and usage
- ✅ Extracts token usage

#### 2. `_check_completion()` (Lines 905-1011)
- ✅ Added `trace_llm_call()` wrapper
- ✅ Added `update_generation()` with output and usage
- ✅ Moved parsing inside context
- ✅ Added error handling with trace updates

## Verify the Fix

Run the test:

```bash
python tests/test_llm_trace_fix.py
```

Expected:
```
✅ _generate_response() has trace_llm_call: ✅
✅ _generate_response() has update_generation: ✅
✅ _check_completion() has trace_llm_call: ✅
✅ _check_completion() has update_generation: ✅
✅ ALL TESTS PASSED!
```

## Trace Coverage (Before vs After)

### Before Fix
- ✅ Reasoning LLM calls: **TRACED**
- ✅ Tool argument generation: **TRACED**
- ❌ Direct response generation: **MISSING** 👈 Fixed!
- ❌ Completion checking: **MISSING** 👈 Fixed!

### After Fix
- ✅ Reasoning LLM calls: **TRACED**
- ✅ Tool argument generation: **TRACED**
- ✅ Direct response generation: **TRACED** 🎉
- ✅ Completion checking: **TRACED** 🎉

## What You'll See in Langfuse Now

New trace hierarchy:

```
agent.default
├── reasoning_phase
│   └── llm.gemma3:27b (reasoning) ✅
├── tool.calculator
│   └── llm.gemma3:27b (tool_args) ✅
├── llm.generate_response ✅ NEW!
│   ├── Input: task + context
│   ├── Output: {"response": "..."}
│   └── Usage: tokens
└── llm.check_completion ✅ NEW!
    ├── Input: request + history
    ├── Output: {is_complete, reasoning}
    └── Usage: tokens
```

## Key Takeaway

**All LLM calls must be wrapped in `trace_llm_call()` and call `update_generation()` inside the context.**

## Full Documentation

See [docs/LLM_TRACE_FIX.md](docs/LLM_TRACE_FIX.md) for complete details.
