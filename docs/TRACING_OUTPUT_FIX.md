# Tracing Output Fix - Complete Summary

## Problem
The Langfuse traces were missing output data for:
1. **`reasoning_phase`** span - No structured output from reasoning
2. **`llm_reasoning`** generation - Only had raw LLM response, not parsed results
3. **`llm_tool_args`** generation - Missing parsed tool arguments

## Solution

### File: `src/linus/agents/agent/reasoning_agent.py`

#### Fix 1: Reasoning Phase Output Tracing (Lines 606-657)

**What was changed:**
- Moved JSON parsing logic inside the reasoning span context
- Added structured output update after successful parsing
- Added error tracking for parsing failures

**Code changes:**
```python
# After parsing the reasoning result successfully:
if reasoning_span and hasattr(self.tracer, 'client'):
    output_data = {
        "has_sufficient_info": result.has_sufficient_info,
        "reasoning": result.reasoning,
        "tasks_count": len(result.tasks),
        "tasks": result.tasks
    }
    self.tracer.client.update_current_span(output=output_data)

# On parsing error:
if reasoning_span and hasattr(self.tracer, 'client'):
    self.tracer.client.update_current_span(
        output={"error": str(e), "raw_response": response_text[:200]},
        level="ERROR"
    )
```

**Result:**
- `reasoning_phase` span now shows structured output with:
  - `has_sufficient_info`: Boolean flag
  - `reasoning`: Analysis text
  - `tasks_count`: Number of planned tasks
  - `tasks`: Array of task objects

#### Fix 2: Tool Arguments Output Tracing (Lines 754-823)

**What was changed:**
- Moved JSON parsing inside the LLM trace context
- Added structured output update with both raw and parsed data
- Added error tracking for parsing failures

**Code changes:**
```python
# After successfully parsing tool arguments:
if llm_span and hasattr(self.tracer, 'update_generation'):
    output_data = {
        "raw_response": response_text,
        "parsed_args": args,
        "tool": tool.name
    }
    self.tracer.update_generation(output_data, usage)

# On parsing error:
if llm_span and hasattr(self.tracer, 'update_generation'):
    error_output = {
        "raw_response": response_text,
        "error": str(e),
        "tool": tool.name
    }
    self.tracer.update_generation(error_output, usage)
```

**Result:**
- `llm_tool_args` generation now shows:
  - `raw_response`: Original LLM output
  - `parsed_args`: Extracted JSON arguments
  - `tool`: Tool name for context
  - Token usage statistics

## Expected Behavior in Langfuse

### Complete Trace Hierarchy

```
agent_run (trace)
├── reasoning_phase (span)
│   ├── Input: User query + context
│   ├── Output: {has_sufficient_info, reasoning, tasks_count, tasks}
│   └── llm_reasoning (generation)
│       ├── Input: Reasoning prompt
│       ├── Output: Raw LLM JSON response
│       └── Usage: {prompt_tokens, completion_tokens, total_tokens}
├── tool_calculator (span)
│   ├── Input: Tool arguments
│   ├── Output: Tool execution result
│   └── llm_tool_args (generation)
│       ├── Input: Tool argument generation prompt
│       ├── Output: {raw_response, parsed_args, tool}
│       └── Usage: {prompt_tokens, completion_tokens, total_tokens}
└── agent_run completion
    ├── Output: Final formatted response
    └── Metadata: {completed, iterations, execution_time}
```

## Verification

Run the verification script to confirm all fixes are in place:

```bash
PYTHONPATH=/Users/udg/Projects/Git/agents/src python verify_trace_fix.py
```

Expected output:
```
✅ SUCCESS: All reasoning tracing checks passed!
✅ SUCCESS: All tool args tracing checks passed!
✅ ALL VERIFICATIONS PASSED!
```

## Benefits

1. **Complete Observability**: Every LLM call now has both input and output traced
2. **Structured Data**: Parsed results are captured alongside raw responses
3. **Error Tracking**: Parsing failures are logged with error details
4. **Debugging**: Easier to identify issues in reasoning or tool argument generation
5. **Token Tracking**: All LLM calls include token usage statistics

## Files Modified

- `src/linus/agents/agent/reasoning_agent.py`:
  - `_reasoning_call()` method (lines 550-657)
  - `_generate_tool_arguments()` method (lines 721-823)

## Testing

To test with a live Langfuse instance:

1. Set environment variables:
   ```bash
   export LANGFUSE_PUBLIC_KEY="pk-..."
   export LANGFUSE_SECRET_KEY="sk-..."
   export LANGFUSE_HOST="https://cloud.langfuse.com"
   ```

2. Run the agent with tracing enabled:
   ```python
   from linus.agents.agent.factory import Agent
   from linus.agents.telemetry import initialize_telemetry

   tracer = initialize_telemetry(
       service_name="test-agent",
       exporter_type="langfuse",
       session_id="test-session-123",
       enabled=True
   )

   agent = Agent(
       model="gemma3:27b",
       tools=get_default_tools(),
       tracer=tracer,
       use_async=True
   )

   response = await agent.run("What is 42 * 17?")
   tracer.flush()
   ```

3. Check Langfuse dashboard for complete traces with all outputs populated

## Status

✅ **COMPLETE** - All tracing output fixes implemented and verified
