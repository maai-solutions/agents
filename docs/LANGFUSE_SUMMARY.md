# Langfuse Integration Summary

## What Was Added

Comprehensive Langfuse support has been added to the ReasoningAgent framework for LLM-specific observability and monitoring.

## Changes Made

### 1. Core Telemetry Module (`src/linus/agents/telemetry.py`)

**New Imports:**
- `langfuse` - Langfuse client
- `langfuse.decorators` - Decorators for tracing
- `time` - For latency tracking

**Updated Configuration (`TelemetryConfig`):**
- Added `langfuse_public_key` parameter
- Added `langfuse_secret_key` parameter
- Added `langfuse_host` parameter (defaults to `https://cloud.langfuse.com`)
- Added `exporter_type="langfuse"` option
- Environment variable support: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

**New Class: `LangfuseTracer`**

A complete tracer implementation with the same API as `AgentTracer` but optimized for Langfuse:

- `trace_agent_run()` - Creates top-level traces for agent execution
- `trace_reasoning_phase()` - Creates spans for reasoning iterations
- `trace_llm_call()` - Creates generation objects for LLM calls with token/latency tracking
- `trace_tool_execution()` - Creates spans for tool executions
- `add_event()` - Adds events to current trace
- `set_attribute()` - Sets metadata on traces
- `record_exception()` - Records exceptions with ERROR level
- `set_status()` - Sets trace status (OK/ERROR)
- `flush()` - Flushes pending traces to Langfuse

**Updated Functions:**
- `setup_telemetry()` - Now supports Langfuse initialization
- `initialize_telemetry()` - Returns `LangfuseTracer` when `exporter_type="langfuse"`
- `get_tracer()` - Can return either `AgentTracer` or `LangfuseTracer`
- `is_langfuse_available()` - New helper to check Langfuse availability

### 2. Dependencies (`requirements.txt`)

Added:
```
# Langfuse observability
langfuse
```

### 3. Documentation

**New Files:**
- `docs/LANGFUSE_INTEGRATION.md` - Complete guide to using Langfuse
  - Installation instructions
  - Configuration steps
  - Feature overview
  - Dashboard usage
  - Troubleshooting
  - Comparison with OpenTelemetry
  - Advanced usage examples

**Updated Files:**
- `docs/TELEMETRY.md` - Added Langfuse to supported backends
- `README.md` - Updated observability feature to mention Langfuse
- `CLAUDE.md` - Added Langfuse configuration and usage example

### 4. Examples

**New File: `examples/langfuse_example.py`**

Complete working example demonstrating:
- Credential validation
- Tracer initialization
- Agent creation with Langfuse
- Running multiple queries
- Trace flushing
- Viewing results in dashboard

## Features

### Automatic Trace Structure

Langfuse creates hierarchical traces:

```
agent_run (Trace)
├── reasoning_phase (Span)
│   └── llm_reasoning (Generation)
├── tool_calculator (Span)
│   └── llm_tool_args (Generation)
└── llm_generate (Generation)
```

### LLM-Specific Tracking

- **Generations**: Dedicated objects for LLM calls with prompt/completion tracking
- **Token Usage**: Automatic token counting (if provided by LLM)
- **Latency**: Automatic latency measurement for all LLM calls
- **Cost Tracking**: Langfuse can calculate costs based on model pricing
- **Error Handling**: Exceptions automatically captured with ERROR level

### Context Managers

All tracing uses context managers for clean resource management:

```python
with tracer.trace_llm_call(prompt, model, "reasoning") as generation:
    # LLM call happens here
    # Context automatically tracks timing and errors
    pass
```

### Null Context Support

When Langfuse is disabled or unavailable, returns `nullcontext()` to avoid overhead.

## Usage

### Quick Start

```python
from linus.agents.telemetry import initialize_telemetry
from linus.agents.agent import create_gemma_agent

# Initialize Langfuse
tracer = initialize_telemetry(
    service_name="my-agent",
    exporter_type="langfuse",
    enabled=True
)

# Create agent with tracer
agent = create_gemma_agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tracer=tracer
)

# Run queries (automatically traced)
response = agent.run("What is 42 * 17?")

# Flush before exit
tracer.flush()
```

### Environment Variables

Required in `.env`:
```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # Optional
```

## Benefits Over OpenTelemetry

### Langfuse Advantages

1. **LLM-Native**: Designed specifically for LLM applications
2. **Token Tracking**: Built-in support for tracking tokens and costs
3. **Prompt Management**: Can version and manage prompts
4. **Dataset Creation**: Build evaluation datasets from production traces
5. **User Feedback**: Collect and track user feedback on outputs
6. **Cost Analysis**: Automatic cost calculation per model

### When to Use Langfuse

- ✅ LLM-focused applications
- ✅ Need prompt versioning
- ✅ Want to track costs/tokens
- ✅ Building evaluation datasets
- ✅ Analyzing LLM performance

### When to Use OpenTelemetry

- ✅ Multi-service distributed tracing
- ✅ Integration with existing observability stack
- ✅ Vendor-neutral requirements
- ✅ Generic application tracing

## Architecture

### Trace Lifecycle

1. **Agent Run Starts**: Creates root trace with user input
2. **Reasoning Phase**: Creates span with iteration context
3. **LLM Call**: Creates generation with prompt/model
4. **Tool Execution**: Creates span with tool name/args
5. **Completion**: Updates trace with final output
6. **Flush**: Sends all traces to Langfuse asynchronously

### Error Handling

Errors are captured at multiple levels:
- **LLM Errors**: Recorded in generation object
- **Tool Errors**: Recorded in tool span
- **Agent Errors**: Recorded in root trace

All errors set level to `ERROR` for easy filtering in dashboard.

## Dashboard Features

Once traces are in Langfuse, you can:

1. **View Traces**: See complete execution hierarchy
2. **Filter by Status**: Find errors quickly
3. **Analyze Performance**: Track latency and token usage
4. **Search Prompts**: Find specific prompts/completions
5. **Create Datasets**: Build test sets from real queries
6. **Track Costs**: Monitor LLM spending
7. **User Feedback**: Correlate traces with user satisfaction

## Testing

To test the integration:

```bash
# Install Langfuse
pip install langfuse

# Set credentials in .env
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

# Run example
python examples/langfuse_example.py

# View traces at https://cloud.langfuse.com
```

## Compatibility

- **Python**: 3.8+
- **Langfuse SDK**: Latest version
- **Self-Hosted**: Fully supported (set `LANGFUSE_HOST`)
- **Async**: Compatible with both sync and async agents

## Future Enhancements

Potential improvements:
- [ ] Add prompt versioning support
- [ ] Integrate user feedback collection
- [ ] Add cost tracking for specific models
- [ ] Support for dataset creation from traces
- [ ] Add evaluation metrics integration
- [ ] Session tracking across multiple queries

## Related Documentation

- [LANGFUSE_INTEGRATION.md](LANGFUSE_INTEGRATION.md) - Detailed integration guide
- [TELEMETRY.md](TELEMETRY.md) - General telemetry documentation
- [TELEMETRY_IMPLEMENTATION.md](TELEMETRY_IMPLEMENTATION.md) - Implementation details
- [examples/langfuse_example.py](../examples/langfuse_example.py) - Working example

## Migration from OpenTelemetry

To switch from OpenTelemetry to Langfuse:

1. Update `.env` with Langfuse credentials
2. Change `exporter_type` from `"console"` to `"langfuse"`
3. No code changes needed - same tracer API
4. Flush traces with `tracer.flush()` before exit

```python
# Before (OpenTelemetry)
tracer = initialize_telemetry(
    exporter_type="console",
    enabled=True
)

# After (Langfuse)
tracer = initialize_telemetry(
    exporter_type="langfuse",
    enabled=True
)
```

## Resources

- **Langfuse Cloud**: https://cloud.langfuse.com
- **Documentation**: https://langfuse.com/docs
- **Python SDK**: https://langfuse.com/docs/sdk/python
- **Self-Hosting**: https://langfuse.com/docs/deployment/self-host
- **GitHub**: https://github.com/langfuse/langfuse
