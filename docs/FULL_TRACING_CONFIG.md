# Full Tracing Configuration (No Truncation)

## Overview

All truncation has been removed from the telemetry and agent tracing code to capture complete inputs, outputs, and tool data without any loss of information.

## Changes Made

### 1. Telemetry Layer (`src/linus/agents/telemetry.py`)

#### LangfuseTracer
- **Line 242**: Removed truncation from reasoning phase input text
  - Before: `input={"text": input_text[:500], ...}`
  - After: `input={"text": input_text, ...}`

- **Line 278**: Removed truncation from LLM call prompts
  - Before: `input=prompt[:1000]`
  - After: `input=prompt`

#### AgentTracer (OpenTelemetry)
- **Line 447**: Removed truncation from agent run input
  - Before: `"agent.input": user_input[:500]`
  - After: `"agent.input": user_input`

- **Line 472**: Removed truncation from reasoning phase input
  - Before: `"agent.reasoning.input": input_text[:500]`
  - After: `"agent.reasoning.input": input_text`

- **Line 501**: Removed truncation from LLM prompts
  - Before: `"llm.prompt": prompt[:1000]`
  - After: `"llm.prompt": prompt`

- **Line 528**: Removed truncation from tool arguments
  - Before: `"tool.args": str(tool_args)[:500]`
  - After: `"tool.args": str(tool_args)`

### 2. Agent Layer (`src/linus/agents/agent/reasoning_agent.py`)

#### Tracing Attributes
- **Line 641**: LLM input attribute - no truncation
- **Line 686**: LLM output attribute - no truncation
- **Line 758**: Tool arguments attribute - no truncation
- **Line 778**: Tool results attribute - no truncation

#### Trace Updates
- **Line 380**: Agent run output update - no truncation
- **Line 773**: Tool span output update - no truncation

#### Context Building
- **Lines 285, 494**: Execution history in context - no truncation
  - Full task results now included in reasoning context

#### Completion Checks
- **Lines 920, 1304**: History formatting - no truncation
  - All task results fully included in completion checks
- **Line 935**: Debug logging - no truncation

#### Error Handling
- **Lines 711, 716**: Sync reasoning error messages - no truncation
- **Lines 1136, 1140**: Async reasoning error messages - no truncation

## Benefits

### 1. Complete Observability
- **Full prompt tracking**: Every character sent to the LLM is captured
- **Complete responses**: All LLM outputs are traced without loss
- **Full tool data**: Tool inputs and outputs are traced completely
- **Streaming/chunks**: All chunks are captured without truncation

### 2. Better Debugging
- See exact inputs that caused errors
- Understand complete context that agents are working with
- Trace entire execution chains without missing data

### 3. Enhanced Analysis
- Accurate token counting and usage tracking
- Complete conversation history for memory systems
- Full context for completion checks and multi-step reasoning

### 4. Production Readiness
- Langfuse will capture complete traces for cost analysis
- OpenTelemetry exporters receive full span data
- No information loss in distributed tracing

## Configuration

No additional configuration is needed. The changes apply automatically to:

- **Langfuse tracing**: When using `exporter_type="langfuse"`
- **OpenTelemetry tracing**: When using `exporter_type="console"`, `"otlp"`, or `"jaeger"`
- **Agent execution**: All sync and async agent methods

## Usage Example

```python
from linus.agents.telemetry import initialize_telemetry
from linus.agents.agent.agent import create_gemma_agent

# Initialize with full tracing
tracer = initialize_telemetry(
    service_name="my-agent",
    exporter_type="langfuse",
    enabled=True
)

# Create agent - all data is fully traced
agent = create_gemma_agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    tracer=tracer
)

# Run queries - everything is captured without truncation
response = agent.run("Complex multi-step query with long output")

# View complete traces in Langfuse dashboard
tracer.flush()
```

## Memory Considerations

### Important Notes

1. **Large outputs**: With no truncation, very large outputs will be stored in full
   - Monitor Langfuse/OTEL collector storage if dealing with multi-MB responses

2. **Token limits**: Agent context windows still apply
   - Full traces are separate from context window limits
   - Memory system handles context window management

3. **Network overhead**: Complete traces may increase network traffic to observability backends
   - Batch processing helps mitigate this (already configured)

## Monitoring Recommendations

1. **Set up alerts** in your observability platform for:
   - Unusually large span sizes (>100KB)
   - High trace ingestion rates

2. **Use Langfuse filters** to:
   - Search by session_id for debugging specific user sessions
   - Filter by trace size for performance analysis
   - Query by token usage for cost optimization

3. **Consider sampling** (optional):
   - For high-volume production, implement trace sampling
   - Keep 100% tracing for development/staging

## Rollback

If you need to restore truncation (e.g., for storage optimization), revert by:

```bash
# Restore previous version
git checkout HEAD~1 -- src/linus/agents/telemetry.py src/linus/agents/agent/reasoning_agent.py
```

Or manually add back truncation with slice syntax like `[:500]`, `[:1000]`, etc.

## Related Documentation

- [Telemetry Guide](TELEMETRY.md)
- [Langfuse Integration](LANGFUSE_INTEGRATION.md)
- [Agent Architecture](../CLAUDE.md)
