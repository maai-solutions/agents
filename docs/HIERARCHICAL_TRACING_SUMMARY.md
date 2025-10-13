# Hierarchical Tracing Implementation Summary

## Overview

Implemented hierarchical naming for traces using the pattern `agent.<name>`, `tool.<name>`, and `llm.<name>` to provide clear, organized observability across the agent framework.

## Changes Made

### 1. Telemetry Module (`src/linus/agents/telemetry.py`)

#### LangfuseTracer Updates
- Added `agent_name` parameter to `__init__` (defaults to "default")
- Updated `trace_agent_run()` to create traces named `agent.<name>`
- Updated `trace_llm_call()` to create generations named `llm.<name>`
- Updated `trace_tool_execution()` to create spans named `tool.<name>`
- Added `llm_name` and `tool_display_name` parameters for custom naming

#### AgentTracer Updates (OpenTelemetry)
- Added `agent_name` parameter to `__init__` (defaults to "default")
- Updated `trace_agent_run()` to create spans named `agent.<name>`
- Updated `trace_llm_call()` to create spans named `llm.<name>`
- Updated `trace_tool_execution()` to create spans named `tool.<name>`
- Added `llm_name` and `tool_display_name` parameters for custom naming

#### initialize_telemetry() Updates
- Added `agent_name` parameter
- Passes `agent_name` to both LangfuseTracer and AgentTracer constructors

### 2. Base Agent Class (`src/linus/agents/agent/base.py`)

- Added `agent_name` parameter to `__init__` (defaults to "default")
- Stores `agent_name` as instance variable: `self.agent_name`
- Updates telemetry tracer's `agent_name` if supported

### 3. ReasoningAgent Class (`src/linus/agents/agent/reasoning_agent.py`)

- Added `agent_name` parameter to `__init__`
- Passes `agent_name` to parent `Agent` class
- Updated `trace_agent_run()` call to pass `self.agent_name`

### 4. Agent Factory (`src/linus/agents/agent/factory.py`)

- Added `agent_name` parameter to `Agent()` factory function
- Passes `agent_name` to `ReasoningAgent` constructor
- Updates tracer's `agent_name` when custom tracer is provided
- Documents the hierarchical naming feature in docstring

## New Files Created

### 1. Example: `example_hierarchical_tracing.py`

Comprehensive example demonstrating:
- Creating multiple agents with different names (calculator, search, file_ops)
- How traces are organized hierarchically
- Integration with Langfuse/OpenTelemetry
- Session grouping with hierarchical names

### 2. Documentation: `docs/HIERARCHICAL_TRACING.md`

Complete documentation covering:
- Overview of hierarchical naming convention
- Naming convention details (agent.*, llm.*, tool.*)
- Usage examples (single agent, multiple agents, sessions)
- Configuration options
- Best practices for naming
- Viewing traces in different backends
- Implementation details
- Troubleshooting guide

## Naming Convention

### Agent Traces: `agent.<name>`
- Set via `agent_name` parameter when creating agent
- Example: `agent.calculator`, `agent.search`, `agent.file_ops`
- Default: `agent.default`

### LLM Traces: `llm.<name>`
- Automatically set based on the model name:
  - `llm.gemma3:27b` - Traces for Gemma3 27B model
  - `llm.gpt-4` - Traces for GPT-4 model
  - `llm.llama3:70b` - Traces for Llama3 70B model
- Model name is extracted from agent configuration automatically
- Makes it easy to identify and compare traces from different models

### Tool Traces: `tool.<name>`
- Automatically derived from tool name
- Example: `tool.calculator`, `tool.search`, `tool.read_file`
- Can be customized via `tool_display_name` parameter

## Usage Example

```python
import asyncio
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

async def main():
    # Initialize telemetry
    tracer = initialize_telemetry(
        service_name="my-app",
        exporter_type="langfuse",
        enabled=True
    )

    # Create agent with custom name
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        agent_name="calculator",  # Creates "agent.calculator" traces
        tracer=tracer,
        use_async=True
    )

    # Run task
    response = await agent.run("Calculate 42 * 17")

    # Flush traces
    tracer.flush()

asyncio.run(main())
```

**Generated trace hierarchy:**
```
agent.calculator
├── llm.gemma3:27b         (Planning phase)
├── llm.gemma3:27b         (Generating tool args)
└── tool.calculator        (Executing tool)
```

## Benefits

1. **Clear Organization**: Easy to identify traces by component type
2. **Better Filtering**: Filter traces by agent, LLM, or tool
3. **Multi-Agent Support**: Distinguish between different agents in the same system
4. **Consistent Naming**: Standard convention across the entire framework
5. **Platform Agnostic**: Works with Langfuse, OpenTelemetry, Jaeger, and console

## Backward Compatibility

All changes are backward compatible:
- `agent_name` parameter is optional (defaults to "default")
- Existing code without `agent_name` will continue to work
- No breaking changes to existing APIs

## Testing

Run the example to test:
```bash
python example_hierarchical_tracing.py
```

View traces in:
- **Langfuse**: Check your Langfuse dashboard for traces named `agent.*`, `llm.<model_name>`, `tool.*`
- **Console**: See hierarchical names in console output
- **Jaeger/OTLP**: View span names in trace timeline

## Next Steps

Potential future enhancements:
1. Add support for custom tool display names at tool definition time
2. Add trace filtering utilities based on hierarchical names
3. Add metrics aggregation by agent/tool name
4. Add trace visualization tools for hierarchical names
