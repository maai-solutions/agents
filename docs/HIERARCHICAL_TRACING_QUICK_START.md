# Hierarchical Tracing - Quick Start Guide

## What is Hierarchical Tracing?

Hierarchical tracing uses a dot-notation naming convention to organize traces:
- `agent.<name>` - Agent execution traces
- `llm.<name>` - LLM call traces
- `tool.<name>` - Tool execution traces

## Quick Start

### 1. Basic Usage

```python
import asyncio
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools

async def main():
    # Create agent with custom name
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        agent_name="calculator",  # 👈 Set agent name here
        use_async=True
    )

    response = await agent.run("Calculate 42 * 17")
    print(response.result)

asyncio.run(main())
```

**Generated traces:**
```
agent.calculator
├── llm.gemma3:27b
├── llm.gemma3:27b
└── tool.calculator
```

### 2. With Telemetry

```python
import asyncio
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

async def main():
    # Initialize telemetry with Langfuse
    tracer = initialize_telemetry(
        service_name="my-app",
        exporter_type="langfuse",
        enabled=True
    )

    # Create agent
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        agent_name="search",  # 👈 Traces: agent.search
        tracer=tracer,
        use_async=True
    )

    response = await agent.run("Search for Python tutorials")

    # Flush traces to Langfuse
    tracer.flush()

asyncio.run(main())
```

### 3. Multiple Agents

```python
import asyncio
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

async def main():
    tracer = initialize_telemetry(
        service_name="multi-agent",
        exporter_type="langfuse",
        enabled=True
    )

    # Create different agents with unique names
    calc_agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        agent_name="calculator",  # agent.calculator
        tracer=tracer,
        use_async=True
    )

    search_agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        agent_name="search",  # agent.search
        tracer=tracer,
        use_async=True
    )

    # Run both agents
    calc_result = await calc_agent.run("What is 100 + 200?")
    search_result = await search_agent.run("Search for AI news")

    tracer.flush()

asyncio.run(main())
```

## Trace Hierarchy

Each agent run creates a hierarchy of traces:

```
agent.<your_agent_name>           ← Top-level agent execution
├── llm.<model_name>              ← LLM call (e.g., llm.gemma3:27b)
├── llm.<model_name>              ← LLM call (e.g., llm.gemma3:27b)
├── tool.<tool_name>              ← Tool execution
└── llm.<model_name>              ← LLM call (e.g., llm.gemma3:27b)
```

## Automatic Naming

### LLM Traces (automatic)
The model name is automatically used for LLM traces:
- `llm.gemma3:27b` - Traces for Gemma3 27B model
- `llm.gpt-4` - Traces for GPT-4 model
- `llm.llama3:70b` - Traces for Llama3 70B model

This allows you to easily identify and compare traces from different models.

### Tool Traces (automatic)
- `tool.calculator` - Math calculations
- `tool.search` - Information search
- `tool.read_file` - File reading
- `tool.shell_command` - Shell commands
- `tool.api_request` - API calls

## Configuration Options

### Set Agent Name

```python
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    agent_name="my_agent",  # Option 1: Direct parameter
    use_async=True
)
```

### Set Default via Telemetry

```python
tracer = initialize_telemetry(
    service_name="my-service",
    exporter_type="langfuse",
    agent_name="default_name",  # Option 2: Default for all agents
    enabled=True
)
```

### With Session Grouping

```python
tracer = initialize_telemetry(
    service_name="my-app",
    exporter_type="langfuse",
    session_id="user-123-session-456",  # Group by session
    agent_name="assistant",
    enabled=True
)

agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    agent_name="assistant",
    tracer=tracer,
    session_id="user-123-session-456",
    use_async=True
)
```

## Best Practices

### ✅ Good Names
```python
agent_name="calculator"        # Clear purpose
agent_name="search"            # Simple and descriptive
agent_name="data_processor"    # snake_case
agent_name="api_client"        # Specific
```

### ❌ Bad Names
```python
agent_name="a1"               # Not descriptive
agent_name="test"             # Too generic
agent_name="DataProcessor"    # Use snake_case
agent_name="API-Client"       # Use underscores
```

## Viewing Traces

### Langfuse
1. Go to your Langfuse dashboard
2. Navigate to **Traces** view
3. Search for traces:
   - `agent.calculator` - Find calculator agent traces
   - `llm.gemma3:27b` - Find all Gemma3 27B model calls
   - `tool.*` - Find all tool executions

### Console
Set `exporter_type="console"` to see traces in terminal:
```bash
TRACE: agent.calculator
  SPAN: llm.gemma3:27b
  SPAN: llm.gemma3:27b
  SPAN: tool.calculator
```

### Jaeger
Set `exporter_type="jaeger"` and view in Jaeger UI:
- Traces appear as spans with hierarchical names
- Filter by span name to find specific components

## Environment Variables

```bash
# Enable telemetry
TELEMETRY_ENABLED=true

# Choose exporter
TELEMETRY_EXPORTER=langfuse  # or console, otlp, jaeger

# Langfuse (if using)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Run Example

```bash
# Run the hierarchical tracing example
python example_hierarchical_tracing.py
```

## Troubleshooting

### Traces show "agent.default"
✅ **Solution:** Set `agent_name` parameter:
```python
agent = Agent(
    ...
    agent_name="my_agent"  # Add this
)
```

### Traces not appearing
✅ **Solution:** Ensure telemetry is enabled and flushed:
```python
tracer = initialize_telemetry(
    ...
    enabled=True  # Check this
)

# After agent runs
tracer.flush()  # Add this
```

### Can't see hierarchical names
✅ **Solution:** Verify you're using a supported exporter:
- `langfuse` ✅
- `console` ✅
- `otlp` ✅
- `jaeger` ✅

## Learn More

- **Full Documentation**: [docs/HIERARCHICAL_TRACING.md](HIERARCHICAL_TRACING.md)
- **Working Example**: [example_hierarchical_tracing.py](../example_hierarchical_tracing.py)
- **Telemetry Guide**: [TELEMETRY.md](TELEMETRY.md)
- **Langfuse Integration**: [LANGFUSE_INTEGRATION.md](LANGFUSE_INTEGRATION.md)
