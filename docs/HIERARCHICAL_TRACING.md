# Hierarchical Tracing

This guide explains how to use hierarchical naming for traces in the agent framework. Hierarchical naming provides a clear, organized structure for observability by using a dot-notation naming convention.

## Overview

The framework supports three types of hierarchical trace names:

- **`agent.<name>`** - For agent execution traces
- **`llm.<name>`** - For LLM call traces
- **`tool.<name>`** - For tool execution traces

This naming convention makes it easy to:
- Filter and search traces by component type
- Identify which agent or tool generated a trace
- Organize traces in observability platforms like Langfuse, Jaeger, or OTLP collectors

## Naming Convention

### Agent Traces: `agent.<name>`

Agent traces represent the top-level execution of an agent. The name is set when creating the agent:

```python
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools

# Create an agent with a custom name
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    agent_name="calculator",  # Creates traces named "agent.calculator"
    use_async=True
)
```

**Default behavior:** If no `agent_name` is provided, traces use `"agent.default"`

### LLM Traces: `llm.<name>`

LLM traces represent calls to the language model. The framework automatically creates these using the model name:

- **`llm.gemma3:27b`** - Traces for Gemma3 27B model
- **`llm.gpt-4`** - Traces for GPT-4 model
- **`llm.llama3:70b`** - Traces for Llama3 70B model

The model name is automatically extracted from the agent configuration and used as the LLM trace name. This makes it easy to identify which model generated specific traces and compare performance across different models.

### Tool Traces: `tool.<name>`

Tool traces represent tool executions. The name is derived from the tool's name:

- **`tool.calculator`** - Calculator tool execution
- **`tool.search`** - Search tool execution
- **`tool.read_file`** - File reader tool execution
- **`tool.shell_command`** - Shell command tool execution

The tool name is automatically extracted from the tool being executed.

## Usage Examples

### Example 1: Single Agent with Custom Name

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
        agent_name="math_solver",  # Traces: agent.math_solver
        tracer=tracer,
        use_async=True
    )

    # Run task - creates hierarchical traces
    response = await agent.run("Calculate 42 * 17")
    print(response.result)

    # Flush traces
    tracer.flush()

asyncio.run(main())
```

**Traces generated:**
```
agent.math_solver
├── llm.gemma3:27b         (Planning phase)
├── llm.gemma3:27b         (Generating calculator arguments)
└── tool.calculator        (Executing calculation)
```

### Example 2: Multiple Agents with Different Names

```python
import asyncio
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

async def main():
    # Initialize shared telemetry
    tracer = initialize_telemetry(
        service_name="multi-agent-app",
        exporter_type="langfuse",
        enabled=True
    )

    # Create specialized agents with different names
    calculator_agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        agent_name="calculator",  # Traces: agent.calculator
        tracer=tracer,
        use_async=True
    )

    search_agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        agent_name="search",  # Traces: agent.search
        tracer=tracer,
        use_async=True
    )

    file_agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        agent_name="file_ops",  # Traces: agent.file_ops
        tracer=tracer,
        use_async=True
    )

    # Run tasks with different agents
    calc_result = await calculator_agent.run("What is 100 + 200?")
    search_result = await search_agent.run("Search for Python tutorials")
    file_result = await file_agent.run("Read README.md")

    # Flush traces
    tracer.flush()

asyncio.run(main())
```

**Traces generated:**
```
agent.calculator
├── llm.gemma3:27b
├── llm.gemma3:27b
└── tool.calculator

agent.search
├── llm.gemma3:27b
├── llm.gemma3:27b
└── tool.search

agent.file_ops
├── llm.gemma3:27b
├── llm.gemma3:27b
└── tool.read_file
```

### Example 3: Session Grouping with Hierarchical Names

Combine hierarchical naming with session IDs for even better organization:

```python
import asyncio
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

async def main():
    # Initialize telemetry with session ID
    tracer = initialize_telemetry(
        service_name="my-app",
        exporter_type="langfuse",
        session_id="user-123-session-456",  # Group by session
        enabled=True
    )

    # Create multiple agents in the same session
    agent1 = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        agent_name="assistant",
        tracer=tracer,
        session_id="user-123-session-456",  # Same session
        use_async=True
    )

    agent2 = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        agent_name="validator",
        tracer=tracer,
        session_id="user-123-session-456",  # Same session
        use_async=True
    )

    # Both agents' traces will be grouped by session
    result1 = await agent1.run("Calculate 50 * 2")
    result2 = await agent2.run("Validate the result")

    tracer.flush()

asyncio.run(main())
```

## Configuration

### Setting Agent Name

You can set the agent name in multiple ways:

#### 1. Via Agent Factory

```python
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    agent_name="my_agent",  # Set here
    use_async=True
)
```

#### 2. Via Telemetry Initialization

```python
tracer = initialize_telemetry(
    service_name="my-service",
    exporter_type="langfuse",
    agent_name="default_agent",  # Default for all agents using this tracer
    enabled=True
)
```

#### 3. Directly on Tracer

```python
tracer = initialize_telemetry(
    service_name="my-service",
    exporter_type="langfuse",
    enabled=True
)

# Update agent name on tracer
tracer.agent_name = "custom_agent"
```

### Best Practices for Naming

1. **Use descriptive names**: Choose names that clearly indicate the agent's purpose
   - ✅ `agent.calculator`, `agent.search`, `agent.file_ops`
   - ❌ `agent.a1`, `agent.test`, `agent.agent`

2. **Use snake_case**: Keep names lowercase with underscores
   - ✅ `agent.data_processor`, `agent.api_client`
   - ❌ `agent.DataProcessor`, `agent.API-Client`

3. **Be consistent**: Use the same naming pattern across your application
   - ✅ `agent.user_manager`, `agent.order_processor`, `agent.notification_sender`
   - ❌ `agent.UserMgr`, `agent.processOrders`, `agent.send-notifications`

4. **Keep it short**: Aim for concise but meaningful names
   - ✅ `agent.validator`, `agent.parser`, `agent.formatter`
   - ❌ `agent.data_validation_and_verification_system`

## Viewing Traces

### Langfuse Dashboard

When using Langfuse as your telemetry exporter, hierarchical traces appear in the dashboard with clear organization:

1. **Traces view**: Filter by trace name using the search bar
   - Search for `agent.calculator` to find all calculator agent traces
   - Search for `llm.gemma3:27b` to find all Gemma3 27B model calls
   - Search for `tool.*` to find all tool executions

2. **Sessions view**: Group traces by session ID and see hierarchical structure within each session

3. **Metrics view**: Analyze performance by agent name, LLM call type, or tool type

### Console Exporter

When using console exporter, traces are printed with hierarchical names:

```
TRACE: agent.calculator
  SPAN: llm.gemma3:27b
    - input: "Plan how to calculate 42 * 17"
    - output: {"tasks": [...]}
  SPAN: llm.gemma3:27b
    - input: "Generate args for calculator"
    - output: {"expression": "42 * 17"}
  SPAN: tool.calculator
    - input: {"expression": "42 * 17"}
    - output: "Result: 714"
```

### OTLP/Jaeger

When using OTLP or Jaeger, hierarchical names appear as span names in the trace timeline:

```
Timeline:
|-- agent.calculator (5.2s)
    |-- llm.gemma3:27b (2.1s)
    |-- llm.gemma3:27b (1.5s)
    |-- tool.calculator (0.1s)
    |-- llm.gemma3:27b (1.5s)  # Completion check
```

## Implementation Details

### Tracer Classes

Both `LangfuseTracer` and `AgentTracer` (OpenTelemetry) support hierarchical naming:

```python
# LangfuseTracer
tracer = LangfuseTracer(
    langfuse_client=client,
    session_id="session-123",
    agent_name="my_agent"  # Sets default agent name
)

# AgentTracer (OpenTelemetry)
tracer = AgentTracer(
    tracer=otel_tracer,
    agent_name="my_agent"  # Sets default agent name
)
```

### Trace Methods

All trace methods support hierarchical naming:

```python
# Agent trace with custom name
async with tracer.trace_agent_run(
    user_input="Calculate something",
    agent_type="ReasoningAgent",
    agent_name="calculator"  # Creates "agent.calculator"
):
    # Agent code here
    pass

# LLM trace with model name (automatic)
async with tracer.trace_llm_call(
    prompt="Generate tool args",
    model="gemma3:27b",  # Creates "llm.gemma3:27b"
    call_type="tool_args"
):
    # LLM call here
    pass

# Tool trace with custom name
async with tracer.trace_tool_execution(
    tool_name="calculator",
    tool_args={"expression": "1+1"},
    tool_display_name="math_calc"  # Creates "tool.math_calc"
):
    # Tool execution here
    pass
```

## Environment Variables

Configure hierarchical tracing using environment variables:

```bash
# Enable/disable telemetry
TELEMETRY_ENABLED=true

# Choose exporter type
TELEMETRY_EXPORTER=langfuse  # or console, otlp, jaeger

# Langfuse configuration
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# OTLP configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Jaeger configuration
JAEGER_AGENT_HOST=localhost
```

## Troubleshooting

### Traces Not Appearing with Custom Names

**Problem:** Traces appear with default names instead of custom names

**Solution:** Make sure you're passing `agent_name` when creating the agent:

```python
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    agent_name="my_agent",  # Make sure this is set
    use_async=True
)
```

### Tracer Not Using Agent Name

**Problem:** Tracer ignores the agent_name parameter

**Solution:** Ensure the tracer is properly initialized and passed to the agent:

```python
# Initialize tracer with agent name
tracer = initialize_telemetry(
    service_name="my-app",
    exporter_type="langfuse",
    agent_name="default",
    enabled=True
)

# Pass tracer to agent and override agent_name
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tracer=tracer,
    agent_name="custom",  # This overrides tracer's agent_name
    use_async=True
)
```

### Tools Not Showing Hierarchical Names

**Problem:** Tool traces don't use the `tool.<name>` format

**Solution:** This is handled automatically. Make sure:
1. Telemetry is enabled: `TELEMETRY_ENABLED=true`
2. Tools have valid names defined in their class
3. You're using a supported exporter (langfuse, console, otlp, jaeger)

## See Also

- [Telemetry Documentation](TELEMETRY.md) - General telemetry setup
- [Langfuse Integration](LANGFUSE_INTEGRATION.md) - Langfuse-specific features
- [Agent Factory](../src/linus/agents/agent/factory.py) - Agent creation
- [Example Code](../example_hierarchical_tracing.py) - Working example
