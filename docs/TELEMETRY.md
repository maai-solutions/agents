# Telemetry & Observability

## Overview

The ReasoningAgent includes comprehensive **telemetry and observability** to monitor and debug agent behavior, LLM interactions, and tool usage in production.

### Supported Backends

- **OpenTelemetry** - Industry-standard distributed tracing (Console, OTLP, Jaeger)
- **Langfuse** - LLM-specific observability platform (⭐ Recommended for LLM applications)

For detailed Langfuse setup and features, see [LANGFUSE_INTEGRATION.md](LANGFUSE_INTEGRATION.md).

### What Gets Traced

✅ **Agent Execution Flow**
- User input and agent type
- Iteration count and completion status
- Execution time and final results

✅ **LLM Calls**
- Reasoning phase inputs and outputs
- Tool argument generation
- Model name and parameters
- Token usage

✅ **Tool Executions**
- Tool name and description
- Input arguments (limited to 500 chars)
- Execution results (limited to 500 chars)
- Success/failure status
- Exceptions and errors

✅ **Performance Metrics**
- Total iterations
- Tool execution count
- Token usage (input/output)
- Execution time

## Installation

```bash
# Core OpenTelemetry packages
pip install opentelemetry-api opentelemetry-sdk

# Exporters (install what you need)
pip install opentelemetry-exporter-otlp        # For OTLP Collector
pip install opentelemetry-exporter-jaeger      # For Jaeger
pip install opentelemetry-instrumentation-fastapi  # For FastAPI auto-instrumentation
```

Or install all at once:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Console Exporter (Development)

Print traces to console for debugging:

```python
from linus.agents.agent import Agent, get_default_tools
from linus.agents.telemetry import initialize_telemetry

# Initialize telemetry
initialize_telemetry(
    service_name="my-agent",
    exporter_type="console",
    enabled=True
)

# Create and use agent
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools()
)

result = agent.run("Calculate 42 * 17")
# Traces will be printed to console
```

### 2. Jaeger Exporter (Production)

Send traces to Jaeger for visualization:

```python
from linus.agents.telemetry import initialize_telemetry

# Start Jaeger (in Docker)
# docker run -d -p 16686:16686 -p 6831:6831/udp jaegertracing/all-in-one

# Initialize telemetry with Jaeger
initialize_telemetry(
    service_name="reasoning-agent",
    exporter_type="jaeger",
    jaeger_endpoint="localhost",  # Jaeger agent host
    enabled=True
)

# Use agent normally - traces sent to Jaeger
# View at http://localhost:16686
```

### 3. OTLP Exporter (OpenTelemetry Collector)

Send traces to an OTLP collector:

```python
initialize_telemetry(
    service_name="reasoning-agent",
    exporter_type="otlp",
    otlp_endpoint="http://localhost:4317",
    enabled=True
)
```

## FastAPI Integration

### Environment Variables

Add to your `.env` file:

```bash
# Enable telemetry
TELEMETRY_ENABLED=true

# Choose exporter: console, jaeger, or otlp
TELEMETRY_EXPORTER=jaeger

# Jaeger configuration
TELEMETRY_JAEGER_ENDPOINT=localhost

# OTLP configuration
TELEMETRY_OTLP_ENDPOINT=http://localhost:4317
```

### Start the API

```bash
python src/app.py
```

The FastAPI app will automatically initialize telemetry on startup. All API requests will be traced!

## Trace Data

### Span Names

Traces are organized hierarchically with the following span names:

- `agent.reasoning` - Reasoning phase (analyzing task, planning actions)
- `agent.execute_task` - Task execution with tools
- `llm.{call_type}` - LLM calls (reasoning, tool_args, generate)
- `tool.{tool_name}` - Individual tool executions

### Span Attributes

Each span includes relevant attributes:

**Agent spans:**
- `agent.input` - User's input query (truncated to 500 chars)
- `agent.type` - Agent type (ReasoningAgent)
- `agent.max_iterations` - Maximum iterations allowed

**LLM spans:**
- `llm.model` - Model name (e.g., gemma3:27b)
- `llm.input` - Prompt sent to LLM (truncated)
- `llm.output` - LLM response (truncated)
- `llm.call_type` - Type of call (reasoning, tool_args, generate)

**Task/Tool spans:**
- `task.description` - Task description
- `task.tool_name` - Tool being used
- `tool.args` - Tool input arguments (truncated to 500 chars)
- `tool.result` - Tool execution result (truncated to 500 chars)

### Span Events

Key events are recorded:

- `tool_not_found` - When requested tool doesn't exist
- `tool_args_generation_failed` - When argument generation fails
- Exceptions are automatically recorded with full stack traces

## Configuration

### TelemetryConfig

```python
from linus.agents.telemetry import TelemetryConfig, setup_telemetry

config = TelemetryConfig(
    service_name="my-reasoning-agent",
    exporter_type="jaeger",           # console, otlp, or jaeger
    otlp_endpoint="http://localhost:4317",
    jaeger_endpoint="localhost",
    enabled=True
)

tracer = setup_telemetry(config)
```

### Programmatic Initialization

```python
from linus.agents.telemetry import initialize_telemetry

tracer = initialize_telemetry(
    service_name="my-agent",
    exporter_type="console",
    enabled=True
)
```

### Disable Telemetry

Set `enabled=False` or don't initialize telemetry at all. The agent will work normally without any tracing overhead.

```python
# Option 1: Don't initialize
# agent = Agent(...)  # No tracing

# Option 2: Explicitly disable
initialize_telemetry(enabled=False)
```

## Viewing Traces

### Jaeger UI

1. Start Jaeger:
   ```bash
   docker run -d -p 16686:16686 -p 6831:6831/udp jaegertracing/all-in-one
   ```

2. Run your agent with Jaeger exporter

3. Open Jaeger UI: http://localhost:16686

4. Select service name (e.g., "reasoning-agent")

5. Click "Find Traces"

You'll see:
- **Trace timeline** - Visual representation of execution flow
- **Span details** - Attributes, events, and durations
- **Dependencies** - Service dependencies graph
- **Error traces** - Spans marked with errors

### Console Output

When using console exporter, traces are printed in JSON format:

```json
{
    "name": "agent.reasoning",
    "context": {
        "trace_id": "0x...",
        "span_id": "0x...",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": null,
    "start_time": "2025-01-XX...",
    "end_time": "2025-01-XX...",
    "status": {
        "status_code": "OK"
    },
    "attributes": {
        "llm.input": "Calculate 42 * 17",
        "llm.model": "gemma3:27b",
        "llm.output": "{\"has_sufficient_info\": true, ...}"
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "service.name": "reasoning-agent"
        }
    }
}
```

## Advanced Usage

### Custom Span Attributes

Add custom attributes within traced methods:

```python
class MyCustomAgent(ReasoningAgent):
    def _reasoning_call(self, input_text: str):
        # Add custom telemetry
        self.tracer.set_attribute("custom.context", "my_value")
        self.tracer.add_event("custom_event", {"key": "value"})

        return super()._reasoning_call(input_text)
```

### Trace Custom Methods

Use the `@trace_method` decorator:

```python
from linus.agents.telemetry import trace_method

class MyAgent(ReasoningAgent):
    @trace_method("agent.custom_method", custom_attr="value")
    def my_custom_method(self, data: str) -> str:
        # Method body
        return "result"
```

### Manual Span Creation

For fine-grained control:

```python
from linus.agents.telemetry import get_tracer

tracer = get_tracer()

with tracer.trace_llm_call(
    prompt="My prompt",
    model="gemma3:27b",
    call_type="custom"
):
    # Your code here
    response = llm.generate(...)
    tracer.set_attribute("custom.response_length", len(response))
```

## Sampling and Performance

### Sampling

For high-volume production, configure sampling in the tracer provider:

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

# Sample 10% of traces
sampler = TraceIdRatioBased(0.1)

# Apply when creating tracer provider (in telemetry.py setup_telemetry)
```

### Performance Impact

OpenTelemetry overhead is minimal:
- **Enabled**: ~1-5% overhead
- **Disabled**: No overhead (check is very fast)

The agent automatically:
- Truncates large strings (inputs, outputs limited to 500 chars)
- Uses batch processing for span export
- Only traces when enabled

## Troubleshooting

### "OpenTelemetry not installed"

```bash
pip install opentelemetry-api opentelemetry-sdk
```

### Traces not appearing in Jaeger

1. Check Jaeger is running:
   ```bash
   docker ps | grep jaeger
   ```

2. Check endpoint configuration:
   ```python
   jaeger_endpoint="localhost"  # Should match Jaeger host
   ```

3. Check firewall/network:
   ```bash
   telnet localhost 6831
   ```

### OTLP connection refused

1. Check OTLP collector is running on port 4317

2. Verify endpoint URL:
   ```python
   otlp_endpoint="http://localhost:4317"  # Include http://
   ```

### No traces in console

1. Ensure telemetry is enabled:
   ```python
   enabled=True
   ```

2. Check logger level allows debug output

3. Verify agent is actually running (not just imported)

## Integration with Observability Platforms

### Grafana + Tempo

Use OTLP exporter with Tempo backend:

```yaml
# docker-compose.yml
services:
  tempo:
    image: grafana/tempo:latest
    ports:
      - "4317:4317"  # OTLP gRPC
```

```python
initialize_telemetry(
    exporter_type="otlp",
    otlp_endpoint="http://localhost:4317"
)
```

### Datadog

Use OTLP exporter with Datadog Agent:

```python
initialize_telemetry(
    exporter_type="otlp",
    otlp_endpoint="http://localhost:4318/v1/traces"  # Datadog OTLP endpoint
)
```

### New Relic

```python
initialize_telemetry(
    exporter_type="otlp",
    otlp_endpoint="https://otlp.nr-data.net:4317"
    # Set NEW_RELIC_API_KEY environment variable
)
```

## Best Practices

1. **Use different service names per environment**
   ```python
   service_name=f"reasoning-agent-{os.getenv('ENV', 'dev')}"
   ```

2. **Disable in unit tests**
   ```python
   if not os.getenv('CI'):
       initialize_telemetry(enabled=True)
   ```

3. **Sample in production**
   - Use 100% sampling in dev/staging
   - Use 10-50% in production (adjust based on volume)

4. **Monitor trace export errors**
   - Check logs for export failures
   - Set up alerts on exporter errors

5. **Add custom context**
   ```python
   self.tracer.set_attribute("user.id", user_id)
   self.tracer.set_attribute("request.id", request_id)
   ```

## Examples

See `examples/telemetry_example.py` for complete working examples:
- Console exporter
- Jaeger exporter
- OTLP exporter
- FastAPI integration

Run with:
```bash
python examples/telemetry_example.py
```

## API Reference

### Functions

- `initialize_telemetry(...)` - Initialize global telemetry
- `setup_telemetry(config)` - Setup with TelemetryConfig
- `get_tracer()` - Get global AgentTracer instance
- `is_telemetry_available()` - Check if OpenTelemetry is installed

### Classes

- `TelemetryConfig` - Configuration dataclass
- `AgentTracer` - Tracer wrapper for agent operations
- `@trace_method` - Decorator for tracing methods

### AgentTracer Methods

- `trace_agent_run(input, agent_type)` - Trace full agent execution
- `trace_reasoning_phase(input, iteration)` - Trace reasoning phase
- `trace_llm_call(prompt, model, call_type)` - Trace LLM call
- `trace_tool_execution(tool_name, args)` - Trace tool execution
- `set_attribute(key, value)` - Add span attribute
- `add_event(name, attributes)` - Add span event
- `record_exception(exception)` - Record exception in span
- `set_status(status_code, description)` - Set span status

---

**Added**: 2025-01-XX
**Status**: ✅ Production Ready
**Dependencies**: `opentelemetry-api`, `opentelemetry-sdk` (optional)
