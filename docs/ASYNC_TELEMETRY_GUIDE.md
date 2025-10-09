# Async Telemetry Usage Guide

## Quick Start

### Using Langfuse with Async Agent

```python
import asyncio
from linus.agents.agent import Agent
from linus.agents.telemetry import initialize_telemetry

async def main():
    # Initialize Langfuse tracer
    tracer = initialize_telemetry(
        service_name="my-agent",
        exporter_type="langfuse",
        session_id="user-session-123",
        enabled=True
    )

    # Create async agent
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        temperature=0.7,
        use_async=True  # ← Enable async mode
    )

    # Assign tracer to agent
    agent.tracer = tracer

    # Run query (use await!)
    response = await agent.run("What is 10 + 20?")

    # Flush traces
    tracer.flush()

    return response

# Run with asyncio
asyncio.run(main())
```

### Using OpenTelemetry with Async Agent

```python
import asyncio
from linus.agents.agent import Agent
from linus.agents.telemetry import initialize_telemetry

async def main():
    # Initialize console tracer (or otlp, jaeger)
    tracer = initialize_telemetry(
        service_name="my-agent",
        exporter_type="console",  # or "otlp", "jaeger"
        enabled=True
    )

    # Create async agent
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        use_async=True
    )

    agent.tracer = tracer

    # Run query
    response = await agent.run("Calculate 42 * 17")

    return response

asyncio.run(main())
```

## Environment Variables

### Langfuse

```bash
# Required
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...

# Optional
LANGFUSE_HOST=https://cloud.langfuse.com  # Default
TELEMETRY_ENABLED=true
TELEMETRY_EXPORTER=langfuse
```

### OpenTelemetry (OTLP)

```bash
TELEMETRY_ENABLED=true
TELEMETRY_EXPORTER=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### OpenTelemetry (Jaeger)

```bash
TELEMETRY_ENABLED=true
TELEMETRY_EXPORTER=jaeger
JAEGER_AGENT_HOST=localhost
```

## Manual Tracing

You can also manually create traces in your async code:

```python
async def my_function():
    # Get tracer from agent or initialize globally
    from linus.agents.telemetry import get_tracer

    tracer = get_tracer()

    # Trace a custom operation
    async with tracer.trace_agent_run("My operation", "CustomAgent") as trace:
        # Your async code here
        result = await some_async_operation()

        # Add custom attributes
        tracer.set_attribute("custom_key", "custom_value")

        # Add events
        tracer.add_event("operation_completed", {
            "result_size": len(result)
        })

    return result
```

## Nested Tracing

Create nested spans for detailed observability:

```python
async def process_query(query: str):
    tracer = get_tracer()

    # Main trace
    async with tracer.trace_agent_run(query, "ProcessingAgent") as trace:

        # Reasoning phase
        async with tracer.trace_reasoning_phase(query, iteration=1) as reasoning:
            plan = await create_plan(query)

        # LLM call
        async with tracer.trace_llm_call(plan, "gemma3:27b", "generation") as llm:
            response = await llm.generate(plan)

        # Tool execution
        async with tracer.trace_tool_execution("calculator", {"expr": "2+2"}) as tool:
            result = await execute_tool(tool)

    return result
```

## Error Handling

Exceptions are automatically recorded in traces:

```python
async def risky_operation():
    tracer = get_tracer()

    async with tracer.trace_agent_run("Risky op", "Agent") as trace:
        try:
            result = await might_fail()
        except Exception as e:
            # Exception is automatically recorded
            tracer.record_exception(e)
            tracer.set_status("ERROR", str(e))
            raise

    return result
```

## Best Practices

### 1. Always Flush Traces

```python
# At the end of your async function
async def main():
    tracer = initialize_telemetry(...)
    agent = create_gemma_agent(...)

    try:
        response = await agent.run(query)
    finally:
        # Ensure traces are sent
        tracer.flush()
```

### 2. Use Session IDs

Group related traces together:

```python
tracer = initialize_telemetry(
    service_name="my-agent",
    exporter_type="langfuse",
    session_id=f"user-{user_id}-session-{session_id}"
)
```

### 3. Add Meaningful Metadata

```python
async with tracer.trace_agent_run(query, "Agent") as trace:
    # Add context
    tracer.set_attribute("user_id", user_id)
    tracer.set_attribute("query_type", "calculation")

    result = await process()
```

### 4. Record Metrics

```python
# After agent execution
metrics = {
    "execution_time": 2.5,
    "tokens_used": 1500,
    "iterations": 3
}
tracer.record_metrics(metrics)
```

## Troubleshooting

### Traces Not Appearing

1. **Check credentials**:
   ```python
   import os
   print(os.getenv("LANGFUSE_PUBLIC_KEY"))  # Should not be None
   ```

2. **Verify tracer is enabled**:
   ```python
   print(f"Tracer enabled: {tracer.enabled}")
   ```

3. **Call flush()**:
   ```python
   tracer.flush()  # Don't forget!
   ```

### Context Manager Errors

If you see errors about context managers:

```python
# ❌ Wrong - using regular with
with tracer.trace_agent_run(...):
    await agent.run()

# ✅ Correct - using async with
async with tracer.trace_agent_run(...):
    await agent.run()
```

### Import Errors

```python
# If Langfuse is not installed
pip install langfuse

# If OpenTelemetry is not installed
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

## Complete Example

```python
import asyncio
import os
from linus.agents.agent import Agent
from linus.agents.telemetry import initialize_telemetry

async def main():
    # Setup
    tracer = initialize_telemetry(
        service_name="demo-agent",
        exporter_type="langfuse",
        session_id="demo-session",
        enabled=True
    )

    agent = Agent(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        temperature=0.7,
        use_async=True
    )

    agent.tracer = tracer

    # Execute multiple queries
    queries = [
        "What is 10 + 20?",
        "Calculate 5 * 7",
        "Search for Python documentation"
    ]

    results = []
    for query in queries:
        try:
            print(f"Processing: {query}")
            response = await agent.run(query, return_metrics=True)

            print(f"Result: {response.result}")
            print(f"Tokens: {response.metrics.total_tokens}")

            results.append(response)

        except Exception as e:
            print(f"Error: {e}")

    # Cleanup
    tracer.flush()

    print(f"\nProcessed {len(results)} queries successfully!")
    return results

if __name__ == "__main__":
    asyncio.run(main())
```

## API Reference

### LangfuseTracer

- `trace_agent_run(user_input, agent_type)` - Main agent trace
- `trace_reasoning_phase(input_text, iteration)` - Reasoning span
- `trace_llm_call(prompt, model, call_type)` - LLM generation
- `trace_tool_execution(tool_name, tool_args)` - Tool execution
- `add_event(name, attributes)` - Add event
- `set_attribute(key, value)` - Set attribute
- `record_exception(exception)` - Record exception
- `record_metrics(metrics)` - Record metrics dict
- `flush()` - Send traces to Langfuse

### AgentTracer (OpenTelemetry)

- `trace_agent_run(user_input, agent_type)` - Main agent span
- `trace_reasoning_phase(input_text, iteration)` - Reasoning span
- `trace_llm_call(prompt, model, call_type)` - LLM call span
- `trace_tool_execution(tool_name, tool_args)` - Tool execution span
- `add_event(name, attributes)` - Add event to span
- `set_attribute(key, value)` - Set span attribute
- `record_exception(exception)` - Record exception
- `record_metrics(metrics)` - Record metrics as attributes

All methods return async context managers compatible with `async with`.
