# Langfuse Integration

This document describes how to use Langfuse for observability and tracing in the ReasoningAgent framework.

## Overview

Langfuse is an open-source LLM engineering platform that provides comprehensive observability for LLM applications. The integration allows you to:

- **Trace agent execution flows** - See the complete execution path from user query to final response
- **Monitor LLM calls** - Track prompts, completions, token usage, and latency
- **Analyze tool executions** - Understand which tools are being used and with what arguments
- **Debug reasoning phases** - Inspect the agent's reasoning process step-by-step
- **Track performance metrics** - Monitor execution time, token usage, and success rates

## Installation

Install Langfuse along with the other dependencies:

```bash
pip install langfuse
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Get Langfuse Credentials

1. Sign up for a Langfuse account at [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a new project
3. Go to Settings → API Keys
4. Copy your Public Key and Secret Key

### 2. Configure Environment Variables

Add your Langfuse credentials to your `.env` file:

```env
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # Optional, defaults to cloud.langfuse.com
```

### 3. Initialize Telemetry

Initialize the telemetry system with Langfuse:

```python
from linus.agents.telemetry import initialize_telemetry

# Initialize Langfuse tracer
tracer = initialize_telemetry(
    service_name="my-reasoning-agent",
    exporter_type="langfuse",  # Use Langfuse instead of OpenTelemetry
    enabled=True
)
```

The system will automatically read credentials from environment variables.

### 4. Use with Agent

Pass the tracer to your agent:

```python
from linus.agents.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools

agent = create_gemma_agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    tracer=tracer  # Pass the Langfuse tracer
)

# Run queries - traces will be sent to Langfuse automatically
response = agent.run("What is 42 * 17?")
```

## Features

### Automatic Trace Creation

The Langfuse integration automatically creates traces for:

1. **Agent Runs**: Top-level trace for each `agent.run()` call
   - Captures user input
   - Tracks agent type
   - Records final output
   - Measures total execution time

2. **Reasoning Phases**: Nested spans for each reasoning iteration
   - Captures reasoning input
   - Tracks iteration number
   - Records reasoning output (task plan)

3. **LLM Calls**: Detailed generation tracking for each LLM invocation
   - Captures prompts
   - Records completions
   - Tracks token usage (if available)
   - Measures latency
   - Categorizes by call type (reasoning, tool_args, generate)

4. **Tool Executions**: Spans for each tool execution
   - Captures tool name
   - Records input arguments
   - Tracks output/results
   - Captures errors

### Trace Structure

A typical agent execution creates a hierarchical trace:

```
agent_run (Trace)
├── reasoning_phase (Span)
│   └── llm_reasoning (Generation)
├── tool_calculator (Span)
│   └── llm_tool_args (Generation)
└── llm_generate (Generation)
```

### Error Handling

Errors are automatically captured and recorded:

```python
try:
    response = agent.run("invalid query")
except Exception as e:
    # Exception is automatically recorded in the trace
    pass
```

### Flushing Traces

Traces are sent asynchronously. To ensure all traces are sent before program exit:

```python
# Flush pending traces
tracer.flush()
```

## Dashboard Features

Once traces are sent to Langfuse, you can use the dashboard to:

1. **View Traces**: See all agent executions with hierarchical structure
2. **Analyze Performance**: Track latency, token usage, and costs
3. **Debug Issues**: Inspect failed executions and errors
4. **Monitor Usage**: Understand LLM usage patterns
5. **Create Datasets**: Build test datasets from production traces
6. **Evaluate Outputs**: Score and evaluate agent responses

## Example: Complete Workflow

```python
import os
from dotenv import load_dotenv
from linus.agents.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

# Load environment variables
load_dotenv()

# Initialize Langfuse
tracer = initialize_telemetry(
    service_name="reasoning-agent",
    exporter_type="langfuse",
    enabled=True
)

# Create agent
agent = create_gemma_agent(
    api_base=os.getenv("LLM_API_BASE"),
    model=os.getenv("LLM_MODEL"),
    tools=get_default_tools(),
    tracer=tracer
)

# Run queries
queries = [
    "What is 42 * 17?",
    "Search for Python async programming",
]

for query in queries:
    print(f"Query: {query}")
    response = agent.run(query)
    print(f"Result: {response.result}\n")

# Flush traces
tracer.flush()
print("View traces at: https://cloud.langfuse.com")
```

## Self-Hosted Langfuse

If you're running a self-hosted Langfuse instance:

```python
tracer = initialize_telemetry(
    service_name="reasoning-agent",
    exporter_type="langfuse",
    langfuse_host="http://localhost:3000",  # Your self-hosted URL
    enabled=True
)
```

Or set in `.env`:

```env
LANGFUSE_HOST=http://localhost:3000
```

## Comparison: Langfuse vs OpenTelemetry

| Feature | Langfuse | OpenTelemetry |
|---------|----------|---------------|
| LLM-specific | ✅ Yes | ❌ No |
| Token tracking | ✅ Built-in | ⚠️ Custom attributes |
| Cost calculation | ✅ Automatic | ❌ Manual |
| Prompt management | ✅ Yes | ❌ No |
| Dataset creation | ✅ Yes | ❌ No |
| Generic tracing | ⚠️ Limited | ✅ Comprehensive |
| Self-hosted | ✅ Yes | ✅ Yes |
| Cloud offering | ✅ Yes | ⚠️ Multiple vendors |

**Use Langfuse when:**
- You need LLM-specific observability
- You want to track costs and tokens
- You need prompt versioning
- You want to build evaluation datasets

**Use OpenTelemetry when:**
- You need generic distributed tracing
- You're integrating with existing observability stack
- You need vendor-neutral tracing

## Troubleshooting

### Traces Not Appearing

1. **Check credentials**: Ensure `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set correctly
2. **Check connectivity**: Verify you can reach the Langfuse host
3. **Flush traces**: Call `tracer.flush()` before program exit
4. **Check logs**: Look for `[TELEMETRY]` log messages

### Import Errors

If you see "Langfuse not installed" warnings:

```bash
pip install langfuse
```

### Authentication Errors

Verify your API keys are correct:
1. Go to Langfuse dashboard → Settings → API Keys
2. Generate new keys if needed
3. Update your `.env` file

## Advanced Usage

### Custom Metadata

Add custom metadata to traces:

```python
# During execution, metadata is automatically added
# You can extend the agent to add custom metadata
tracer.set_attribute("user_id", "user123")
tracer.set_attribute("session_id", "session456")
```

### Manual Events

Add custom events to traces:

```python
tracer.add_event("custom_event", {
    "key": "value",
    "timestamp": time.time()
})
```

### Conditional Tracing

Enable tracing only in production:

```python
tracer = initialize_telemetry(
    exporter_type="langfuse",
    enabled=os.getenv("ENVIRONMENT") == "production"
)
```

## Resources

- **Langfuse Documentation**: [https://langfuse.com/docs](https://langfuse.com/docs)
- **Langfuse GitHub**: [https://github.com/langfuse/langfuse](https://github.com/langfuse/langfuse)
- **API Reference**: [https://langfuse.com/docs/sdk/python](https://langfuse.com/docs/sdk/python)
- **Self-Hosting Guide**: [https://langfuse.com/docs/deployment/self-host](https://langfuse.com/docs/deployment/self-host)

## See Also

- [TELEMETRY.md](TELEMETRY.md) - General telemetry documentation
- [TELEMETRY_IMPLEMENTATION.md](TELEMETRY_IMPLEMENTATION.md) - Implementation details
- [examples/langfuse_example.py](../examples/langfuse_example.py) - Example usage
