# Langfuse as Default Telemetry

Langfuse is now configured as the default telemetry system for the ReasoningAgent framework.

## Overview

Langfuse provides LLM-specific observability with automatic tracking of:
- Agent execution traces
- LLM calls (prompts, completions, tokens)
- Tool executions
- Performance metrics
- Cost tracking

## Configuration

### Environment Variables

The following environment variables are configured in `.env`:

```bash
# Telemetry Configuration
TELEMETRY_ENABLED=true
TELEMETRY_EXPORTER=langfuse

# Langfuse Credentials
LANGFUSE_PUBLIC_KEY=pk-lf-c53d2125-8f32-4d43-93cf-75c1cceb040a
LANGFUSE_SECRET_KEY=sk-lf-48bf57c7-47dc-4a69-baa4-99a79123b534
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Settings

Default values are set in `src/linus/settings/settings.py`:

```python
class Settings(BaseSettings):
    # Telemetry configuration
    telemetry_enabled: bool = True
    telemetry_exporter: str = "langfuse"  # Default exporter

    # Langfuse configuration
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
```

## Installation

Langfuse is included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install langfuse
```

## Usage

### Automatic Initialization

When you start the FastAPI application, Langfuse is automatically initialized:

```bash
python src/app.py
```

The app startup (`src/app.py`) automatically:
1. Loads settings from `.env`
2. Initializes Langfuse tracer
3. Passes tracer to the agent
4. Traces all agent operations

### Manual Initialization

```python
from linus.agents.telemetry import initialize_telemetry
from linus.agents.agent import Agent
from linus.settings import Settings

settings = Settings()

# Initialize Langfuse tracer
tracer = initialize_telemetry(
    service_name="my-agent",
    exporter_type="langfuse",  # or use settings.telemetry_exporter
    langfuse_public_key=settings.langfuse_public_key,
    langfuse_secret_key=settings.langfuse_secret_key,
    langfuse_host=settings.langfuse_host,
    enabled=True
)

# Create agent with tracer
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tracer=tracer  # Pass tracer to agent
)

# Run queries - automatically traced
response = agent.run("What is 42 * 17?")

# Flush traces before exit
tracer.flush()
```

## What Gets Traced

### Agent Runs
- User input
- Agent type
- Execution status
- Final output

### Reasoning Phase
- Input text
- Iteration number
- Reasoning output
- Planned tasks

### LLM Calls
- Prompt sent to LLM
- Model name
- Call type (reasoning, tool_args, generate)
- Response
- Token usage
- Latency

### Tool Executions
- Tool name
- Arguments
- Results
- Execution status

## Viewing Traces

1. Visit [Langfuse Cloud](https://cloud.langfuse.com)
2. Log in with your credentials
3. View traces for your project
4. Analyze:
   - Execution flow
   - Token usage
   - Costs
   - Performance metrics
   - Error rates

## Testing the Setup

Run the verification script:

```bash
python test_langfuse_setup.py
```

Expected output:

```
============================================================
LANGFUSE TELEMETRY CONFIGURATION TEST
============================================================

1. Langfuse Package:
   ✓ Langfuse available: True

2. Environment Configuration:
   ✓ Telemetry enabled: True
   ✓ Default exporter: langfuse
   ✓ Langfuse host: https://cloud.langfuse.com
   ✓ Public key configured: Yes
   ✓ Secret key configured: Yes

3. Tracer Initialization:
   ✓ Tracer type: LangfuseTracer
   ✓ Tracer enabled: True
   ✓ Client initialized: Yes

============================================================
✅ SUCCESS: Langfuse is configured as default telemetry!
============================================================
```

## Alternative Exporters

While Langfuse is the default, you can switch to other exporters by changing `TELEMETRY_EXPORTER`:

```bash
# Console output (for debugging)
TELEMETRY_EXPORTER=console

# OpenTelemetry OTLP
TELEMETRY_EXPORTER=otlp
TELEMETRY_OTLP_ENDPOINT=http://localhost:4317

# Jaeger
TELEMETRY_EXPORTER=jaeger
TELEMETRY_JAEGER_ENDPOINT=localhost

# Langfuse (default)
TELEMETRY_EXPORTER=langfuse
```

## Troubleshooting

### Langfuse Not Available

If you see the error "Langfuse not installed":

```bash
pip install langfuse
```

### Credentials Not Provided

Ensure your `.env` file contains:

```bash
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
```

### Tracer Not Enabled

Check that:
1. `TELEMETRY_ENABLED=true` in `.env`
2. `TELEMETRY_EXPORTER=langfuse` in `.env`
3. Credentials are set correctly
4. Langfuse package is installed

## Implementation Details

### Tracer Classes

- **`LangfuseTracer`**: Wrapper for Langfuse client (`src/linus/agents/telemetry.py`)
- **`AgentTracer`**: Wrapper for OpenTelemetry (alternative to Langfuse)

### Context Managers

The `LangfuseTracer` provides context managers for tracing:

```python
# Trace agent run
with tracer.trace_agent_run(user_input, agent_type):
    # Agent execution

# Trace reasoning phase
with tracer.trace_reasoning_phase(input_text, iteration):
    # Reasoning logic

# Trace LLM call
with tracer.trace_llm_call(prompt, model, call_type):
    # LLM call

# Trace tool execution
with tracer.trace_tool_execution(tool_name, tool_args):
    # Tool execution
```

### Integration Points

1. **Agent Factory** (`src/linus/agents/agent/factory.py`):
   - Accepts `tracer` parameter
   - Passes tracer to agent instance

2. **ReasoningAgent** (`src/linus/agents/agent/reasoning_agent.py`):
   - Uses tracer for all operations
   - Traces reasoning, LLM calls, tool executions

3. **FastAPI App** (`src/app.py`):
   - Initializes tracer on startup
   - Injects tracer into agent

## Benefits of Langfuse

1. **LLM-Specific**: Designed specifically for LLM observability
2. **Token Tracking**: Automatic tracking of token usage and costs
3. **Prompt Management**: View and analyze prompts across runs
4. **User-Friendly UI**: Clean, intuitive interface for viewing traces
5. **Cost Analysis**: Track costs per model, user, or session
6. **Performance Metrics**: Analyze latency and throughput
7. **Error Tracking**: Identify and debug errors quickly

## References

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse Python SDK](https://github.com/langfuse/langfuse-python)
- [Project Telemetry Guide](./TELEMETRY.md)
- [Langfuse Integration Guide](./LANGFUSE_INTEGRATION.md)
