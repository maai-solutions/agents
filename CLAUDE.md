# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ReasoningAgent framework for LLMs without native tool support. The agent uses OpenAI-compatible APIs (Ollama, OpenAI, etc.) and implements a two-phase approach:
1. **Reasoning Phase**: Analyzes the task and plans which tools to use
2. **Execution Phase**: Generates tool arguments and executes the planned tasks

The framework supports any OpenAI-compatible endpoint, including:
- **Ollama**: Local models like Gemma3:27b, Llama, etc.
- **OpenAI**: GPT-4, GPT-3.5-turbo, etc.
- **Other providers**: Any service implementing the OpenAI API spec

## Running the Project

### Prerequisites
- Ollama running locally with Gemma3:27b model
- Python 3.12+
- Dependencies: `pip install -r requirements.txt`

### Development Commands

**Start the FastAPI server:**
```bash
cd /Users/udg/Projects/ai/agents
python src/app.py
# or
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

**Run the interactive demo:**
```bash
python example_usage.py
```

**Test reasoning phase only:**
```bash
python example_usage.py --test-reasoning
```

**Test the API:**
```bash
# Ensure the API is running first
python test_api.py
```

**Test rich logging features:**
```bash
python example_rich_logging.py
```

### Configuration
Environment variables are configured in `.env`:

**LLM Configuration:**
- `LLM_API_BASE`: OpenAI-compatible API endpoint (default: `http://localhost:11434/v1`)
- `LLM_MODEL`: Model name (default: `gemma3:27b`)
- `LLM_API_KEY`: API key for authentication (default: `not-needed` for Ollama)
- `LLM_TEMPERATURE`: Temperature for responses (default: `0.7`)
- `LLM_MAX_TOKENS`: Maximum tokens to generate (optional)
- `LLM_TOP_P`: Nucleus sampling parameter (optional)
- `LLM_TOP_K`: Top-k sampling parameter (optional, Ollama-specific)
- `AGENT_VERBOSE`: Enable debug logging (default: `true`)

**Observability Configuration (Langfuse is the default):**
- `TELEMETRY_ENABLED`: Enable/disable telemetry (default: `true`)
- `TELEMETRY_EXPORTER`: Exporter type - `langfuse`, `console`, `otlp`, `jaeger` (default: `langfuse`)
- `LANGFUSE_PUBLIC_KEY`: Langfuse public API key (required for Langfuse)
- `LANGFUSE_SECRET_KEY`: Langfuse secret API key (required for Langfuse)
- `LANGFUSE_HOST`: Langfuse host URL (default: `https://cloud.langfuse.com`)
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OpenTelemetry OTLP endpoint (default: `http://localhost:4317`)
- `JAEGER_AGENT_HOST`: Jaeger agent host (default: `localhost`)

## Architecture

### Core Components

**Agent Implementation** (`src/linus/agents/agent/agent.py`):
- `Agent`: Base class with OpenAI client integration, input/output validation, and state management
- `ReasoningAgent`: Two-phase agent implementation for LLMs without native tool support
  - Uses `openai.OpenAI` or `openai.AsyncOpenAI` for LLM calls (no LangChain dependency)
  - `_reasoning_call()`: Analyzes user request and plans tasks (returns JSON with reasoning and task list)
  - `_execute_task_with_tool()`: Executes individual tasks using tools
  - `_generate_tool_arguments()`: Uses LLM to generate JSON arguments for tool calls
  - `_format_final_response()`: Combines multi-step results into coherent response
  - `_get_generation_kwargs()`: Builds generation parameters (temperature, max_tokens, top_p, top_k)

**Agent Factory** (`src/linus/agents/agent/factory.py`):
- `Agent()`: Factory function to create configured ReasoningAgent instances
  - Supports both sync (`OpenAI`) and async (`AsyncOpenAI`) clients
  - Configurable memory management (in-memory or vector store)
  - Built-in telemetry/tracing support (Langfuse or OpenTelemetry)

**Tools** (`src/linus/agents/agent/tools.py`):
- `SearchTool`: Mock search implementation
- `CalculatorTool`: Math expression evaluation (uses `eval` - replace in production)
- `FileReaderTool`: File reading
- `ShellCommandTool`: Shell command execution
- `APIRequestTool`: HTTP API requests
- `get_default_tools()`: Returns list of available tools
- `create_custom_tool()`: Utility to create custom tools from functions

**Logging** (`src/linus/agents/logging_config.py`):
- `setup_rich_logging()`: Configure rich logging with beautiful console output
- `log_with_panel()`: Display messages in bordered panels
- `log_with_table()`: Display data in formatted tables
- `log_metrics()`: Display metrics in a table format
- `log_tree()`: Display nested data as a tree structure
- Rich integration provides colored output, syntax highlighting, and enhanced tracebacks
- See [docs/RICH_LOGGING.md](docs/RICH_LOGGING.md) for detailed guide

**Telemetry** (`src/linus/agents/telemetry.py`):
- `initialize_telemetry()`: Setup observability (OpenTelemetry or Langfuse)
- `AgentTracer`: OpenTelemetry wrapper for agent operations
- `LangfuseTracer`: Langfuse wrapper for LLM observability
- Supports: Console, OTLP, Jaeger, and Langfuse exporters
- Traces: Agent runs, reasoning phases, LLM calls, tool executions
- See [docs/TELEMETRY.md](docs/TELEMETRY.md) and [docs/LANGFUSE_INTEGRATION.md](docs/LANGFUSE_INTEGRATION.md)

**FastAPI Application** (`src/app.py`):
- `/agent/query`: Main endpoint for agent queries
- `/agent/reasoning`: Test reasoning phase only
- `/tools/test`: Test individual tools
- `/tools`: List available tools
- `/agent/batch`: Process multiple queries
- `/history`: Conversation history management

### Key Design Patterns

**Two-Phase Execution:**
The agent cannot use tools directly, so it:
1. First calls LLM to reason about the task and output structured JSON with planned tasks
2. For each task requiring a tool, calls LLM again to generate tool arguments as JSON
3. Executes tools with generated arguments
4. Aggregates results and formats final response

**Input/Output Schema Support:**
Agents support optional Pydantic models for:
- `input_schema`: Validates incoming requests
- `output_schema`: Structures responses
- `output_key`: Saves results to shared state dictionary for multi-agent workflows

**Shared State:**
Agents can share a state dictionary to pass context between multiple agents or across multiple calls.

### Debugging

Comprehensive debug logging is implemented with prefixed tags:
- `[RUN]`: Main execution flow
- `[REASONING]`: Reasoning phase (prompts, responses, parsed results)
- `[EXECUTION]`: Task execution flow
- `[TOOL_ARGS]`: Tool argument generation
- `[GENERATE]`: Direct LLM responses without tools
- `[FINAL]`: Final response formatting

Filter logs with: `grep "[REASONING]" agent_api.log`

Configure loguru level in your code or use `AGENT_VERBOSE=true` in `.env`.

## Key Implementation Details

**JSON Extraction:**
The agent uses regex to extract JSON from LLM responses since models may include extra text:
```python
json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
```

**Tool Execution:**
Tools are stored in `agent.tool_map` (dict keyed by tool name). The agent:
1. Gets tool schema from `tool.args_schema.model_json_schema()` (Pydantic v2)
2. Prompts LLM to generate arguments matching the schema
3. Executes: `tool.arun(tool_args)` (async) or `tool.run(tool_args)` (sync)

**API Async Handling:**
FastAPI endpoints run synchronous agent code using `loop.run_in_executor()` to avoid blocking.

**OpenAI Client Usage:**
The agent uses the native OpenAI Python client instead of LangChain:
- Direct calls to `client.chat.completions.create()`
- Message format: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`
- Response format: `response.choices[0].message.content`
- Token tracking: `response.usage.prompt_tokens`, `response.usage.completion_tokens`

## Module Import Path

When importing from the project:
```python
from linus.agents.agent.factory import Agent
from linus.agents.agent.reasoning_agent import ReasoningAgent
from linus.agents.agent.tools import get_default_tools, create_custom_tool
```

Note: The `src/` directory must be in PYTHONPATH or use relative imports from `src/`.

## Creating an Agent

### Basic Usage (Ollama)
```python
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
import asyncio

# Create agent for local Ollama (async by default)
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
    top_k=40,  # Ollama-specific
    tools=get_default_tools(),
    verbose=True,
    use_async=True  # Use AsyncOpenAI client
)

# Run the agent (async)
async def main():
    response = await agent.run("What is 42 * 17?")
    print(response.result)  # AgentResponse with result, metrics, execution_history

asyncio.run(main())
```

### OpenAI API Usage
```python
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools

agent = Agent(
    api_base="https://api.openai.com/v1",
    model="gpt-4",
    api_key="sk-...",  # Your OpenAI API key
    temperature=0.5,
    max_tokens=1000,
    top_p=0.9,
    tools=get_default_tools(),
    verbose=True,
    use_async=True
)
```

### Synchronous Usage
```python
# Create synchronous agent (not recommended for production)
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    use_async=False,  # Use sync OpenAI client
    temperature=0.7
)

# Note: ReasoningAgent.run() is async, so you still need to await it
response = await agent.run("Calculate 100 + 200")
```

### Generation Parameters
- **`temperature`** (float, 0.0-2.0): Controls randomness. Higher = more creative/random
- **`max_tokens`** (int, optional): Maximum tokens to generate in completion
- **`top_p`** (float, 0.0-1.0): Nucleus sampling. Alternative to temperature
- **`top_k`** (int, optional): Top-k sampling. Only supported by Ollama (passed via `extra_body`)

### Response Format
`agent.run()` returns an `AgentResponse` object with:
```python
{
    "result": "The final answer",  # String or Pydantic model
    "metrics": {
        "total_iterations": 1,
        "total_tokens": 1500,
        "execution_time_seconds": 2.5,
        "llm_calls": 3,
        "tool_executions": 2,
        "successful_tool_calls": 2,
        "failed_tool_calls": 0,
        "task_completed": true,
        "avg_tokens_per_llm_call": 500.0,
        "success_rate": 1.0
    },
    "execution_history": [
        {
            "iteration": 1,
            "task": "Calculate the result",
            "tool": "calculator",
            "result": "714",
            "status": "completed"
        }
    ],
    "completion_status": {
        "is_complete": true,
        "reasoning": "Task completed successfully"
    }
}
```

To get just the result string: `agent.run(query, return_metrics=False)`

### Using Langfuse for Observability

Langfuse provides LLM-specific observability with automatic tracking of prompts, completions, tokens, and costs:

```python
import asyncio
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

# Initialize Langfuse tracer
tracer = initialize_telemetry(
    service_name="my-agent",
    exporter_type="langfuse",  # Use Langfuse
    enabled=True
)

# Create agent with Langfuse tracing
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    tracer=tracer,  # Pass tracer to agent
    use_async=True
)

# Run queries - automatically traced in Langfuse
async def main():
    response = await agent.run("What is 42 * 17?")
    print(response.result)

    # Flush traces before exit
    tracer.flush()

asyncio.run(main())
```

**With Session ID for monitoring sessions:**
```python
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools

# Create agent with session_id for Langfuse session grouping
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    session_id="user-123-session-456",  # Groups traces by session in Langfuse
    use_async=True
)

# All traces from this agent will be grouped under the same session
async def main():
    response = await agent.run("What is 42 * 17?")
    print(response.result)

asyncio.run(main())
```

**Environment variables required:**
- `LANGFUSE_PUBLIC_KEY`: Your Langfuse public API key
- `LANGFUSE_SECRET_KEY`: Your Langfuse secret API key
- `LANGFUSE_HOST`: Langfuse host (default: `https://cloud.langfuse.com`)

See [docs/LANGFUSE_INTEGRATION.md](docs/LANGFUSE_INTEGRATION.md) for detailed setup and usage.
