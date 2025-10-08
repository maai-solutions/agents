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

### Configuration
Environment variables are configured in `.env`:
- `LLM_API_BASE`: OpenAI-compatible API endpoint (default: `http://localhost:11434/v1`)
- `LLM_MODEL`: Model name (default: `gemma3:27b`)
- `LLM_API_KEY`: API key for authentication (default: `not-needed` for Ollama)
- `LLM_TEMPERATURE`: Temperature for responses (default: `0.7`)
- `LLM_MAX_TOKENS`: Maximum tokens to generate (optional)
- `LLM_TOP_P`: Nucleus sampling parameter (optional)
- `LLM_TOP_K`: Top-k sampling parameter (optional, Ollama-specific)
- `AGENT_VERBOSE`: Enable debug logging (default: `true`)

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
- `create_gemma_agent()`: Factory function to create configured ReasoningAgent instances

**Tools** (`src/linus/agents/agent/tools.py`):
- `SearchTool`: Mock search implementation
- `CalculatorTool`: Math expression evaluation (uses `eval` - replace in production)
- `FileReaderTool`: File reading
- `ShellCommandTool`: Shell command execution
- `APIRequestTool`: HTTP API requests
- `get_default_tools()`: Returns list of available tools
- `create_custom_tool()`: Utility to create custom tools from functions

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
1. Gets tool schema from `tool.args_schema.schema()`
2. Prompts LLM to generate arguments matching the schema
3. Executes: `tool.run(tool_args)`

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
from linus.agents.agent.agent import ReasoningAgent, create_gemma_agent
from linus.agents.agent.tools import get_default_tools, create_custom_tool
```

Note: The `src/` directory must be in PYTHONPATH or use relative imports from `src/`.

## Creating an Agent

### Basic Usage (Ollama)
```python
from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools

# Create agent for local Ollama
agent = create_gemma_agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
    top_k=40,  # Ollama-specific
    tools=get_default_tools(),
    verbose=True
)

# Run the agent
response = agent.run("What is 42 * 17?")
print(response.result)  # AgentResponse with result, metrics, execution_history
```

### OpenAI API Usage
```python
agent = create_gemma_agent(
    api_base="https://api.openai.com/v1",
    model="gpt-4",
    api_key="sk-...",  # Your OpenAI API key
    temperature=0.5,
    max_tokens=1000,
    top_p=0.9,
    tools=get_default_tools(),
    verbose=True
)
```

### Async Usage
```python
# Create async agent
agent = create_gemma_agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    use_async=True,  # Use AsyncOpenAI client
    temperature=0.7
)

# Use async run method
response = await agent.arun("Calculate 100 + 200")
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
