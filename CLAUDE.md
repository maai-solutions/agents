# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ReasoningAgent framework for LLMs without native tool support (specifically Gemma3:27b via Ollama). The agent uses a two-phase approach:
1. **Reasoning Phase**: Analyzes the task and plans which tools to use
2. **Execution Phase**: Generates tool arguments and executes the planned tasks

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
- `LLM_API_BASE`: Ollama endpoint (default: `http://localhost:11434/v1`)
- `LLM_MODEL`: Model name (default: `gemma3:27b`)
- `LLM_TEMPERATURE`: Temperature for responses (default: `0.7`)
- `AGENT_VERBOSE`: Enable debug logging (default: `true`)

## Architecture

### Core Components

**Agent Implementation** (`src/linus/agents/agent/agent.py`):
- `Agent`: Base class with input/output validation and state management
- `ReasoningAgent`: Two-phase agent implementation for LLMs without native tool support
  - `_reasoning_call()`: Analyzes user request and plans tasks (returns JSON with reasoning and task list)
  - `_execute_task_with_tool()`: Executes individual tasks using tools
  - `_generate_tool_arguments()`: Uses LLM to generate JSON arguments for tool calls
  - `_format_final_response()`: Combines multi-step results into coherent response

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

## Module Import Path

When importing from the project:
```python
from linus.agents.agent.agent import ReasoningAgent, create_gemma_agent
from linus.agents.agent.tools import get_default_tools, create_custom_tool
```

Note: The `src/` directory must be in PYTHONPATH or use relative imports from `src/`.
