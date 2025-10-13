# ReasoningAgent Documentation

## Overview

The `ReasoningAgent` is a sophisticated AI agent implementation designed to work with LLMs that don't have native tool/function calling support (like Gemma, Llama, and other open-source models). It uses a **two-phase approach** to break down complex tasks and execute them systematically.

### Key Features

- **Two-Phase Execution**: Separates reasoning from execution for better control
- **Iterative Loop**: Continues until task completion or max iterations reached
- **Memory Management**: Optional persistent memory across conversations
- **Telemetry Support**: Built-in tracing with Langfuse and OpenTelemetry
- **Rich Logging**: Beautiful console output with progress tracking
- **Async-First**: Fully asynchronous for high-performance applications
- **OpenAI-Compatible**: Works with any OpenAI-compatible API (Ollama, OpenAI, etc.)

## Architecture

### Two-Phase Approach

The ReasoningAgent uses a unique two-phase execution model:

#### Phase 1: Reasoning Phase
The agent analyzes the user's request and creates a plan:
- Evaluates if there's sufficient information to proceed
- Breaks down the task into discrete steps
- Identifies which tools are needed for each step
- Returns structured JSON with reasoning and task list

#### Phase 2: Execution Phase
For each planned task:
- Generates tool arguments using another LLM call
- Executes the tool with validated arguments
- Collects results and updates context
- Continues to next task with enriched context

#### Phase 3: Completion Check
After each iteration:
- Validates if the task is complete
- Identifies missing steps
- Plans next actions if incomplete
- Returns final response when complete

### Iterative Loop

The agent runs in a loop until:
- Task is marked as complete
- Maximum iterations reached (default: 10)
- Insufficient information and no tasks planned

Each iteration:
1. Builds context from memory + previous results
2. Runs reasoning phase with updated context
3. Executes all planned tasks
4. Checks for completion
5. Either returns or continues to next iteration

## Class Structure

### Constructor Parameters

```python
ReasoningAgent(
    llm: Union[AsyncOpenAI, OpenAI],              # OpenAI client instance
    model: str,                                    # Model name (e.g., "gemma3:27b")
    tools: List[BaseTool],                         # Available tools
    verbose: bool = False,                         # Enable debug logging
    input_schema: Optional[Type[BaseModel]] = None,  # Input validation schema
    output_schema: Optional[Type[BaseModel]] = None, # Output formatting schema
    output_key: Optional[str] = None,              # Key for shared state storage
    state: Optional[SharedState] = None,           # Shared state for multi-agent
    max_iterations: int = 10,                      # Maximum execution loops
    memory_manager: Optional[MemoryManager] = None, # Persistent memory
    memory_context_ratio: float = 0.3,             # Memory context proportion
    temperature: float = 0.7,                      # LLM sampling temperature
    max_tokens: Optional[int] = None,              # Max tokens per completion
    top_p: Optional[float] = None,                 # Nucleus sampling
    top_k: Optional[int] = None,                   # Top-k sampling (Ollama)
    api_base: Optional[str] = None,                # API endpoint reference
    use_json_format: bool = False                  # Force JSON response format
)
```

### Core Methods

#### `run(input_data, return_metrics=True)`
Main entry point for agent execution.

**Parameters:**
- `input_data`: User request (string, dict, or Pydantic model)
- `return_metrics`: If `True`, return `AgentResponse` with metrics; if `False`, return only result

**Returns:**
- `AgentResponse` (with metrics) or formatted result (without metrics)

**Example:**
```python
# With metrics
response = await agent.run("Calculate 42 * 17", return_metrics=True)
print(response.result)           # "714"
print(response.metrics)          # AgentMetrics object
print(response.execution_history) # List of executed tasks

# Without metrics (just result)
result = await agent.run("Calculate 42 * 17", return_metrics=False)
print(result)  # "714"
```

#### `_reasoning_call(input_text, iteration)`
Performs the reasoning phase to analyze the request and plan tasks.

**Parameters:**
- `input_text`: The user's request with context
- `iteration`: Current iteration number

**Returns:**
- `ReasoningResult` with:
  - `has_sufficient_info`: Boolean indicating if task can proceed
  - `tasks`: List of planned tasks with tool names
  - `reasoning`: Explanation of the analysis

**Example Response:**
```json
{
    "has_sufficient_info": true,
    "reasoning": "User wants to calculate 42 * 17. I'll use the calculator tool.",
    "tasks": [
        {
            "description": "Calculate 42 * 17",
            "tool_name": "calculator",
            "requires_user_input": false
        }
    ]
}
```

#### `_execute_task_with_tool(task, context)`
Executes a single task using the specified tool.

**Parameters:**
- `task`: `TaskExecution` object with task details
- `context`: Current context including previous results

**Returns:**
- String result from tool execution

**Process:**
1. Validates tool exists in tool map
2. Generates tool arguments via `_generate_tool_arguments()`
3. Executes tool with `tool.arun(args)`
4. Updates metrics and traces
5. Returns result or error message

#### `_generate_tool_arguments(task, tool, context)`
Uses LLM to generate JSON arguments for tool calls.

**Parameters:**
- `task`: Task requiring tool execution
- `tool`: Tool instance to use
- `context`: Current context for argument generation

**Returns:**
- Dictionary of tool arguments or `None` if generation failed

**Features:**
- Uses tool schema from `tool.args_schema.model_json_schema()`
- Lower temperature (0.3) for consistent JSON generation
- Multiple fallback strategies for JSON extraction
- Telemetry tracking with structured output

#### `_check_completion(original_request, execution_history)`
Validates if the task has been completed successfully.

**Parameters:**
- `original_request`: The original user request
- `execution_history`: List of executed tasks and results

**Returns:**
- Dictionary with:
  - `is_complete`: Boolean completion status
  - `reasoning`: Explanation of status
  - `missing_steps`: List of remaining steps
  - `next_action`: What to do next

**Example:**
```json
{
    "is_complete": true,
    "reasoning": "Successfully calculated 42 * 17 = 714",
    "missing_steps": [],
    "next_action": "none"
}
```

## Response Format

### AgentResponse Object

When `return_metrics=True`, the agent returns an `AgentResponse` object:

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
        "reasoning_calls": 1,
        "completion_checks": 1,
        "task_completed": true,
        "iterations_to_completion": 1,
        "avg_tokens_per_llm_call": 500.0,
        "success_rate": 1.0
    },
    "execution_history": [
        {
            "iteration": 1,
            "task": "Calculate 42 * 17",
            "tool": "calculator",
            "result": "714",
            "status": "completed"
        }
    ],
    "completion_status": {
        "is_complete": true,
        "reasoning": "Task completed successfully",
        "missing_steps": [],
        "next_action": "none"
    }
}
```

## Usage Examples

### Basic Usage with Ollama

```python
import asyncio
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools

# Create agent
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    api_key="not-needed",
    temperature=0.7,
    tools=get_default_tools(),
    verbose=True,
    use_async=True
)

# Run simple calculation
async def main():
    response = await agent.run("What is 42 * 17?")
    print(response.result)  # "The result of 42 * 17 is 714"
    print(f"Tokens used: {response.metrics.total_tokens}")

asyncio.run(main())
```

### Multi-Step Task with Tools

```python
# Complex task requiring multiple tools
response = await agent.run(
    "Search for information about Python asyncio, then read the README.md file"
)

# Check execution history
for step in response.execution_history:
    print(f"Step {step['iteration']}: {step['task']}")
    print(f"  Tool: {step['tool']}")
    print(f"  Status: {step['status']}")
    print(f"  Result: {step['result'][:100]}...")
```

### With Custom Temperature and Max Tokens

```python
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    temperature=0.3,      # Lower temperature for more focused responses
    max_tokens=1000,      # Limit response length
    top_p=0.9,            # Nucleus sampling
    top_k=40,             # Top-k sampling (Ollama-specific)
    tools=get_default_tools(),
    use_async=True
)

response = await agent.run("Explain quantum computing in simple terms")
```

### With Memory Management

```python
from linus.agents.agent.memory import MemoryManager

# Create memory manager
memory = MemoryManager(
    max_context_tokens=4000,
    storage_path="./agent_memory"
)

# Create agent with memory
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    memory_manager=memory,
    memory_context_ratio=0.3,  # Use 30% of context for memory
    use_async=True
)

# First conversation
await agent.run("My name is Alice and I like Python programming")

# Second conversation - agent remembers previous context
response = await agent.run("What's my name and what do I like?")
print(response.result)  # "Your name is Alice and you like Python programming"

# Check memory stats
stats = memory.get_memory_stats()
print(f"Total memories: {stats['total_memories']}")
```

### With Input/Output Schemas

```python
from pydantic import BaseModel, Field
from typing import List

# Define input schema
class TaskInput(BaseModel):
    task: str = Field(..., description="The task to perform")
    priority: int = Field(1, ge=1, le=5, description="Priority level")

# Define output schema
class TaskOutput(BaseModel):
    result: str = Field(..., description="The task result")
    steps_taken: List[str] = Field(..., description="Steps performed")
    success: bool = Field(..., description="Whether task succeeded")

# Create agent with schemas
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    input_schema=TaskInput,
    output_schema=TaskOutput,
    use_async=True
)

# Use with structured input
task_input = TaskInput(task="Calculate 42 * 17", priority=5)
response = await agent.run(task_input)

# Response is automatically formatted as TaskOutput
print(response.result.result)        # "714"
print(response.result.steps_taken)   # ["Used calculator tool", ...]
print(response.result.success)       # True
```

### With Shared State (Multi-Agent)

```python
from linus.agents.graph.state import SharedState

# Create shared state
state = SharedState()

# Agent 1: Research agent
research_agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=[SearchTool()],
    state=state,
    output_key="research_results",
    use_async=True
)

# Agent 2: Summary agent
summary_agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=[],
    state=state,
    use_async=True
)

# Agent 1 stores results in state
await research_agent.run("Search for latest AI trends")

# Agent 2 can access results from state
research_data = state.get("research_results")
await summary_agent.run(f"Summarize this research: {research_data}")
```

### With Langfuse Tracing

```python
from linus.agents.telemetry import initialize_telemetry

# Initialize Langfuse tracer
tracer = initialize_telemetry(
    service_name="my-agent",
    exporter_type="langfuse",
    enabled=True
)

# Create agent with tracing
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    tracer=tracer,
    session_id="user-123-session-456",  # Group traces by session
    use_async=True
)

# All operations are automatically traced in Langfuse
response = await agent.run("What is 42 * 17?")

# Flush traces before exit
tracer.flush()
```

### Error Handling

```python
try:
    response = await agent.run("Invalid request that will fail")

    if not response.completion_status["is_complete"]:
        print(f"Task incomplete: {response.completion_status['reasoning']}")
        print(f"Missing steps: {response.completion_status['missing_steps']}")

    # Check for failed tool calls
    if response.metrics.failed_tool_calls > 0:
        print(f"Warning: {response.metrics.failed_tool_calls} tool calls failed")

except Exception as e:
    print(f"Agent execution failed: {e}")
```

### Iterative Task Example

```python
# Complex task requiring multiple iterations
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    max_iterations=5,  # Allow up to 5 reasoning-execution cycles
    verbose=True,
    use_async=True
)

response = await agent.run(
    "Find the weather in New York, then calculate how many degrees "
    "warmer it is compared to the average temperature of 55°F"
)

# Check how many iterations it took
print(f"Completed in {response.metrics.total_iterations} iterations")
print(f"Total LLM calls: {response.metrics.llm_calls}")
print(f"Tools executed: {response.metrics.tool_executions}")
```

## Debugging and Logging

### Enable Verbose Logging

```python
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    verbose=True,  # Enable debug logging
    use_async=True
)
```

### Log Prefixes

The agent uses structured log prefixes for filtering:

- `[RUN]`: Main execution flow and iteration tracking
- `[REASONING]`: Reasoning phase (prompts, responses, parsed results)
- `[EXECUTION]`: Task execution flow
- `[TOOL-ARGS]`: Tool argument generation
- `[FINAL]`: Final response formatting
- `[MEMORY]`: Memory operations
- `[METRICS]`: Performance metrics

### Filter Logs

```bash
# View only reasoning logs
python app.py 2>&1 | grep "\[REASONING\]"

# View only execution logs
python app.py 2>&1 | grep "\[EXECUTION\]"

# View errors only
python app.py 2>&1 | grep "ERROR"
```

### Rich Console Output

When rich is available, the agent displays beautiful formatted output:

```python
# Metrics are displayed in a formatted table
# Reasoning results shown in bordered panels
# Progress tracking with visual indicators
```

## Performance Optimization

### Token Usage

```python
# Monitor token usage
response = await agent.run("Your query")
print(f"Total tokens: {response.metrics.total_tokens}")
print(f"Average per LLM call: {response.metrics.avg_tokens_per_llm_call}")

# Optimize with max_tokens
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    max_tokens=500,  # Limit each completion
    tools=get_default_tools(),
    use_async=True
)
```

### Memory Context Optimization

```python
# Adjust memory context ratio
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    memory_manager=memory,
    memory_context_ratio=0.2,  # Use only 20% of context for memory
    use_async=True
)
```

### Iteration Limits

```python
# Limit iterations for faster responses
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    max_iterations=3,  # Stop after 3 reasoning-execution cycles
    tools=get_default_tools(),
    use_async=True
)
```

## Advanced Features

### Custom Prompts

The agent uses three main prompts that can be customized:

1. **Reasoning Prompt** (`_create_reasoning_prompt()`): For task analysis and planning
2. **Execution Prompt** (`_create_execution_prompt()`): For tool argument generation
3. **Completion Check Prompt** (`_create_completion_check_prompt()`): For validating completion

To customize, subclass `ReasoningAgent` and override these methods.

### Telemetry and Observability

The agent automatically traces:
- Agent runs with input/output
- Reasoning phases with parsed results
- LLM calls with prompts and completions
- Tool executions with arguments and results
- Token usage and performance metrics

All traces are sent to configured exporters (Langfuse, OpenTelemetry, Jaeger).

### JSON Format Enforcement

```python
# Force JSON response format (requires model support)
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    use_json_format=True,  # Use response_format={"type": "json_object"}
    tools=get_default_tools(),
    use_async=True
)
```

**Note:** Not all models support this. Ollama models generally don't support `response_format`.

## Common Patterns

### Search and Analyze Pattern

```python
# Agent automatically plans: search → analyze → respond
response = await agent.run(
    "Search for information about FastAPI and explain its main features"
)
```

### Calculate and Compare Pattern

```python
# Multi-step calculation with comparison
response = await agent.run(
    "Calculate 42 * 17, then calculate 50 * 15, and tell me which is larger"
)
```

### Read and Summarize Pattern

```python
# Read file and generate summary
response = await agent.run(
    "Read the file ./README.md and provide a 3-sentence summary"
)
```

### API and Process Pattern

```python
# Fetch data and process it
response = await agent.run(
    "Fetch weather data for New York and calculate if it's warmer than average"
)
```

## Troubleshooting

### Agent Returns Incomplete Results

**Issue:** Task is marked incomplete after max iterations.

**Solution:**
- Increase `max_iterations` parameter
- Check execution history to see what's failing
- Review logs for tool execution errors
- Simplify the task or break it into smaller parts

### Tool Argument Generation Fails

**Issue:** Agent can't generate valid JSON arguments for tools.

**Solution:**
- Lower the temperature for tool argument generation (automatic)
- Ensure tool schemas are valid Pydantic models
- Check tool descriptions are clear and unambiguous
- Use simpler parameter names and types

### Memory Context Issues

**Issue:** Agent doesn't remember previous context.

**Solution:**
- Ensure `memory_manager` is passed to agent
- Increase `memory_context_ratio` (default: 0.3)
- Check memory stats with `memory.get_memory_stats()`
- Verify memory storage path is writable

### High Token Usage

**Issue:** Agent uses too many tokens per request.

**Solution:**
- Set `max_tokens` to limit each completion
- Reduce `memory_context_ratio` to use less memory context
- Lower `max_iterations` to limit reasoning cycles
- Use more specific queries to reduce reasoning complexity

## Best Practices

1. **Start Simple**: Begin with simple queries to test agent setup
2. **Use Verbose Mode**: Enable verbose logging during development
3. **Monitor Metrics**: Check token usage and execution time
4. **Handle Failures**: Always check `completion_status` and handle incomplete tasks
5. **Use Memory Wisely**: Don't store sensitive data in memory
6. **Trace Everything**: Enable telemetry for production debugging
7. **Validate Tools**: Test tools independently before using with agent
8. **Schema Validation**: Use input/output schemas for structured data
9. **Async Always**: Use async methods for better performance
10. **Flush Traces**: Always call `tracer.flush()` before exit

## Related Documentation

- [Factory Function](AGENT_FACTORY.md) - Creating agents with the factory
- [Tools](TOOLS.md) - Available tools and custom tool creation
- [Telemetry](TELEMETRY.md) - Observability and tracing setup
- [Langfuse Integration](LANGFUSE_INTEGRATION.md) - LLM-specific tracing
- [Memory Management](MEMORY.md) - Persistent memory across conversations
- [Rich Logging](RICH_LOGGING.md) - Beautiful console output

## API Reference

### ReasoningAgent Class

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `run()` | Execute agent with input | `input_data`, `return_metrics` | `AgentResponse` or result |
| `_reasoning_call()` | Perform reasoning phase | `input_text`, `iteration` | `ReasoningResult` |
| `_execute_task_with_tool()` | Execute task with tool | `task`, `context` | `str` |
| `_generate_tool_arguments()` | Generate tool arguments | `task`, `tool`, `context` | `Dict` or `None` |
| `_check_completion()` | Check task completion | `original_request`, `execution_history` | `Dict` |
| `_generate_response()` | Generate direct response | `task_description`, `context` | `str` |
| `_format_final_response()` | Format final response | `original_request`, `results` | `str` |

### AgentMetrics Fields

| Field | Type | Description |
|-------|------|-------------|
| `total_iterations` | `int` | Number of reasoning-execution cycles |
| `total_tokens` | `int` | Total tokens used across all LLM calls |
| `execution_time_seconds` | `float` | Total execution time |
| `llm_calls` | `int` | Total number of LLM API calls |
| `tool_executions` | `int` | Total tool executions attempted |
| `successful_tool_calls` | `int` | Successfully completed tool calls |
| `failed_tool_calls` | `int` | Failed tool calls |
| `reasoning_calls` | `int` | Number of reasoning phase calls |
| `completion_checks` | `int` | Number of completion checks |
| `task_completed` | `bool` | Whether task completed successfully |
| `iterations_to_completion` | `int` or `None` | Iterations needed to complete |
| `avg_tokens_per_llm_call` | `float` | Average tokens per LLM call |
| `success_rate` | `float` | Ratio of successful tool calls |

## License

This project is part of the Linus Agent Framework. See main repository for license details.
