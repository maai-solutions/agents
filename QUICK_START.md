# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install openai tiktoken pydantic loguru fastapi uvicorn langchain-core
```

## Prerequisites

- **Ollama** (for local models): Install from [ollama.ai](https://ollama.ai) and pull a model:
  ```bash
  ollama pull gemma3:27b
  ```
- **OpenAI API** (for cloud models): Get your API key from [platform.openai.com](https://platform.openai.com/api-keys)

## Basic Usage

### Simple Agent with Ollama

```python
from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools

tools = get_default_tools()
agent = create_gemma_agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
    tools=tools
)

result = agent.run("Calculate 42 * 17", return_metrics=False)
print(result)
```

### Simple Agent with OpenAI

```python
agent = create_gemma_agent(
    api_base="https://api.openai.com/v1",
    model="gpt-4",
    api_key="sk-your-api-key",
    temperature=0.5,
    max_tokens=1000,
    top_p=0.9,
    tools=tools
)

result = agent.run("What is the capital of France?")
print(result)
```

### Agent with Memory

```python
agent = create_gemma_agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    api_key="not-needed",
    temperature=0.7,
    tools=tools,
    enable_memory=True,
    max_context_tokens=4096,
    memory_context_ratio=0.3
)

# Conversation with context
agent.run("My name is Alice")
agent.run("Calculate 10 + 5")
agent.run("What's my name?")  # Remembers "Alice"
```

### Agent with Metrics

```python
response = agent.run("Complex task here", return_metrics=True)

print(f"Result: {response.result}")
print(f"Iterations: {response.metrics.total_iterations}")
print(f"Tokens: {response.metrics.total_tokens}")
print(f"Time: {response.metrics.execution_time_seconds}s")
```

## Configuration Cheat Sheet

### For Short Tasks (Ollama)

```python
agent = create_gemma_agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=1024,
    tools=tools,
    max_iterations=5,              # Quick completion
    enable_memory=False,           # No memory needed
    verbose=False                  # Minimal logging
)
```

### For Conversational Use (OpenAI)

```python
agent = create_gemma_agent(
    api_base="https://api.openai.com/v1",
    model="gpt-4",
    api_key="sk-your-key",
    temperature=0.6,
    max_tokens=2000,
    top_p=0.9,
    tools=tools,
    max_iterations=10,
    enable_memory=True,
    max_context_tokens=6000,
    memory_context_ratio=0.4,      # 40% for memory
    max_memory_size=100,
    verbose=True
)
```

## Generation Parameters Guide

| Parameter | Range | Purpose | When to Use |
|-----------|-------|---------|-------------|
| `temperature` | 0.0-2.0 | Controls randomness | Lower (0.3-0.5) for factual tasks, higher (0.7-1.0) for creative tasks |
| `max_tokens` | 1-∞ | Max output length | Set based on expected response length |
| `top_p` | 0.0-1.0 | Nucleus sampling | Alternative to temperature, usually 0.9-0.95 |
| `top_k` | 1-∞ | Top-k sampling | Ollama only, typically 40-60 |

### For Production

```python
agent = create_gemma_agent(
    tools=tools,
    max_iterations=10,
    enable_memory=True,
    max_context_tokens=6000,
    memory_context_ratio=0.3,
    max_memory_size=200,
    memory_backend="in_memory",
    verbose=False                  # Log to file instead
)
```

## Common Patterns

### Pattern 1: Single Query

```python
agent = create_gemma_agent(tools=tools)
answer = agent.run("What is 100 / 4?", return_metrics=False)
```

### Pattern 2: Multi-Turn Conversation

```python
agent = create_gemma_agent(tools=tools, enable_memory=True)

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = agent.run(user_input, return_metrics=False)
    print(f"Agent: {response}")
```

### Pattern 3: With Progress Tracking

```python
agent = create_gemma_agent(tools=tools)

tasks = ["Calculate 10*2", "Search Python", "Get time"]
for task in tasks:
    print(f"Processing: {task}")
    response = agent.run(task, return_metrics=True)
    print(f"  Completed in {response.metrics.execution_time_seconds:.2f}s")
```

### Pattern 4: Memory Persistence

```python
import json

# Save session
agent = create_gemma_agent(tools=tools, enable_memory=True)
# ... use agent ...
with open('session.json', 'w') as f:
    json.dump(agent.memory_manager.export_memories(), f)

# Restore session
agent = create_gemma_agent(tools=tools, enable_memory=True)
with open('session.json', 'r') as f:
    agent.memory_manager.import_memories(json.load(f))
```

## Available Tools

Default tools include:
- **calculator**: Perform mathematical calculations
- **search**: Search for information (mock implementation)
- **read_file**: Read file contents
- **shell_command**: Execute shell commands
- **api_request**: Make HTTP API requests
- **get_time**: Get current date/time

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_base` | `http://localhost:11434/v1` | LLM API endpoint |
| `model` | `gemma3:27b` | Model name |
| `max_iterations` | `10` | Max reasoning loops |
| `enable_memory` | `False` | Enable memory |
| `max_context_tokens` | `4096` | Context window size |
| `memory_context_ratio` | `0.3` | Memory % of context |
| `max_memory_size` | `100` | Max memories to keep |
| `verbose` | `True` | Enable debug logging |

## Metrics Available

```python
response.metrics.to_dict()
```

Returns:
- `total_iterations`: Reasoning loops
- `total_tokens`: Total tokens used
- `total_input_tokens`: Input tokens
- `total_output_tokens`: Output tokens
- `execution_time_seconds`: Total time
- `llm_calls`: Number of LLM calls
- `tool_executions`: Tools executed
- `successful_tool_calls`: Successful tools
- `failed_tool_calls`: Failed tools
- `success_rate`: Tool success rate
- `task_completed`: Whether task completed
- `iterations_to_completion`: Iterations needed

## Memory Operations

```python
# Get stats
stats = agent.memory_manager.get_memory_stats()

# Search memories
results = agent.memory_manager.search_memories("query", limit=5)

# Add custom memory
agent.memory_manager.add_memory(
    content="Important info",
    importance=0.9
)

# Export/import
exported = agent.memory_manager.export_memories()
agent.memory_manager.import_memories(exported)

# Clear
agent.memory_manager.clear_memory()
```

## Testing

```bash
# Run all tests
python tests/test_agent_loop.py
python tests/test_agent_metrics.py
python tests/test_agent_memory.py

# Run specific test
python -c "from tests.test_agent_memory import test_basic_memory; test_basic_memory()"
```

## Troubleshooting

### Memory not working?

```python
# Check if enabled
print(f"Memory: {agent.memory_manager is not None}")

# Check stats
if agent.memory_manager:
    print(agent.memory_manager.get_memory_stats())
```

### Context overflow?

```python
# Reduce memory ratio
memory_context_ratio=0.2

# Or reduce context size
max_context_tokens=3000
```

### Agent not completing tasks?

```python
# Increase iterations
max_iterations=15

# Check logs
verbose=True
```

## Documentation

- **Full Documentation**: See `AGENT_ENHANCEMENTS.md`
- **Memory Guide**: See `MEMORY.md`
- **Metrics Guide**: See `METRICS.md`

## Examples

See `tests/` directory for comprehensive examples:
- `test_agent_loop.py` - Iterative execution
- `test_agent_metrics.py` - Metrics tracking
- `test_agent_memory.py` - Memory management
