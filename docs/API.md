# Agent API Reference

Complete API documentation for creating and using agents with tools, memory, and metrics.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core Classes](#core-classes)
- [Agent Creation](#agent-creation)
- [State Management](#state-management)
- [Tools API](#tools-api)
- [Memory API](#memory-api)
- [Metrics API](#metrics-api)
- [Custom Implementations](#custom-implementations)

---

## Quick Start

### Basic Agent (Ollama)

```python
from linus.agents.agent import Agent
from linus.agents.agent.tools import get_default_tools

# Create agent with Ollama
tools = get_default_tools()
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
    top_k=40,
    tools=tools
)

# Run task
result = agent.run("Calculate 42 * 17")
print(result)
```

### With OpenAI

```python
agent = Agent(
    api_base="https://api.openai.com/v1",
    model="gpt-4",
    api_key="sk-your-api-key",
    temperature=0.5,
    max_tokens=1000,
    top_p=0.9,
    tools=tools,
    enable_memory=True,
    max_context_tokens=4096
)

response = agent.run("Your task", return_metrics=True)
print(f"Result: {response.result}")
print(f"Metrics: {response.metrics.to_dict()}")
```

---

## Core Classes

### Agent

Base class for all agents. Uses OpenAI client (sync or async).

```python
class Agent:
    def __init__(
        self,
        llm: Union[OpenAI, AsyncOpenAI],
        model: str,
        tools: List[BaseTool],
        verbose: bool = False,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        output_key: Optional[str] = None,
        state: Optional[SharedState] = None,
        memory_manager: Optional[MemoryManager] = None
    )
```

**Parameters:**
- `llm` (Union[OpenAI, AsyncOpenAI]): OpenAI client instance
- `model` (str): Model name (e.g., "gemma3:27b", "gpt-4")
- `tools` (List[BaseTool]): Available tools for the agent
- `verbose` (bool): Enable debug logging (default: False)
- `input_schema` (Optional[Type[BaseModel]]): Pydantic model for input validation
- `output_schema` (Optional[Type[BaseModel]]): Pydantic model for output structure
- `output_key` (Optional[str]): Key to save output in shared state
- `state` (Optional[SharedState]): SharedState instance for state management
- `memory_manager` (Optional[MemoryManager]): Memory manager instance

**Methods:**

```python
def run(
    self,
    input_data: Union[str, BaseModel, Dict[str, Any]]
) -> Union[str, BaseModel]:
    """Run the agent on the given input."""
    pass
```

---

### ReasoningAgent

Advanced agent with iterative reasoning-execution loop.

```python
class ReasoningAgent(Agent):
    def __init__(
        self,
        llm: Union[OpenAI, AsyncOpenAI],
        model: str,
        tools: List[BaseTool],
        verbose: bool = False,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        output_key: Optional[str] = None,
        state: Optional[SharedState] = None,
        max_iterations: int = 10,
        memory_manager: Optional[MemoryManager] = None,
        memory_context_ratio: float = 0.3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    )
```

**Additional Parameters:**
- `max_iterations` (int): Maximum reasoning-execution loops (default: 10)
- `memory_context_ratio` (float): Ratio of context for memory 0.0-1.0 (default: 0.3)
- `temperature` (float): Sampling temperature 0.0-2.0 (default: 0.7)
- `max_tokens` (Optional[int]): Maximum tokens to generate
- `top_p` (Optional[float]): Nucleus sampling 0.0-1.0
- `top_k` (Optional[int]): Top-k sampling (Ollama-specific)

**Methods:**

```python
def run(
    self,
    input_data: Union[str, BaseModel, Dict[str, Any]],
    return_metrics: bool = True
) -> Union[str, BaseModel, AgentResponse]:
    """
    Run the agent with iterative reasoning loop.

    Args:
        input_data: User query/task (string, dict, or Pydantic model)
        return_metrics: Return full AgentResponse with metrics (default: True)

    Returns:
        AgentResponse if return_metrics=True, otherwise just the result
    """
    pass
```

**Example:**

```python
from linus.agents.agent.agent import ReasoningAgent
from openai import OpenAI

llm = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")
agent = ReasoningAgent(
    llm=llm,
    model="gemma3:27b",
    tools=tools,
    max_iterations=10,
    temperature=0.7,
    max_tokens=2048
)

# Simple result
result = agent.run("Calculate 10 + 5", return_metrics=False)

# Full response with metrics
response = agent.run("Calculate 10 + 5", return_metrics=True)
```

---

## Agent Creation

### Agent()

Factory function to create a pre-configured ReasoningAgent for OpenAI-compatible APIs.

```python
def Agent(
    api_base: str = "http://localhost:11434/v1",
    model: str = "gemma3:27b",
    api_key: str = "not-needed",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    output_key: Optional[str] = None,
    state: Optional[SharedState] = None,
    max_iterations: int = 10,
    enable_memory: bool = False,
    memory_backend: str = "in_memory",
    max_context_tokens: int = 4096,
    memory_context_ratio: float = 0.3,
    max_memory_size: Optional[int] = 100,
    use_async: bool = False
) -> ReasoningAgent:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_base` | str | `"http://localhost:11434/v1"` | OpenAI-compatible API endpoint |
| `model` | str | `"gemma3:27b"` | Model name (e.g., "gpt-4", "gemma3:27b") |
| `api_key` | str | `"not-needed"` | API key ("not-needed" for Ollama) |
| `temperature` | float | `0.7` | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | `None` | Max tokens to generate |
| `top_p` | float | `None` | Nucleus sampling (0.0-1.0) |
| `top_k` | int | `None` | Top-k sampling (Ollama only) |
| `tools` | List[BaseTool] | `None` | Available tools |
| `verbose` | bool | `True` | Enable verbose logging |
| `input_schema` | Type[BaseModel] | `None` | Input validation schema |
| `output_schema` | Type[BaseModel] | `None` | Output structure schema |
| `output_key` | str | `None` | State key for output |
| `state` | SharedState | `None` | SharedState instance |
| `max_iterations` | int | `10` | Max reasoning loops |
| `enable_memory` | bool | `False` | Enable memory |
| `memory_backend` | str | `"in_memory"` | Backend type |
| `max_context_tokens` | int | `4096` | Context window size |
| `memory_context_ratio` | float | `0.3` | Memory percentage |
| `max_memory_size` | int | `100` | Max memories |
| `use_async` | bool | `False` | Use AsyncOpenAI client |

**Returns:**
- `ReasoningAgent`: Configured agent instance

**Examples:**

```python
from linus.agents.agent import Agent
from linus.agents.agent.tools import get_default_tools

# Ollama agent with custom parameters
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
    top_k=40,
    tools=get_default_tools()
)

# OpenAI agent
agent = Agent(
    api_base="https://api.openai.com/v1",
    model="gpt-4",
    api_key="sk-your-key",
    temperature=0.5,
    max_tokens=1000,
    top_p=0.9,
    tools=get_default_tools(),
    enable_memory=True
)

# Async agent
agent = Agent(
    model="gemma3:27b",
    use_async=True,
    tools=get_default_tools()
)
result = await agent.arun("Your task")
```

---

## State Management

### Overview

Agents support two types of state management:

1. **Dict State** (default): Simple dictionary-based state for backward compatibility
2. **SharedState**: Advanced state from the graph module with metadata, history tracking, and multi-agent support

Both state types are **fully compatible** - agents can use either transparently through the `StateWrapper`.

### Dict State

Traditional dictionary state for simple use cases.

```python
from linus.agents.agent import Agent

# Create agent with dict state
state = {"user_id": 123, "session": "abc"}
agent = Agent(tools=tools, state=state)

# Access state
agent.state["user_id"]  # 123

# Modify state
agent.state["new_key"] = "value"
```

### SharedState

Advanced state system with metadata, history, and multi-agent coordination.

```python
from linus.agents.agent import Agent
from linus.agents.graph.state import SharedState

# Create shared state
shared_state = SharedState()
shared_state.set("user_id", 123, source="init", metadata={"priority": "high"})

# Create agent with SharedState
agent = Agent(tools=tools, state=shared_state)

# Agent automatically wraps SharedState - all dict operations work
agent.state["user_id"]  # 123
agent.state["new_key"] = "value"
agent.state.get("missing", "default")  # "default"

# Data synchronized with underlying SharedState
print(shared_state.get("new_key"))  # "value"
```

### Multi-Agent State Sharing

Multiple agents can share the same SharedState for coordination.

```python
from linus.agents.graph.state import SharedState

# Create shared state
shared_state = SharedState()

# Create multiple agents with same state
agent1 = Agent(tools=tools, state=shared_state)
agent2 = Agent(tools=tools, state=shared_state)

# Agent 1 sets data
agent1.state["result_from_agent1"] = "analysis complete"

# Agent 2 can see it
print(agent2.state["result_from_agent1"])  # "analysis complete"

# Both agents share same state
agent1.state.keys() == agent2.state.keys()  # True
```

### StateWrapper API

When using SharedState, agents automatically wrap it with `StateWrapper` which provides dict-like interface:

**Methods:**

```python
# Dict-like operations
value = wrapper[key]                    # Get with KeyError if missing
wrapper[key] = value                    # Set value
key in wrapper                          # Check existence
value = wrapper.get(key, default)       # Get with default

# Collection operations
keys = wrapper.keys()                   # Get all keys
values = wrapper.values()               # Get all values
items = wrapper.items()                 # Get key-value pairs
length = len(wrapper)                   # Number of items

# Batch operations
wrapper.update({"k1": "v1", "k2": "v2"})  # Update multiple
```

**Example:**

```python
from linus.agents.graph.state import SharedState
from linus.agents.agent.agent import StateWrapper

shared_state = SharedState()
wrapper = StateWrapper(shared_state)

# Use like a dict
wrapper["name"] = "Alice"
wrapper["age"] = 30

print(wrapper.keys())     # ['name', 'age']
print(wrapper.items())    # [('name', 'Alice'), ('age', 30)]
print(len(wrapper))       # 2

# Data in underlying SharedState
print(shared_state.get("name"))  # "Alice"
```

### State in DAG Workflows

When using agents in DAG workflows, use SharedState for automatic state sharing:

```python
from linus.agents.graph import AgentDAG, AgentNode, DAGExecutor, SharedState

# Create shared state
state = SharedState()

# Create agents that share state
analyzer = Agent(tools=tools, state=state, output_key="analysis")
processor = Agent(tools=tools, state=state, output_key="processed")

# Build DAG
dag = AgentDAG("Pipeline")
dag.add_node(AgentNode(name="analyze", agent=analyzer, output_key="analysis"))
dag.add_node(AgentNode(name="process", agent=processor, output_key="processed"))
dag.add_edge("analyze", "process")

# Execute - all agents share state
executor = DAGExecutor(dag)
result = executor.execute(initial_state={"input": "data"})

# Access final state
print(result.final_state["analysis"])
print(result.final_state["processed"])
```

### Best Practices

1. **Use dict state** for simple, single-agent scenarios
2. **Use SharedState** for:
   - Multi-agent coordination
   - DAG workflows
   - Tracking state history
   - State metadata and sources

3. **State is transparent** - existing code using `agent.state` as dict works with both types

4. **History tracking** - SharedState automatically tracks all changes with timestamps and sources

---

## Tools API

### BaseTool

Base class for all tools (from LangChain).

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    """Input schema for the tool."""
    query: str = Field(description="The query parameter")
    limit: int = Field(default=10, description="Result limit")

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "Description of what the tool does"
    args_schema: Type[BaseModel] = MyToolInput

    def _run(
        self,
        query: str,
        limit: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool."""
        # Implementation
        return "result"

    async def _arun(self, *args, **kwargs):
        """Async version (optional)."""
        raise NotImplementedError("Async not supported")
```

### Built-in Tools

#### get_default_tools()

Get pre-configured default tools.

```python
from linus.agents.agent.tools import get_default_tools

tools = get_default_tools()
```

**Returns:**
- `List[BaseTool]`: List containing:
  - `SearchTool`: Search for information
  - `CalculatorTool`: Perform calculations
  - `FileReaderTool`: Read file contents
  - `ShellCommandTool`: Execute shell commands
  - `APIRequestTool`: Make HTTP requests

#### create_custom_tool()

Create a custom tool from a function.

```python
from linus.agents.agent.tools import create_custom_tool

def my_function(param1: str, param2: int = 5) -> str:
    """Function to wrap as a tool."""
    return f"Processed {param1} with {param2}"

tool = create_custom_tool(
    name="my_tool",
    description="Process something with parameters",
    func=my_function,
    args_schema=None  # Optional Pydantic schema
)
```

**Parameters:**
- `name` (str): Tool name
- `description` (str): Tool description
- `func` (callable): Function to wrap
- `args_schema` (Optional[Type[BaseModel]]): Argument schema

**Returns:**
- `StructuredTool`: Tool instance

### Tool Examples

#### Example 1: Simple Tool

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.callbacks import CallbackManagerForToolRun

class WeatherInput(BaseModel):
    city: str = Field(description="City name")

class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = "Get weather for a city"
    args_schema: Type[BaseModel] = WeatherInput

    def _run(
        self,
        city: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # Call weather API
        return f"Weather in {city}: Sunny, 72°F"

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError()

# Use the tool
weather_tool = WeatherTool()
agent = Agent(tools=[weather_tool])
```

#### Example 2: Tool with Complex Input

```python
class DatabaseQueryInput(BaseModel):
    table: str = Field(description="Table name")
    columns: List[str] = Field(description="Columns to select")
    where: Optional[str] = Field(default=None, description="WHERE clause")
    limit: int = Field(default=10, description="Result limit")

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Query the database"
    args_schema: Type[BaseModel] = DatabaseQueryInput

    def _run(
        self,
        table: str,
        columns: List[str],
        where: Optional[str] = None,
        limit: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query = f"SELECT {', '.join(columns)} FROM {table}"
        if where:
            query += f" WHERE {where}"
        query += f" LIMIT {limit}"

        # Execute query and return results
        return f"Query: {query}\nResults: ..."

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError()
```

---

## Memory API

### MemoryManager

Manages agent memory with token-aware context building.

```python
from linus.agents.agent.memory import MemoryManager, InMemoryBackend

backend = InMemoryBackend(max_size=100)
memory_mgr = MemoryManager(
    backend=backend,
    max_context_tokens=4096,
    summary_threshold_tokens=2048,
    llm=llm,  # For summarization
    encoding_name="cl100k_base"
)
```

**Constructor Parameters:**
- `backend` (MemoryBackend): Storage backend
- `max_context_tokens` (int): Maximum context window
- `summary_threshold_tokens` (int): When to trigger summarization
- `llm` (Optional[ChatOpenAI]): LLM for summarization
- `encoding_name` (str): Tokenizer encoding (default: "cl100k_base")

**Methods:**

#### add_memory()

Add a new memory entry.

```python
def add_memory(
    self,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    importance: float = 1.0,
    entry_type: str = "interaction"
) -> None:
```

**Parameters:**
- `content` (str): Memory content
- `metadata` (Optional[Dict]): Custom metadata
- `importance` (float): Importance score 0.0-1.0
- `entry_type` (str): Type ("interaction", "observation", "thought")

**Example:**

```python
memory_mgr.add_memory(
    content="User prefers technical explanations",
    metadata={"category": "preference"},
    importance=0.9,
    entry_type="observation"
)
```

#### get_context()

Get memory context within token limit.

```python
def get_context(
    self,
    max_tokens: Optional[int] = None,
    include_summary: bool = True,
    query: Optional[str] = None
) -> str:
```

**Parameters:**
- `max_tokens` (Optional[int]): Max tokens (uses max_context_tokens if None)
- `include_summary` (bool): Include summary if available
- `query` (Optional[str]): Query for semantic search

**Returns:**
- `str`: Formatted context string

**Example:**

```python
context = memory_mgr.get_context(
    max_tokens=1000,
    include_summary=True,
    query="previous calculations"
)
```

#### search_memories()

Search for relevant memories.

```python
def search_memories(
    self,
    query: str,
    limit: int = 5
) -> List[MemoryEntry]:
```

**Parameters:**
- `query` (str): Search query
- `limit` (int): Maximum results

**Returns:**
- `List[MemoryEntry]`: Matching memories

**Example:**

```python
results = memory_mgr.search_memories("calculation", limit=5)
for mem in results:
    print(f"{mem.timestamp}: {mem.content}")
```

#### get_memory_stats()

Get memory statistics.

```python
def get_memory_stats(self) -> Dict[str, Any]:
```

**Returns:**

```python
{
    "total_memories": 50,
    "total_tokens": 2450,
    "has_summary": True,
    "summary_tokens": 300,
    "max_context_tokens": 4096,
    "utilization": 0.60  # 60% of max context
}
```

#### export_memories()

Export memories for persistence.

```python
def export_memories(self) -> List[Dict[str, Any]]:
```

**Returns:**
- `List[Dict]`: JSON-serializable memory list

**Example:**

```python
import json

exported = memory_mgr.export_memories()
with open('memories.json', 'w') as f:
    json.dump(exported, f)
```

#### import_memories()

Import memories from export.

```python
def import_memories(
    self,
    memories: List[Dict[str, Any]]
) -> None:
```

**Example:**

```python
with open('memories.json', 'r') as f:
    memories = json.load(f)
memory_mgr.import_memories(memories)
```

#### clear_memory()

Clear all memories and summary.

```python
def clear_memory(self) -> None:
```

#### count_tokens()

Count tokens in text.

```python
def count_tokens(self, text: str) -> int:
```

### MemoryEntry

Individual memory entry.

```python
@dataclass
class MemoryEntry:
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    importance: float
    entry_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
```

### Memory Backends

#### InMemoryBackend

Fast in-memory storage with deque.

```python
from linus.agents.agent.memory import InMemoryBackend

backend = InMemoryBackend(max_size=100)
```

**Methods:**
- `add(entry: MemoryEntry) -> None`
- `get_recent(limit: int = 10) -> List[MemoryEntry]`
- `search(query: str, limit: int = 5) -> List[MemoryEntry]`
- `clear() -> None`
- `get_all() -> List[MemoryEntry]`
- `count() -> int`

#### VectorStoreBackend

Vector store for semantic search (stub).

```python
from linus.agents.agent.memory import VectorStoreBackend

backend = VectorStoreBackend(embedding_model=None)
```

**Same methods as InMemoryBackend**

### create_memory_manager()

Factory function for memory manager.

```python
from linus.agents.agent.memory import create_memory_manager

memory_mgr = create_memory_manager(
    backend_type="in_memory",      # or "vector_store"
    max_context_tokens=4096,
    summary_threshold_tokens=2048,
    llm=llm,
    max_size=100
)
```

---

## Metrics API

### AgentMetrics

Metrics collected during execution.

```python
@dataclass
class AgentMetrics:
    total_iterations: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    execution_time_seconds: float
    reasoning_calls: int
    tool_executions: int
    completion_checks: int
    llm_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    task_completed: bool
    iterations_to_completion: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed metrics."""
```

**Computed Fields:**
- `avg_tokens_per_llm_call`: Average tokens per LLM call
- `success_rate`: Tool success rate (0.0-1.0)

**Example:**

```python
response = agent.run("Task", return_metrics=True)
metrics = response.metrics

print(f"Iterations: {metrics.total_iterations}")
print(f"Tokens: {metrics.total_tokens}")
print(f"Time: {metrics.execution_time_seconds}s")
print(f"Success Rate: {metrics.to_dict()['success_rate']:.0%}")
```

### AgentResponse

Complete response with result and metrics.

```python
@dataclass
class AgentResponse:
    result: Union[str, BaseModel]
    metrics: AgentMetrics
    execution_history: List[Dict[str, Any]]
    completion_status: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
```

**Fields:**
- `result`: The actual answer/output
- `metrics`: Performance metrics
- `execution_history`: Detailed execution log
- `completion_status`: Task completion validation

**Example:**

```python
response = agent.run("Calculate 42 * 17", return_metrics=True)

# Access result
print(response.result)

# Access metrics
print(response.metrics.to_dict())

# Access history
for step in response.execution_history:
    print(f"{step['task']}: {step['result']}")

# Check completion
if response.completion_status:
    print(f"Complete: {response.completion_status['is_complete']}")
```

---

## Custom Implementations

### Creating a Custom Agent

```python
from linus.agents.agent.agent import Agent
from typing import Union, Dict, Any
from pydantic import BaseModel

class MyCustomAgent(Agent):
    def __init__(self, llm, tools, **kwargs):
        super().__init__(llm, tools, **kwargs)
        # Custom initialization
        self.custom_config = {}

    def run(
        self,
        input_data: Union[str, BaseModel, Dict[str, Any]]
    ) -> Union[str, BaseModel]:
        """Custom run logic."""

        # 1. Validate input
        input_text = self._validate_and_convert_input(input_data)

        # 2. Your custom logic here
        result = self._process(input_text)

        # 3. Format output
        return self._format_output(result)

    def _process(self, input_text: str) -> str:
        """Custom processing logic."""
        # Use tools
        for tool_name, tool in self.tool_map.items():
            if self._should_use_tool(tool_name, input_text):
                result = tool.run({"query": input_text})
                return result

        # Fallback to LLM
        from langchain_core.messages import HumanMessage
        response = self.llm.invoke([HumanMessage(content=input_text)])
        return response.content

    def _should_use_tool(self, tool_name: str, input_text: str) -> bool:
        """Decide if tool should be used."""
        # Custom logic
        return tool_name in input_text.lower()
```

### Creating a Custom Tool

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
import requests

class APIToolInput(BaseModel):
    endpoint: str = Field(description="API endpoint")
    method: str = Field(default="GET", description="HTTP method")
    payload: Optional[dict] = Field(default=None, description="Request payload")

class CustomAPITool(BaseTool):
    """Custom API integration tool."""

    name: str = "custom_api"
    description: str = "Call custom API endpoints"
    args_schema: Type[BaseModel] = APIToolInput

    # Tool-specific configuration
    base_url: str = "https://api.example.com"
    api_key: str = "your-api-key"

    def _run(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Optional[dict] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute API call."""
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=payload)
        else:
            return f"Unsupported method: {method}"

        return response.text

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not implemented")

# Usage
tool = CustomAPITool(base_url="https://api.example.com", api_key="key")
agent = Agent(tools=[tool])
```

### Creating a Custom Memory Backend

```python
from linus.agents.agent.memory import MemoryBackend, MemoryEntry
from typing import List
import redis

class RedisMemoryBackend(MemoryBackend):
    """Redis-based memory backend."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.key_prefix = "agent:memory:"

    def add(self, entry: MemoryEntry) -> None:
        """Add memory to Redis."""
        key = f"{self.key_prefix}{entry.timestamp.timestamp()}"
        self.redis.set(key, entry.to_dict())

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get recent memories from Redis."""
        keys = self.redis.keys(f"{self.key_prefix}*")
        keys = sorted(keys, reverse=True)[:limit]

        memories = []
        for key in keys:
            data = self.redis.get(key)
            memories.append(MemoryEntry.from_dict(data))

        return memories

    def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Search memories in Redis."""
        all_memories = self.get_all()
        query_lower = query.lower()

        matches = [
            mem for mem in all_memories
            if query_lower in mem.content.lower()
        ]

        return sorted(
            matches,
            key=lambda m: (m.importance, m.timestamp),
            reverse=True
        )[:limit]

    def clear(self) -> None:
        """Clear all memories."""
        keys = self.redis.keys(f"{self.key_prefix}*")
        if keys:
            self.redis.delete(*keys)

    def get_all(self) -> List[MemoryEntry]:
        """Get all memories."""
        keys = self.redis.keys(f"{self.key_prefix}*")
        return [
            MemoryEntry.from_dict(self.redis.get(key))
            for key in keys
        ]

    def count(self) -> int:
        """Count memories."""
        return len(self.redis.keys(f"{self.key_prefix}*"))

# Usage
from linus.agents.agent.memory import MemoryManager

backend = RedisMemoryBackend(redis_url="redis://localhost:6379")
memory_mgr = MemoryManager(
    backend=backend,
    max_context_tokens=4096,
    llm=llm
)
```

### Structured Input/Output

```python
from pydantic import BaseModel, Field

# Define input schema
class TaskInput(BaseModel):
    objective: str = Field(description="Task objective")
    constraints: List[str] = Field(default_factory=list, description="Constraints")
    priority: int = Field(default=5, description="Priority 1-10")

# Define output schema
class TaskOutput(BaseModel):
    status: str = Field(description="Completion status")
    result: str = Field(description="Task result")
    steps_taken: List[str] = Field(description="Steps executed")
    recommendations: Optional[List[str]] = Field(default=None)

# Create agent with schemas
agent = Agent(
    tools=tools,
    input_schema=TaskInput,
    output_schema=TaskOutput
)

# Use with structured input
task = TaskInput(
    objective="Analyze data",
    constraints=["max 5 minutes", "use only public APIs"],
    priority=8
)

response = agent.run(task, return_metrics=True)

# Access structured output
if isinstance(response.result, TaskOutput):
    print(f"Status: {response.result.status}")
    print(f"Steps: {response.result.steps_taken}")
```

---

## Complete Example

### Multi-Agent System with Memory

```python
from linus.agents.agent import Agent
from linus.agents.agent.tools import get_default_tools, create_custom_tool
from linus.agents.agent.memory import create_memory_manager

# Custom tools
def analyze_data(data: str) -> str:
    """Analyze data and return insights."""
    return f"Analysis of {data}: [insights here]"

def generate_report(analysis: str) -> str:
    """Generate report from analysis."""
    return f"Report based on: {analysis}"

# Create tools
tools = get_default_tools()
tools.append(create_custom_tool("analyze", "Analyze data", analyze_data))
tools.append(create_custom_tool("report", "Generate report", generate_report))

# Shared memory
shared_memory = create_memory_manager(
    backend_type="in_memory",
    max_context_tokens=6000,
    max_size=200
)

# Create specialized agents
data_agent = Agent(
    tools=tools,
    enable_memory=True,
    max_context_tokens=6000,
    memory_context_ratio=0.2
)

report_agent = Agent(
    tools=tools,
    enable_memory=True,
    max_context_tokens=6000,
    memory_context_ratio=0.3
)

# Workflow
data_response = data_agent.run(
    "Analyze sales data for Q4",
    return_metrics=True
)

report_response = report_agent.run(
    "Generate executive report based on the analysis",
    return_metrics=True
)

# Results
print(f"Data Analysis: {data_response.result}")
print(f"Report: {report_response.result}")

# Combined metrics
total_tokens = (
    data_response.metrics.total_tokens +
    report_response.metrics.total_tokens
)
print(f"Total tokens used: {total_tokens}")
```

---

## Error Handling

```python
from linus.agents.agent import Agent

agent = Agent(tools=tools)

try:
    response = agent.run("Complex task", return_metrics=True)

    if not response.metrics.task_completed:
        print(f"Warning: Task incomplete")
        print(f"Reason: {response.completion_status['reasoning']}")

    if response.metrics.failed_tool_calls > 0:
        print(f"Warning: {response.metrics.failed_tool_calls} tool failures")

except Exception as e:
    print(f"Error: {e}")
```

---

## Best Practices

### 1. Tool Design

```python
# ✅ Good: Clear description, proper schema
class GoodToolInput(BaseModel):
    query: str = Field(description="Specific search query")
    limit: int = Field(default=10, description="Max results (1-100)")

class GoodTool(BaseTool):
    name: str = "search_knowledge"
    description: str = "Search knowledge base for specific information about products, policies, or procedures"
    args_schema: Type[BaseModel] = GoodToolInput

# ❌ Bad: Vague description, no schema
class BadTool(BaseTool):
    name: str = "tool"
    description: str = "Does stuff"  # Too vague!
```

### 2. Memory Configuration

```python
# ✅ Good: Appropriate settings for use case
agent = Agent(
    enable_memory=True,
    max_context_tokens=6000,      # Leave headroom
    memory_context_ratio=0.3,     # Balanced
    max_memory_size=100           # Reasonable limit
)

# ❌ Bad: Will cause context overflow
agent = Agent(
    enable_memory=True,
    max_context_tokens=100000,    # Too large!
    memory_context_ratio=0.9,     # Too much for memory
    max_memory_size=None          # Unlimited - dangerous!
)
```

### 3. Metrics Usage

```python
# ✅ Good: Monitor and optimize
response = agent.run(task, return_metrics=True)

if response.metrics.execution_time_seconds > 30:
    logger.warning("Task took too long, consider optimization")

if response.metrics.total_tokens > 5000:
    logger.warning("High token usage, check context size")

# ❌ Bad: Ignore metrics
response = agent.run(task, return_metrics=False)
# No visibility into performance!
```

---

## API Changelog

### Version 1.0.0 (Current)

- Initial release with:
  - `Agent` and `ReasoningAgent` classes
  - Memory management system
  - Comprehensive metrics tracking
  - Multiple storage backends
  - Tool creation utilities

---

## Support

For issues, questions, or contributions:
- GitHub: [github.com/maai-solutions/agents](https://github.com/maai-solutions/agents)
- Documentation: See `QUICK_START.md`, `MEMORY.md`, `METRICS.md`
