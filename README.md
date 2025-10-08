# Multi-Agent System with DAG Orchestration

A complete framework for building sophisticated AI agent systems with memory, metrics tracking, and DAG-based workflow orchestration.

## 🌟 Features

### Core Agent System
- ✅ **Iterative Reasoning Loop** - Agents loop through reasoning-execution-validation until task completion
- ✅ **Async/Await Support** - Non-blocking execution with `arun()` for high performance
- ✅ **Comprehensive Metrics** - Track tokens, execution time, iterations, tool usage, and success rates
- ✅ **Memory Management** - Persistent context with automatic token management and summarization
- ✅ **Tool Integration** - Extensible tool system with built-in and custom tools
- ✅ **Structured I/O** - Optional Pydantic schemas for input validation and output structure

### DAG Orchestration
- ✅ **Multi-Agent Workflows** - Coordinate multiple agents in complex DAGs
- ✅ **Parallel Execution** - Run independent DAG nodes concurrently with `aexecute()`
- ✅ **Conditional Execution** - Dynamic routing based on state
- ✅ **Parallel Planning** - Automatic detection of parallelizable tasks
- ✅ **Error Recovery** - Configurable error handling per node
- ✅ **Unified State Management** - SharedState with metadata, history, and multi-agent coordination
- ✅ **Visualization** - Built-in DAG visualization for debugging

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/maai-solutions/agents.git
cd agents

# Install dependencies
pip install -r requirements.txt

# Or install specific packages
pip install openai pydantic loguru tiktoken fastapi uvicorn langchain-core
```

## 🔧 Prerequisites

- **Local LLM (Ollama)**: Install [Ollama](https://ollama.ai) and pull a model:
  ```bash
  ollama pull gemma3:27b
  ```
- **OpenAI API**: Get an API key from [OpenAI](https://platform.openai.com/api-keys)

## 🚀 Quick Start

### Single Agent (Ollama)

```python
from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools

# Create agent with Ollama
tools = get_default_tools()
agent = create_gemma_agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
    top_k=40,
    tools=tools,
    enable_memory=True,
    max_context_tokens=4096
)

# Run task
response = agent.run("Calculate 42 * 17", return_metrics=True)

print(f"Result: {response.result}")
print(f"Iterations: {response.metrics.total_iterations}")
print(f"Tokens: {response.metrics.total_tokens}")
print(f"Time: {response.metrics.execution_time_seconds}s")
```

### Single Agent (OpenAI)

```python
# Create agent with OpenAI
agent = create_gemma_agent(
    api_base="https://api.openai.com/v1",
    model="gpt-4",
    api_key="sk-your-api-key-here",
    temperature=0.5,
    max_tokens=1000,
    top_p=0.9,
    tools=get_default_tools()
)

# Use async version
response = await agent.arun("What is the weather today?")
print(response.result)
```

### Multi-Agent DAG

```python
from linus.agents.graph import AgentDAG, AgentNode, DAGExecutor

# Create agents
analyzer = create_gemma_agent(tools=tools)
processor = create_gemma_agent(tools=tools)
reporter = create_gemma_agent(tools=tools)

# Build workflow
dag = AgentDAG(name="DataPipeline")

dag.add_node(AgentNode(name="analyze", agent=analyzer, output_key="analysis"))
dag.add_node(AgentNode(name="process", agent=processor, output_key="processed"))
dag.add_node(AgentNode(name="report", agent=reporter, output_key="final_report"))

dag.add_edge("analyze", "process")
dag.add_edge("process", "report")

# Execute
executor = DAGExecutor(dag)
result = executor.execute(initial_state={"input": "Your data"})

print(f"Status: {result.status}")
print(f"Report: {result.final_state['final_report']}")
```

## 📚 Documentation

### Core Documentation
- **[API.md](API.md)** - Complete API reference
- **[QUICK_START.md](QUICK_START.md)** - Quick reference guide
- **[AGENT_ENHANCEMENTS.md](AGENT_ENHANCEMENTS.md)** - Architecture details

### Feature-Specific
- **[ASYNC.md](ASYNC.md)** - Async/await and parallel execution
- **[STATE_UNIFIED.md](STATE_UNIFIED.md)** - Unified SharedState management
- **[MEMORY.md](MEMORY.md)** - Memory system guide
- **[METRICS.md](METRICS.md)** - Metrics tracking
- **[DAG.md](DAG.md)** - DAG orchestration guide
- **[LOGGING.md](LOGGING.md)** - Logging configuration

## 🧪 Testing

```bash
# Run tests
python tests/test_agent_loop.py
python tests/test_agent_metrics.py
python tests/test_agent_memory.py
python tests/test_dag.py

# Run examples
python examples/dag_example.py
```

## 📁 Project Structure

```
agents/
├── src/linus/agents/
│   ├── agent/          # Core agent system
│   └── graph/          # DAG orchestration
├── tests/              # Test suites
├── examples/           # Usage examples
└── docs/              # Documentation
```

## 🤝 Contributing

Contributions welcome! Please submit a pull request.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/maai-solutions/agents/issues)
- **Documentation**: See `/docs` directory

---

**Version**: 1.0.0
