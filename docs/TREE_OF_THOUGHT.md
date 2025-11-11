# Tree of Thought Agent

The Tree of Thought (ToT) agent implements a sophisticated multi-phase reasoning approach for complex task planning and execution. It's particularly effective for tasks that benefit from deliberate planning, reflection, and tool selection.

## Overview

The Tree of Thought agent follows a four-phase approach:

1. **Initial Thought Generation**: Analyzes the user's request and creates a detailed plan with tool selection
2. **Reflection Phase**: Critically evaluates the initial plan and refines it (optional)
3. **Tool Selection & Filtering**: Identifies and selects the most relevant tools for the task (optional)
4. **Execution Phase**: Executes the refined plan step-by-step with selected tools

## Key Features

### 1. Multi-Phase Reasoning

Unlike traditional agents that plan and execute in a single pass, ToT agents separate reasoning from execution:

- **Reasoning Phase**: Uses a dedicated reasoning model (can be different from execution model) to analyze the task
- **Reflection**: Self-critiques and improves the initial plan before execution
- **Execution Phase**: Uses the execution model to carry out the refined plan

### 2. Intelligent Tool Selection

The agent can automatically filter tools based on task requirements:

- Analyzes which tools are relevant for the specific task
- Reduces noise by only providing relevant tools to the execution phase
- Improves efficiency and reduces hallucination

### 3. Separate Reasoning Model

You can use a different model for reasoning vs execution:

```python
from linus.agents.agent.factory import TreeOfThought

# Use DeepSeek-R1 for reasoning, Gemma3 for execution
tot_agent = TreeOfThought(
    model="gemma3:27b",           # Fast execution
    reasoning_model="deepseek-r1", # Powerful reasoning
    reasoning_temperature=0.9,     # High creativity for planning
    temperature=0.5                # Low variance for execution
)
```

### 4. Configurable Features

- **Enable/disable reflection**: Trade speed for quality
- **Enable/disable tool filtering**: Use all tools or filter to relevant ones
- **Adjustable temperatures**: Different creativity levels for reasoning vs execution
- **Memory support**: Optional memory management for context persistence

## Usage

### Basic Usage

```python
import asyncio
from linus.agents.agent.factory import TreeOfThought
from linus.agents.agent.tools import get_default_tools

async def main():
    # Create ToT agent
    tot_agent = TreeOfThought(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=get_default_tools(),
        enable_reflection=True,
        enable_tool_filtering=True
    )

    # Run a complex query
    response = await tot_agent.run(
        "Calculate 42 * 17, then search for information about that number"
    )

    print(response.result)
    print(f"Execution time: {response.metrics.execution_time_seconds:.2f}s")
    print(f"Tool executions: {response.metrics.tool_executions}")

asyncio.run(main())
```

### Advanced: Separate Reasoning Model

```python
from linus.agents.agent.factory import TreeOfThought

# Use different models for different phases
tot_agent = TreeOfThought(
    # Execution model (fast)
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    temperature=0.5,

    # Reasoning model (powerful)
    reasoning_model="qwen2.5:32b",
    reasoning_temperature=0.9,

    # ToT features
    enable_reflection=True,
    enable_tool_filtering=True,
    max_reflection_depth=2,

    tools=get_default_tools()
)
```

### With Different API Endpoints

```python
from linus.agents.agent.factory import TreeOfThought

# Reasoning and execution on different endpoints
tot_agent = TreeOfThought(
    # Execution on local Ollama
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    api_key="not-needed",

    # Reasoning on cloud provider
    reasoning_api_base="https://api.openai.com/v1",
    reasoning_model="gpt-4",
    reasoning_api_key="sk-...",

    tools=get_default_tools()
)
```

### With Memory Management

```python
from linus.agents.agent.factory import TreeOfThought

tot_agent = TreeOfThought(
    model="gemma3:27b",
    tools=get_default_tools(),
    enable_memory=True,
    memory_backend="in_memory",
    max_context_tokens=4096,
    memory_context_ratio=0.3
)
```

## Configuration Parameters

### Core Parameters

- **`model`** (str): Model name for execution (e.g., "gemma3:27b", "gpt-4")
- **`api_base`** (str): API endpoint (default: "http://localhost:11434/v1")
- **`api_key`** (str): API key for authentication
- **`tools`** (List[BaseTool]): List of available tools
- **`verbose`** (bool): Enable verbose logging (default: True)

### ToT-Specific Parameters

- **`reasoning_model`** (str, optional): Model for reasoning phase (defaults to main model)
- **`reasoning_api_base`** (str, optional): API base for reasoning model
- **`reasoning_api_key`** (str, optional): API key for reasoning model
- **`enable_reflection`** (bool): Enable reflection phase (default: True)
- **`enable_tool_filtering`** (bool): Filter tools based on task (default: True)
- **`max_reflection_depth`** (int): Maximum reflection iterations (default: 2)
- **`reasoning_temperature`** (float): Temperature for reasoning (default: 0.8)

### LLM Parameters

- **`temperature`** (float): Sampling temperature for execution (default: 0.7)
- **`max_tokens`** (int, optional): Maximum tokens to generate
- **`top_p`** (float, optional): Nucleus sampling parameter
- **`top_k`** (int, optional): Top-k sampling (Ollama-specific)

### Memory Parameters

- **`enable_memory`** (bool): Enable memory management (default: False)
- **`memory_backend`** (str): Memory type ("in_memory" or "vector_store")
- **`max_context_tokens`** (int): Max context window size (default: 4096)
- **`memory_context_ratio`** (float): Memory context ratio (default: 0.3)

## Architecture

### Phase 1: Initial Thought Generation

The agent analyzes the user's request and generates an initial plan:

```json
{
  "reasoning": "The user wants to calculate a product and then search for it...",
  "planned_tools": ["calculator", "search"],
  "planned_steps": [
    {
      "step_number": 1,
      "description": "Calculate 42 * 17",
      "tool": "calculator",
      "rationale": "Need to compute the product first"
    },
    {
      "step_number": 2,
      "description": "Search for information about 714",
      "tool": "search",
      "rationale": "Use the result to search for information"
    }
  ],
  "confidence": 0.9
}
```

### Phase 2: Reflection (Optional)

The agent critically evaluates and refines the initial plan:

```json
{
  "refined_reasoning": "Initial plan is good, but we should also...",
  "refined_steps": [...],  // Improved steps
  "tool_names": ["calculator", "search"],
  "alternative_approaches": [
    "Could use a different calculation approach",
    "Consider caching the result"
  ]
}
```

### Phase 3: Tool Selection (Optional)

If tool filtering is enabled, only relevant tools are used:

```python
# Before filtering: 10 tools available
# After filtering: 2 tools selected (calculator, search)
```

### Phase 4: Execution

The agent executes the refined plan step-by-step:

1. For each step in the plan:
   - Generate tool arguments using LLM
   - Execute the tool
   - Collect results and citations
   - Update context for next step

2. Synthesize final response from all step results

## When to Use ToT Agent

### Best Use Cases

1. **Complex Multi-Step Tasks**: Tasks requiring careful planning across multiple tools
2. **Critical Decision Making**: When the quality of planning is paramount
3. **Creative Problem Solving**: Tasks requiring exploration of alternative approaches
4. **Tool-Heavy Workflows**: When selecting the right tools is critical to success

### Example Scenarios

- **Research Analysis**: "Research market trends, analyze data, and generate a report"
- **Multi-Stage Calculations**: "Calculate ROI across multiple scenarios and compare"
- **Data Pipeline**: "Extract data from file, transform it, and load into database"
- **Creative Writing**: "Generate story outline, develop characters, write chapters"

### When NOT to Use ToT Agent

1. **Simple Queries**: Single-step tasks where planning overhead isn't justified
2. **Time-Critical Tasks**: When speed is more important than plan quality
3. **Straightforward Tool Usage**: When the tool choice is obvious

For these cases, use the standard `ReasoningAgent` instead.

## Performance Considerations

### Reflection Impact

| Configuration | Speed | Plan Quality | Use Case |
|--------------|-------|--------------|----------|
| With Reflection | Slower | Higher | Critical tasks |
| Without Reflection | Faster | Good | Time-sensitive tasks |

### Tool Filtering Impact

| Configuration | Speed | Accuracy | Use Case |
|--------------|-------|----------|----------|
| With Filtering | Faster execution | Higher (less noise) | Many tools available |
| Without Filtering | Slower execution | Variable | Few tools or all needed |

### Model Selection Impact

Using a separate reasoning model:

**Benefits:**
- Use powerful reasoning model (e.g., GPT-4, DeepSeek-R1) for planning
- Use fast execution model (e.g., Gemma3) for tool execution
- Optimize cost/performance trade-off

**Trade-offs:**
- Requires access to multiple models
- May have increased latency for reasoning phase
- Potential cost implications for cloud models

## Comparison with Other Agents

### vs ReasoningAgent

| Feature | ToT Agent | ReasoningAgent |
|---------|-----------|----------------|
| Planning Depth | Deep (multi-phase) | Single-phase |
| Reflection | Yes (optional) | No |
| Tool Filtering | Yes (optional) | No |
| Separate Reasoning Model | Yes | No |
| Best For | Complex tasks | Standard tasks |
| Execution Speed | Slower (more thorough) | Faster |

### vs CoordinatorAgent

| Feature | ToT Agent | CoordinatorAgent |
|---------|-----------|------------------|
| Architecture | Single agent with phases | Multiple subagents |
| Tool Usage | Direct tool execution | Delegates to subagents |
| Planning Style | Tree-based reasoning | Sequential coordination |
| Best For | Complex single-agent tasks | Multi-agent orchestration |

## Examples

See [examples/example_tot_usage.py](../examples/example_tot_usage.py) for comprehensive examples including:

1. Basic ToT agent usage
2. Using separate reasoning models
3. Disabling reflection for speed
4. Disabling tool filtering
5. Feature comparison benchmarks

## Implementation Details

### Data Models

The ToT agent uses several Pydantic models defined in [tot.py](../src/linus/agents/agent/tot.py):

- **`ThoughtNode`**: Represents a single thought with reasoning and planned steps
- **`ReflectionResult`**: Contains refined reasoning and improved plan
- **`Citation`**: Tracks source documents for responses (when using retrieval tools)

### Logging

The agent uses structured logging with prefixes:

- `[TOT-RUN]`: Main execution flow
- `[TOT-THOUGHT]`: Initial thought generation
- `[TOT-REFLECT]`: Reflection phase
- `[TOT-FILTER]`: Tool filtering
- `[TOT-EXEC]`: Step execution
- `[TOT-ARGS]`: Tool argument generation

### Telemetry

All LLM calls and tool executions are traced for observability:

- Initial thought generation
- Reflection
- Tool argument generation
- Tool execution
- Final response synthesis

## Extending the ToT Agent

### Custom Tool Selection Logic

You can extend the tool filtering logic:

```python
from linus.agents.agent.tot import TreeOfThoughtAgent

class CustomToTAgent(TreeOfThoughtAgent):
    async def _filter_tools(self, tool_names: List[str]) -> List[BaseTool]:
        # Custom filtering logic
        filtered = await super()._filter_tools(tool_names)

        # Add additional filtering criteria
        filtered = [t for t in filtered if meets_custom_criteria(t)]

        return filtered
```

### Custom Reflection Logic

```python
class CustomToTAgent(TreeOfThoughtAgent):
    async def _reflect_on_thought(
        self,
        input_text: str,
        initial_thought: ThoughtNode
    ) -> ReflectionResult:
        # Add custom reflection logic
        result = await super()._reflect_on_thought(input_text, initial_thought)

        # Post-process reflection
        result.refined_steps = enhance_steps(result.refined_steps)

        return result
```

## Troubleshooting

### Issue: Slow Execution

**Causes:**
- Reflection enabled with high depth
- Too many tools (filtering disabled)
- Large context window

**Solutions:**
- Disable reflection: `enable_reflection=False`
- Enable tool filtering: `enable_tool_filtering=True`
- Use faster reasoning model
- Reduce max_reflection_depth

### Issue: Poor Plan Quality

**Causes:**
- Reflection disabled
- Low reasoning temperature
- Weak reasoning model

**Solutions:**
- Enable reflection: `enable_reflection=True`
- Increase reasoning_temperature: `reasoning_temperature=0.9`
- Use more powerful reasoning model
- Provide more context in the query

### Issue: Wrong Tools Selected

**Causes:**
- Tool descriptions unclear
- Reasoning model not understanding task
- Tool filtering too aggressive

**Solutions:**
- Improve tool descriptions
- Use more powerful reasoning model
- Disable tool filtering temporarily: `enable_tool_filtering=False`
- Provide more context in query

## Best Practices

1. **Use Clear Queries**: Provide detailed task descriptions for better planning
2. **Choose Appropriate Models**: Match model capabilities to task complexity
3. **Tune Temperatures**: Higher for reasoning (0.8-0.9), lower for execution (0.5-0.7)
4. **Monitor Metrics**: Track execution time and tool usage to optimize configuration
5. **Start Simple**: Begin with basic configuration, add features as needed
6. **Test Reflection**: Benchmark with/without reflection for your specific use case
7. **Leverage Tool Filtering**: Enable for large tool sets, disable for small sets

## References

- **Original Paper**: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)
- **Lightagent Implementation**: [src/lightagent/la_core.py](../src/lightagent/la_core.py)
- **Base Agent**: [src/linus/agents/agent/base.py](../src/linus/agents/agent/base.py)
- **Factory Functions**: [src/linus/agents/agent/factory.py](../src/linus/agents/agent/factory.py)
