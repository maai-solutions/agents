# CoordinatorAgent Documentation

## Overview

The `CoordinatorAgent` is a sophisticated orchestration agent that coordinates multiple specialized subagents to accomplish complex tasks. Unlike the `ReasoningAgent` which directly executes tools, the `CoordinatorAgent` delegates work to specialized subagents, each with their own capabilities and tools.

## Key Features

### 1. **Multi-Agent Orchestration**
- Manages multiple specialized subagents
- Routes tasks to the most appropriate subagent
- Coordinates dependencies between tasks

### 2. **Adaptive Planning**
- Creates high-level execution plans based on task requirements
- Evaluates progress after each step
- Recalculates plans when needed based on results

### 3. **Plan Evaluation and Recalculation**
After each step execution, the coordinator:
- Evaluates if the current plan is still valid
- Determines if the task is complete
- Decides whether to continue, replan, or mark as complete
- Adapts to unexpected results or failures

### 4. **Hierarchical Tracing**
- Full telemetry support with hierarchical tracing
- Traces coordinator operations, subagent executions, and LLM calls
- Integration with Langfuse and OpenTelemetry

### 5. **Shared State and Memory**
- Maintains shared state across subagents
- Optional memory management for context persistence
- Context enrichment for subagent inputs

## Architecture

### Core Components

```
CoordinatorAgent
├── Planning Phase: Creates execution plan
├── Execution Phase: Runs plan steps using subagents
└── Evaluation Phase: Assesses progress and decides next action
```

### Workflow

```
┌─────────────────────────────────────────────────────────┐
│                    User Request                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              1. Planning Phase                          │
│  • Analyze request and context                          │
│  • Identify available subagents                         │
│  • Create step-by-step execution plan                   │
│  • Assign each step to appropriate subagent             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              2. Execution Phase                         │
│  • Execute each step in plan                            │
│  • Check dependencies before execution                  │
│  • Pass enriched context to subagent                    │
│  • Collect results from each step                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              3. Evaluation Phase                        │
│  • Evaluate progress and results                        │
│  • Determine if plan is still valid                     │
│  • Check if task is complete                            │
│  • Decide: continue / replan / complete                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
                ┌──────┴───────┐
                │ Complete?    │
                └──┬─────────┬─┘
                   │         │
                Yes│         │No (replan)
                   │         │
                   ▼         └─────┐
              ┌────────┐           │
              │ Return │           │
              │ Result │           │
              └────────┘           │
                                   │
              Loop back to Planning◄┘
```

### SubAgent Wrapper

Each subagent is wrapped in a `SubAgent` class that provides:
- **agent**: The actual agent instance
- **name**: Identifier for the subagent
- **description**: What the subagent does
- **capabilities**: List of capabilities/tools

## Usage

### Basic Example

```python
import asyncio
from linus.agents.agent.factory import Agent, Coordinator
from linus.agents.agent.coordinator_agent import SubAgent
from linus.agents.agent.tools import SearchTool, CalculatorTool

async def main():
    # Create specialized subagents
    research_agent = Agent(
        model="gemma3:27b",
        tools=[SearchTool()],
        agent_name="researcher"
    )

    calculator_agent = Agent(
        model="gemma3:27b",
        tools=[CalculatorTool()],
        agent_name="calculator"
    )

    # Wrap as SubAgents
    subagents = [
        SubAgent(
            agent=research_agent,
            name="researcher",
            description="Searches for information",
            capabilities=["search", "web_research"]
        ),
        SubAgent(
            agent=calculator_agent,
            name="calculator",
            description="Performs calculations",
            capabilities=["math", "calculator"]
        )
    ]

    # Create coordinator
    coordinator = Coordinator(
        model="gemma3:27b",
        subagents=subagents,
        verbose=True,
        max_iterations=15
    )

    # Execute complex task
    response = await coordinator.run(
        "Search for the current world population, "
        "then calculate what 10% of that number is",
        return_metrics=True
    )

    print(response.result)
    print(f"Completed in {response.metrics.total_iterations} iterations")

asyncio.run(main())
```

### Factory Function

```python
from linus.agents.agent.factory import Coordinator

coordinator = Coordinator(
    api_base="http://localhost:11434/v1",  # Ollama endpoint
    model="gemma3:27b",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
    subagents=subagents,
    tools=[],  # Optional tools for coordinator itself
    verbose=True,
    max_iterations=15,
    enable_memory=True,
    memory_backend="in_memory",
    use_async=True,
    tracer=tracer,  # Optional telemetry
    session_id="session-123",  # Optional session ID
    agent_name="my_coordinator"
)
```

### Configuration Options

#### Core Parameters

- **`model`** (str): LLM model to use (e.g., "gemma3:27b", "gpt-4")
- **`subagents`** (List[SubAgent]): List of subagents to coordinate
- **`max_iterations`** (int, default: 15): Maximum plan-execute-evaluate loops
- **`verbose`** (bool): Enable detailed logging
- **`agent_name`** (str): Name for hierarchical tracing

#### LLM Parameters

- **`temperature`** (float, 0.0-2.0): Sampling temperature
- **`max_tokens`** (int): Maximum tokens to generate
- **`top_p`** (float): Nucleus sampling parameter
- **`top_k`** (int): Top-k sampling (Ollama-specific)
- **`use_json_format`** (bool): Force JSON response format

#### Memory and State

- **`enable_memory`** (bool): Enable memory management
- **`memory_backend`** (str): "in_memory" or "vector_store"
- **`memory_context_ratio`** (float): Ratio of context for memory
- **`state`** (SharedState): Shared state across subagents

#### I/O Schemas

- **`input_schema`** (BaseModel): Pydantic model for input validation
- **`output_schema`** (BaseModel): Pydantic model for output
- **`output_key`** (str): Key to save output in shared state

## Response Format

The `coordinator.run()` method returns an `AgentResponse` (when `return_metrics=True`):

```python
{
    "result": "The final answer",  # String or Pydantic model
    "metrics": {
        "total_iterations": 3,
        "total_tokens": 5000,
        "execution_time_seconds": 12.5,
        "llm_calls": 9,
        "task_completed": true
    },
    "execution_history": [
        {
            "step_number": 1,
            "subagent": "researcher",
            "description": "Search for world population",
            "status": "completed",
            "result": "The current world population is..."
        },
        {
            "step_number": 2,
            "subagent": "calculator",
            "description": "Calculate 10% of population",
            "status": "completed",
            "result": "The result is..."
        }
    ],
    "completion_status": {
        "task_completed": true,
        "plan_still_valid": true,
        "evaluation_reasoning": "Task completed successfully",
        "next_action": "complete",
        "completion_summary": "Successfully found and calculated..."
    }
}
```

## Planning Format

The coordinator creates plans in this JSON format:

```json
{
    "reasoning": "Analysis of the task and planning rationale",
    "overall_goal": "What the complete plan should achieve",
    "plan": [
        {
            "step_number": 1,
            "description": "What this step accomplishes",
            "assigned_subagent": "researcher",
            "input": "Information to provide to the subagent",
            "expected_output": "What this step should produce",
            "dependencies": []
        },
        {
            "step_number": 2,
            "description": "Next step description",
            "assigned_subagent": "calculator",
            "input": "Use result from step 1",
            "expected_output": "Final calculation",
            "dependencies": [1]
        }
    ]
}
```

## Evaluation Format

After each execution phase, the coordinator evaluates:

```json
{
    "task_completed": false,
    "plan_still_valid": true,
    "evaluation_reasoning": "Step 1 completed but need more steps",
    "next_action": "continue",  // or "replan" or "complete"
    "suggested_changes": "If replanning, what should change",
    "completion_summary": null  // Only if complete
}
```

### Next Actions

- **`continue`**: Plan is valid, continue with next steps
- **`replan`**: Plan needs adjustment, create new plan
- **`complete`**: Task is finished, return result

## Advanced Features

### Dependency Management

Steps can declare dependencies on previous steps:

```python
{
    "step_number": 3,
    "dependencies": [1, 2],  # Requires steps 1 and 2 to complete
    "description": "Combine results from steps 1 and 2"
}
```

The coordinator automatically:
- Checks dependencies before executing steps
- Skips steps with unmet dependencies
- Tracks completed steps across iterations

### Context Enrichment

Each subagent receives enriched input:

```
Original request: [User's original query]

Current step: [Description of what this step should accomplish]

[Step-specific input from plan]

Previous results:
- Step 1: [Previous result summary]
- Step 2: [Previous result summary]
```

### Plan Recalculation

The coordinator recalculates the plan when:
- A step fails or produces unexpected results
- New information reveals the plan is insufficient
- Evaluation determines the plan is no longer valid

Example scenario:
1. Initial plan: Search → Calculate → Summarize
2. Search returns no results
3. Evaluation detects failure
4. Coordinator replans: Try different search → Calculate → Summarize

### Memory Integration

When memory is enabled:

```python
coordinator = Coordinator(
    model="gemma3:27b",
    subagents=subagents,
    enable_memory=True,
    memory_backend="in_memory",
    memory_context_ratio=0.3
)
```

The coordinator:
- Stores conversation history in memory
- Retrieves relevant context for planning
- Passes memory context to subagents
- Maintains memory across multiple runs

## Telemetry and Tracing

### Hierarchical Tracing

The coordinator creates a hierarchical trace structure:

```
coordinator.run
├── coordinator.planning (LLM call)
├── coordinator.step_execution
│   ├── subagent.researcher.run
│   │   ├── agent.reasoning (LLM call)
│   │   └── tool.search
│   └── subagent.calculator.run
│       ├── agent.reasoning (LLM call)
│       └── tool.calculator
├── coordinator.evaluation (LLM call)
└── coordinator.final_formatting (LLM call)
```

### With Langfuse

```python
from linus.agents.telemetry import initialize_telemetry

tracer = initialize_telemetry(
    service_name="coordinator-app",
    exporter_type="langfuse",
    enabled=True
)

coordinator = Coordinator(
    model="gemma3:27b",
    subagents=subagents,
    tracer=tracer,
    session_id="user-123"
)

response = await coordinator.run("Complex task...")

# View traces in Langfuse dashboard
tracer.flush()
```

### Trace Subagent Execution

Each subagent execution is traced automatically:

```python
async with self.telemetry.trace_subagent_execution(
    subagent_name, enriched_input
):
    result = await subagent.agent.run(enriched_input)
```

## Best Practices

### 1. Design Specialized Subagents

Create subagents with focused responsibilities:

```python
# ✅ Good: Focused capabilities
search_agent = Agent(tools=[SearchTool()])
calc_agent = Agent(tools=[CalculatorTool()])
file_agent = Agent(tools=[FileReaderTool()])

# ❌ Bad: Too many responsibilities in one agent
kitchen_sink_agent = Agent(tools=get_all_possible_tools())
```

### 2. Provide Clear Descriptions

Help the coordinator make good routing decisions:

```python
SubAgent(
    agent=research_agent,
    name="researcher",
    description="Searches for information using web search. "
                "Best for finding current facts, statistics, and online resources.",
    capabilities=["web_search", "information_retrieval", "fact_finding"]
)
```

### 3. Set Appropriate Max Iterations

Consider task complexity when setting `max_iterations`:

```python
# Simple tasks: 5-10 iterations
coordinator = Coordinator(subagents=subagents, max_iterations=10)

# Complex tasks with potential replanning: 15-20 iterations
coordinator = Coordinator(subagents=subagents, max_iterations=20)
```

### 4. Handle Failures Gracefully

The coordinator automatically handles failures, but you can improve resilience:

```python
# Add a fallback general-purpose agent
subagents = [
    SubAgent(agent=specialized_agent1, ...),
    SubAgent(agent=specialized_agent2, ...),
    SubAgent(
        agent=Agent(tools=get_default_tools()),
        name="fallback",
        description="General-purpose assistant for tasks that don't fit other agents",
        capabilities=["general", "fallback", "flexible"]
    )
]
```

### 5. Use Shared State for Context

Share data between subagents via shared state:

```python
from linus.agents.graph.state import SharedState

state = SharedState()

# All subagents and coordinator share this state
coordinator = Coordinator(
    subagents=subagents,
    state=state
)

# Subagents can read/write shared data
state.set("user_preference", "detailed", source="agent")
```

### 6. Enable Verbose for Debugging

During development, use verbose mode:

```python
coordinator = Coordinator(
    subagents=subagents,
    verbose=True  # See detailed logs
)
```

## Examples

### Example 1: Research and Analysis

```python
async def research_and_analyze():
    research_agent = Agent(
        model="gemma3:27b",
        tools=[SearchTool()],
        agent_name="researcher"
    )

    analyst_agent = Agent(
        model="gemma3:27b",
        tools=[CalculatorTool()],
        agent_name="analyst"
    )

    subagents = [
        SubAgent(
            agent=research_agent,
            name="researcher",
            description="Finds information through web search",
            capabilities=["search", "research"]
        ),
        SubAgent(
            agent=analyst_agent,
            name="analyst",
            description="Analyzes data and performs calculations",
            capabilities=["analysis", "math"]
        )
    ]

    coordinator = Coordinator(
        model="gemma3:27b",
        subagents=subagents,
        verbose=True
    )

    response = await coordinator.run(
        "Find the GDP of the US and China, "
        "then calculate the difference between them",
        return_metrics=True
    )

    return response
```

### Example 2: File Processing Pipeline

```python
async def process_files():
    file_reader = Agent(
        model="gemma3:27b",
        tools=[FileReaderTool()],
        agent_name="reader"
    )

    data_processor = Agent(
        model="gemma3:27b",
        tools=[CalculatorTool()],
        agent_name="processor"
    )

    summarizer = Agent(
        model="gemma3:27b",
        tools=[],
        agent_name="summarizer"
    )

    subagents = [
        SubAgent(
            agent=file_reader,
            name="reader",
            description="Reads and extracts data from files",
            capabilities=["file_reading", "extraction"]
        ),
        SubAgent(
            agent=data_processor,
            name="processor",
            description="Processes and analyzes extracted data",
            capabilities=["data_processing", "analysis"]
        ),
        SubAgent(
            agent=summarizer,
            name="summarizer",
            description="Creates summaries and reports",
            capabilities=["summarization", "reporting"]
        )
    ]

    coordinator = Coordinator(
        model="gemma3:27b",
        subagents=subagents,
        max_iterations=20
    )

    response = await coordinator.run(
        "Read data.csv, calculate the average of the 'score' column, "
        "and create a summary report",
        return_metrics=True
    )

    return response
```

### Example 3: Multi-Step Research

```python
async def complex_research():
    search_agent = Agent(
        model="gemma3:27b",
        tools=[SearchTool()],
        agent_name="searcher"
    )

    calculator_agent = Agent(
        model="gemma3:27b",
        tools=[CalculatorTool()],
        agent_name="calculator"
    )

    writer_agent = Agent(
        model="gemma3:27b",
        tools=[],
        agent_name="writer"
    )

    subagents = [
        SubAgent(
            agent=search_agent,
            name="searcher",
            description="Searches for information online",
            capabilities=["search", "web_research"]
        ),
        SubAgent(
            agent=calculator_agent,
            name="calculator",
            description="Performs mathematical calculations",
            capabilities=["math", "calculations"]
        ),
        SubAgent(
            agent=writer_agent,
            name="writer",
            description="Writes summaries and explanations",
            capabilities=["writing", "summarization"]
        )
    ]

    coordinator = Coordinator(
        model="gemma3:27b",
        subagents=subagents,
        verbose=True,
        max_iterations=15
    )

    response = await coordinator.run(
        "Research the Fibonacci sequence, calculate the 15th number, "
        "and write a brief explanation of why it's significant",
        return_metrics=True
    )

    return response
```

## Troubleshooting

### Issue: Coordinator creates invalid plans

**Solution**: Ensure subagent descriptions are clear and specific:

```python
# ✅ Good
SubAgent(
    agent=agent,
    name="calculator",
    description="Performs arithmetic operations: addition, subtraction, "
                "multiplication, division, and evaluates mathematical expressions",
    capabilities=["math", "arithmetic", "calculator"]
)

# ❌ Bad
SubAgent(
    agent=agent,
    name="calculator",
    description="Does stuff",
    capabilities=["stuff"]
)
```

### Issue: Too many iterations without completion

**Solution**:
1. Increase `max_iterations` for complex tasks
2. Check if subagents are returning useful results
3. Review evaluation logic in logs

### Issue: Subagent execution fails

**Solution**:
1. Test subagents individually first
2. Check subagent tools are properly configured
3. Enable verbose logging to see detailed errors

### Issue: Plan doesn't recalculate when needed

**Solution**: The evaluation phase determines when to replan. Ensure:
1. Subagents return informative error messages
2. Max iterations is sufficient for replanning attempts
3. Check evaluation reasoning in execution history

## Performance Considerations

### Token Usage

The coordinator makes multiple LLM calls:
- Planning: ~500-1500 tokens per iteration
- Evaluation: ~300-800 tokens per iteration
- Subagent executions: Varies by subagent
- Final formatting: ~200-500 tokens

**Optimization tips**:
- Use `max_tokens` to limit generation
- Enable `memory_context_ratio` < 0.3 to reduce context
- Use smaller models for subagents if appropriate

### Execution Time

Execution time depends on:
- Number of plan steps
- Subagent execution time
- Number of replanning iterations
- Network latency (for API-based LLMs)

**Optimization tips**:
- Run independent steps in parallel (future feature)
- Use local models (Ollama) for faster response
- Set reasonable `max_iterations`

## Comparison: CoordinatorAgent vs ReasoningAgent

| Feature | CoordinatorAgent | ReasoningAgent |
|---------|------------------|----------------|
| **Purpose** | Orchestrate multiple agents | Execute tools directly |
| **Complexity** | Complex multi-agent workflows | Single-agent reasoning |
| **Tools** | Tools via subagents | Direct tool access |
| **Planning** | High-level plan with steps | Task-level reasoning |
| **Evaluation** | After each step | After all tasks |
| **Best For** | Complex, multi-domain tasks | Focused, single-domain tasks |
| **Overhead** | Higher (more LLM calls) | Lower (fewer LLM calls) |

**When to use CoordinatorAgent**:
- Task requires multiple specialized capabilities
- Need adaptive planning and replanning
- Want to compose existing agents
- Building complex multi-step workflows

**When to use ReasoningAgent**:
- Task can be solved with a single set of tools
- Need faster execution
- Building a specialized agent for one domain
- Want simpler architecture

## See Also

- [ReasoningAgent Documentation](./REASONING_AGENT.md)
- [Hierarchical Tracing Guide](./HIERARCHICAL_TRACING.md)
- [Telemetry Documentation](./TELEMETRY.md)
- [Langfuse Integration](./LANGFUSE_INTEGRATION.md)
