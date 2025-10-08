# Agent Metrics Documentation

## Overview

The ReasoningAgent now provides comprehensive metrics tracking for all executions, giving you insights into performance, resource usage, and execution patterns.

## Metrics Tracked

### Performance Metrics
- **`total_iterations`**: Number of reasoning-execution loop iterations
- **`execution_time_seconds`**: Total time taken to complete the task
- **`task_completed`**: Boolean indicating if the task was successfully completed
- **`iterations_to_completion`**: Number of iterations needed to complete (null if incomplete)

### LLM Usage Metrics
- **`llm_calls`**: Total number of calls made to the language model
- **`reasoning_calls`**: Number of reasoning phase calls
- **`completion_checks`**: Number of task completion validation calls
- **`avg_tokens_per_llm_call`**: Average tokens used per LLM call

### Token Usage Metrics
- **`total_input_tokens`**: Total input/prompt tokens consumed
- **`total_output_tokens`**: Total output/completion tokens generated
- **`total_tokens`**: Combined total of all tokens used

### Tool Execution Metrics
- **`tool_executions`**: Total number of tool calls attempted
- **`successful_tool_calls`**: Number of successful tool executions
- **`failed_tool_calls`**: Number of failed tool executions
- **`success_rate`**: Ratio of successful to total tool calls

## Usage

### Basic Usage with Metrics

```python
from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools

# Create agent
tools = get_default_tools()
agent = create_gemma_agent(tools=tools)

# Run with metrics (default)
response = agent.run("Calculate 42 * 17", return_metrics=True)

# Access metrics
print(f"Execution time: {response.metrics.execution_time_seconds}s")
print(f"Total tokens: {response.metrics.total_tokens}")
print(f"Iterations: {response.metrics.total_iterations}")

# Get full metrics dictionary
metrics_dict = response.metrics.to_dict()
print(metrics_dict)
```

### Running Without Metrics

```python
# For backward compatibility or when metrics aren't needed
result = agent.run("Calculate 42 * 17", return_metrics=False)
# Returns only the result string/object
```

### AgentResponse Structure

When `return_metrics=True`, the agent returns an `AgentResponse` object with:

```python
@dataclass
class AgentResponse:
    result: Union[str, BaseModel]           # The actual answer/result
    metrics: AgentMetrics                    # Performance and usage metrics
    execution_history: List[Dict[str, Any]] # Detailed execution log
    completion_status: Optional[Dict]        # Task completion validation results
```

### Accessing Execution History

```python
response = agent.run("Multi-step task...", return_metrics=True)

for step in response.execution_history:
    print(f"Iteration {step['iteration']}: {step['task']}")
    print(f"  Tool: {step['tool']}")
    print(f"  Status: {step['status']}")
    print(f"  Result: {step['result'][:100]}...")
```

### Exporting Metrics

```python
import json

response = agent.run("Your task", return_metrics=True)

# Export just metrics
with open('metrics.json', 'w') as f:
    json.dump(response.metrics.to_dict(), f, indent=2)

# Export full response
with open('full_response.json', 'w') as f:
    json.dump(response.to_dict(), f, indent=2, default=str)
```

## Metrics Examples

### Simple Task
```
📊 Performance Metrics:
  ⏱️  Execution Time: 2.145 seconds
  🔄 Total Iterations: 1
  ✅ Task Completed: True
  🎯 Iterations to Completion: 1

🤖 LLM Usage:
  📞 Total LLM Calls: 3
  🧠 Reasoning Calls: 1
  ✔️  Completion Checks: 1

🎫 Token Usage:
  📥 Input Tokens: 845
  📤 Output Tokens: 234
  📊 Total Tokens: 1,079
  📈 Avg Tokens/Call: 359.67

🛠️  Tool Usage:
  🔧 Tool Executions: 1
  ✅ Successful: 1
  ❌ Failed: 0
  📊 Success Rate: 100%
```

### Complex Multi-Step Task
```
📊 Performance Metrics:
  ⏱️  Execution Time: 8.756 seconds
  🔄 Total Iterations: 3
  ✅ Task Completed: True
  🎯 Iterations to Completion: 3

🤖 LLM Usage:
  📞 Total LLM Calls: 12
  🧠 Reasoning Calls: 3
  ✔️  Completion Checks: 3

🎫 Token Usage:
  📥 Input Tokens: 3,452
  📤 Output Tokens: 892
  📊 Total Tokens: 4,344
  📈 Avg Tokens/Call: 362.00

🛠️  Tool Usage:
  🔧 Tool Executions: 5
  ✅ Successful: 5
  ❌ Failed: 0
  📊 Success Rate: 100%
```

## Use Cases

### Performance Monitoring
```python
# Track performance over time
responses = []
for task in tasks:
    response = agent.run(task, return_metrics=True)
    responses.append({
        'task': task,
        'time': response.metrics.execution_time_seconds,
        'tokens': response.metrics.total_tokens,
        'iterations': response.metrics.total_iterations
    })

# Analyze average performance
avg_time = sum(r['time'] for r in responses) / len(responses)
avg_tokens = sum(r['tokens'] for r in responses) / len(responses)
```

### Cost Estimation
```python
# Estimate API costs based on token usage
response = agent.run(task, return_metrics=True)

# Example pricing (adjust to your provider)
INPUT_COST_PER_1K = 0.0001
OUTPUT_COST_PER_1K = 0.0002

cost = (
    (response.metrics.total_input_tokens / 1000) * INPUT_COST_PER_1K +
    (response.metrics.total_output_tokens / 1000) * OUTPUT_COST_PER_1K
)
print(f"Estimated cost: ${cost:.4f}")
```

### Debugging and Optimization
```python
response = agent.run(task, return_metrics=True)

# Check if task is taking too long
if response.metrics.execution_time_seconds > 10:
    print("⚠️ Task exceeded time threshold")

# Check if too many iterations
if response.metrics.total_iterations > 5:
    print("⚠️ Task required many iterations - may need simplification")

# Check tool success rate
if response.metrics.success_rate < 0.8:
    print("⚠️ Low tool success rate - check tool implementations")
```

## API Integration

When using the agent through the FastAPI interface, metrics are automatically included in the response:

```python
# In app.py
response = AgentResponse(
    query=request.query,
    response=result.result,
    reasoning=reasoning,
    tools_used=tools_used,
    execution_time=result.metrics.execution_time_seconds,
    metrics=result.metrics.to_dict(),  # Include full metrics
    timestamp=datetime.now().isoformat(),
    session_id=request.session_id
)
```

## Testing

Run the metrics test suite:

```bash
python tests/test_agent_metrics.py
```

This will:
- Test metrics collection across different task complexities
- Validate metric accuracy
- Compare performance across task types
- Export sample metrics to JSON files

## Notes

- Token counts are extracted from LLM response metadata when available
- If token metadata is unavailable, estimation is used (4 chars ≈ 1 token)
- Metrics tracking adds minimal overhead (<1% of execution time)
- All metrics are non-invasive and don't affect agent behavior
