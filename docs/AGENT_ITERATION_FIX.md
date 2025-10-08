# Agent Tool Iteration Fix

## Problem

The ReasoningAgent was prematurely stopping execution when a tool returned no results or empty data, even though it had identified alternative tools to try. This prevented the agent from being persistent and trying multiple approaches to complete a task.

### Specific Issues

1. **Early exit on insufficient info**: When `has_sufficient_info=false` in the reasoning phase, the agent would immediately exit without attempting any planned tasks.

2. **Conservative completion checking**: The completion check prompt didn't explicitly encourage trying alternative tools when one tool failed.

3. **Lack of guidance in reasoning**: The reasoning prompt didn't instruct the agent to plan alternative approaches when tools might fail.

## Solution

### 1. Updated Reasoning Prompt ([src/linus/agents/agent/reasoning_agent.py:123-162](src/linus/agents/agent/reasoning_agent.py#L123-L162))

Added explicit guidelines to encourage persistence:

```python
IMPORTANT GUIDELINES:
- Set "has_sufficient_info" to true if you can ATTEMPT the task, even if success is uncertain
- Only set "has_sufficient_info" to false if the request is completely unclear or missing critical parameters
- If a tool might not return results, plan alternative tools or approaches in subsequent tasks
- When previous tools returned no results, plan different tools or different queries
- Be persistent and creative in planning alternative approaches
```

### 2. Updated Completion Check Prompt ([src/linus/agents/agent/reasoning_agent.py:188-213](src/linus/agents/agent/reasoning_agent.py#L188-L213))

Added guidelines to prevent premature completion:

```python
IMPORTANT GUIDELINES:
- If a tool returned no results or empty data, the task is NOT complete unless all reasonable alternatives have been tried
- Suggest trying alternative tools or different query approaches when tools return no results
- Only mark as complete when the user's request has been successfully answered or when all reasonable attempts have been exhausted
- Be persistent: if one approach didn't work, plan to try another approach rather than giving up
```

### 3. Improved Early Exit Logic ([src/linus/agents/agent/reasoning_agent.py:292-303](src/linus/agents/agent/reasoning_agent.py#L292-L303))

Modified the run loop to allow execution even when `has_sufficient_info=false` if tasks are planned:

```python
# If no sufficient info and this is the first iteration, exit early
# But allow retries in subsequent iterations with updated context
if not reasoning_result.has_sufficient_info and iteration == 1:
    result = f"I need more information to complete this task. {reasoning_result.reasoning}"
    logger.warning(f"[RUN] Insufficient information: {reasoning_result.reasoning}")

    # Return immediately only if there are no tasks planned
    if not reasoning_result.tasks:
        return self._format_output(result)

    # If there are tasks planned, continue to execute them
    logger.info(f"[RUN] Proceeding with {len(reasoning_result.tasks)} planned tasks despite insufficient info flag")
```

The same fix was applied to the async version ([src/linus/agents/agent/reasoning_agent.py:498-509](src/linus/agents/agent/reasoning_agent.py#L498-L509)).

## Testing

Created [test_tool_iteration.py](../test_tool_iteration.py) to verify the fix:

### Test Setup
- Two tools: `empty_search` (returns no results) and `alternative_search` (returns results)
- User query: "What is quantum computing?"

### Expected Behavior
1. Agent tries `empty_search` first
2. Gets no results
3. Agent recognizes task is not complete
4. Agent tries `alternative_search` in next iteration
5. Task completes successfully

### Test Results
✅ **All tests passed:**
- Agent performed 3 iterations (not just 1)
- Agent tried both tools
- Task completed successfully
- Agent correctly tried alternative tool after first tool failed

## Impact

### Before Fix
- Agent would give up after first tool returned no results
- Users would see "I don't have enough information" even when alternative approaches existed
- Poor user experience and wasted potential

### After Fix
- Agent persists through multiple iterations
- Tries alternative tools and approaches
- Only gives up after exhausting reasonable options
- Better completion rates and user satisfaction

## Usage

No API changes required. The fix is transparent to existing code. The agent will now automatically:

1. Plan multiple tools/approaches in the reasoning phase
2. Execute them across multiple iterations
3. Only stop when truly complete or max_iterations reached

### Configuration

You can control iteration behavior with the `max_iterations` parameter:

```python
from linus.agents.agent.factory import Agent

agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    max_iterations=10,  # Allow up to 10 iterations (default)
    tools=[tool1, tool2, tool3],
    verbose=True
)
```

## Related Files

- [src/linus/agents/agent/reasoning_agent.py](../src/linus/agents/agent/reasoning_agent.py) - Main implementation
- [test_tool_iteration.py](../test_tool_iteration.py) - Test script
- [CLAUDE.md](../CLAUDE.md) - Project documentation

## Future Improvements

1. **Smart tool selection**: Use embeddings or tool usage history to prioritize which tools to try
2. **Dynamic replanning**: Allow agent to replan strategy mid-execution based on partial results
3. **Tool chaining**: Automatically chain related tools (e.g., search → summarize → format)
4. **Failure analysis**: Learn from failed tool calls to avoid similar failures in future iterations
