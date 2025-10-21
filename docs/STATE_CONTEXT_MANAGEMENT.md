# State Context Management

This guide explains how to manage shared state context in prompts to prevent token overflow when working with large state dictionaries.

## Overview

When agents use shared state to pass information between steps, the entire state dictionary is typically included in the LLM prompt. For long-running workflows with many state entries, this can exceed the model's context window and cause errors.

The framework provides **three strategies** for managing state context, inspired by the `MemoryManager` pattern:

1. **FULL** - Include all state (no truncation)
2. **CLIP** - Keep only the most recent N entries
3. **COMPACT** - Use LLM to summarize state content

## State Context Strategies

### Strategy 1: FULL (Default)

Includes all state entries without any truncation. This is the default behavior.

**Use when:**
- State is small and fits within context limits
- You need complete visibility of all state
- Token usage is not a concern

**Example:**
```python
from linus.agents.graph.state import SharedState, StateContextStrategy

state = SharedState(context_strategy=StateContextStrategy.FULL)
state.set("result_1", "Task completed successfully")
state.set("result_2", "Generated report with 500 entries")

# Get full context
context = state.get_context(strategy=StateContextStrategy.FULL)
print(context)
# Output: All state entries as JSON
```

**Warning:** If state exceeds `max_context_tokens`, a warning will be logged but all content is still included.

---

### Strategy 2: CLIP

Keeps only the most recent state entries based on timestamp, dropping older entries until the context fits within the token limit.

**Use when:**
- Recent state is more relevant than old state
- You have many intermediate results
- You want deterministic truncation

**How it works:**
1. Sort state entries by timestamp (most recent first)
2. Add entries one by one until token limit is reached
3. Drop remaining older entries

**Example:**
```python
from linus.agents.graph.state import SharedState, StateContextStrategy

state = SharedState(
    max_context_tokens=500,  # Limit context to 500 tokens
    context_strategy=StateContextStrategy.CLIP
)

# Add multiple state entries
for i in range(20):
    state.set(f"step_{i}_result", f"Completed step {i} with result data...")

# Get clipped context (only recent entries that fit)
context = state.get_context(
    strategy=StateContextStrategy.CLIP,
    max_tokens=500
)

# Check statistics
stats = state.get_state_stats()
print(f"Kept entries that fit in {stats['max_context_tokens']} tokens")
```

**Output format:**
```
Shared state (most recent 5 entries): {
  "step_19_result": "...",
  "step_18_result": "...",
  ...
}
```

---

### Strategy 3: COMPACT

Uses an LLM to intelligently summarize the state content, preserving important information while reducing token usage.

**Use when:**
- State is large but all information is relevant
- You need semantic compression
- Recent entries aren't necessarily more important

**How it works:**
1. Check if full state exceeds token limit
2. If yes, call LLM to generate a concise summary
3. Cache summary for reuse
4. Fallback to CLIP if LLM fails

**Example:**
```python
from openai import AsyncOpenAI
from linus.agents.graph.state import SharedState, StateContextStrategy

# Create LLM client for summarization
llm_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

state = SharedState(
    max_context_tokens=500,
    llm_client=llm_client,
    model="gemma3:27b",
    context_strategy=StateContextStrategy.COMPACT
)

# Add large state
state.set("analysis_result", "Very long analysis with 5000 words...")
state.set("database_query", "SELECT * FROM ... (1000 rows)")
state.set("api_response", "Large JSON response with 200 fields...")

# Get compacted context (LLM will summarize)
context = state.get_context(
    strategy=StateContextStrategy.COMPACT,
    max_tokens=500
)

# Check statistics
stats = state.get_state_stats()
print(f"Reduced from {stats['total_tokens']} to {stats['summary_tokens']} tokens")
```

**Output format:**
```
Shared state (summarized):
The analysis completed with 5000 words covering key findings. Database query
returned 1000 rows from the main table. API response contained 200 fields with
user data and configuration settings.
```

---

## Configuration

### Basic Configuration

Configure state context management when creating a `SharedState` instance:

```python
from linus.agents.graph.state import SharedState, StateContextStrategy

state = SharedState(
    max_context_tokens=1000,           # Max tokens for state context
    summary_threshold_tokens=500,      # When to trigger summarization
    llm_client=your_llm_client,        # For COMPACT strategy
    model="gemma3:27b",                # Model for summarization
    encoding_name="cl100k_base",       # Tiktoken encoding
    context_strategy=StateContextStrategy.CLIP  # Default strategy
)
```

### Agent Configuration

Use the `StateConfig` class for agent-level configuration:

```python
from linus.agents.agent.config import AgentParams, StateConfig
from linus.agents.graph.state import StateContextStrategy

params = AgentParams(
    state_config=StateConfig(
        max_state_context_tokens=1000,
        state_context_strategy=StateContextStrategy.CLIP,
        summary_threshold_tokens=500
    )
)
```

### Environment Variables

You can also configure via environment variables (recommended for production):

```bash
# .env file
STATE_MAX_CONTEXT_TOKENS=1000
STATE_CONTEXT_STRATEGY=clip  # or 'full', 'compact'
STATE_SUMMARY_THRESHOLD=500
```

---

## Token Counting

The framework uses `tiktoken` for accurate token counting (same as OpenAI models).

**Supported encodings:**
- `cl100k_base` - GPT-4, GPT-3.5-turbo, text-embedding-ada-002
- `p50k_base` - Codex models
- `r50k_base` - GPT-3 models

**Fallback:** If tiktoken is unavailable, uses estimation (1 token ≈ 4 characters).

**Example:**
```python
state = SharedState()
text = "The quick brown fox jumps over the lazy dog"
tokens = state.count_tokens(text)
print(f"Tokens: {tokens}")  # Output: Tokens: 9
```

---

## Agent Integration

Agents automatically use the configured strategy when building prompts.

### ReasoningAgent

```python
from linus.agents.agent.factory import Agent
from linus.agents.graph.state import SharedState, StateContextStrategy

# Create state with strategy
state = SharedState(
    max_context_tokens=800,
    context_strategy=StateContextStrategy.CLIP
)

# Create agent
agent = Agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    state=state,
    use_async=True
)

# Run agent - state context is automatically managed
response = await agent.run("Analyze the data from previous steps")
```

### CoordinatorAgent

The `CoordinatorAgent` also supports state context strategies:

```python
from linus.agents.agent.coordinator_agent import CoordinatorAgent

coordinator = CoordinatorAgent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    available_agents=my_agents,
    state=state  # State with configured strategy
)

# Coordinator automatically uses strategy when building plan context
response = await coordinator.run("Coordinate multi-step workflow")
```

---

## State Statistics

Monitor state usage with `get_state_stats()`:

```python
stats = state.get_state_stats()
print(stats)
```

**Output:**
```python
{
    "total_entries": 10,
    "total_tokens": 2500,
    "has_summary": True,
    "summary_tokens": 450,
    "max_context_tokens": 1000,
    "history_length": 15,
    "utilization": 2.5  # 250% over limit
}
```

**Metrics:**
- `total_entries` - Number of state keys
- `total_tokens` - Full state token count
- `has_summary` - Whether summary exists
- `summary_tokens` - Summary token count
- `max_context_tokens` - Configured limit
- `history_length` - Total state changes
- `utilization` - Ratio of tokens to limit

---

## Best Practices

### 1. Choose the Right Strategy

| Scenario | Recommended Strategy | Reason |
|----------|---------------------|--------|
| Few state entries (<5) | FULL | Simple, no overhead |
| Many intermediate results | CLIP | Recent data more relevant |
| Large but important state | COMPACT | Preserves all semantics |
| Long-running workflows | CLIP or COMPACT | Prevents overflow |

### 2. Set Appropriate Token Limits

```python
# Leave headroom for other prompt components
memory_tokens = 1500        # For conversation history
state_tokens = 1000         # For shared state
execution_history = 500     # For task history
system_prompt = 300         # For instructions
# Total: 3300 tokens (safe for 4K context models)

state = SharedState(max_context_tokens=state_tokens)
```

### 3. Use COMPACT Sparingly

COMPACT strategy requires an extra LLM call, which adds:
- Latency (~1-3 seconds)
- Cost (API tokens)
- Potential errors (LLM unavailable)

**Tip:** Use CLIP for most cases. Reserve COMPACT for scenarios where semantic compression is critical.

### 4. Monitor Utilization

```python
stats = state.get_state_stats()
if stats.get('utilization', 0) > 0.8:
    logger.warning(f"State is at {stats['utilization']:.0%} capacity")
    # Consider switching to CLIP or COMPACT
```

### 5. Clear Old State

```python
# Clear state periodically in long-running workflows
if iteration > 50:
    state.clear()
    logger.info("Cleared state to prevent overflow")
```

### 6. Use Summary Caching

For COMPACT strategy, summaries are cached until `clear_summary()` is called:

```python
# Reuse summary
context1 = state.get_context(strategy=StateContextStrategy.COMPACT)
context2 = state.get_context(strategy=StateContextStrategy.COMPACT)  # Uses cached summary

# Force new summary
state.clear_summary()
context3 = state.get_context(strategy=StateContextStrategy.COMPACT)  # Generates new summary
```

---

## Advanced Usage

### Dynamic Strategy Selection

Switch strategies based on state size:

```python
def get_adaptive_context(state: SharedState) -> str:
    stats = state.get_state_stats()
    tokens = stats['total_tokens']
    max_tokens = stats.get('max_context_tokens', float('inf'))

    if tokens <= max_tokens:
        return state.get_context(strategy=StateContextStrategy.FULL)
    elif tokens <= max_tokens * 2:
        return state.get_context(strategy=StateContextStrategy.CLIP)
    else:
        return state.get_context(strategy=StateContextStrategy.COMPACT)
```

### Custom Token Limits Per Call

Override limits at call time:

```python
# Configured with 1000 tokens
state = SharedState(max_context_tokens=1000)

# Use different limit for this call
context = state.get_context(
    strategy=StateContextStrategy.CLIP,
    max_tokens=500  # Temporary override
)
```

### Multi-Agent Workflows

Different agents can use different strategies:

```python
# Coordinator uses COMPACT (sees summarized state)
coordinator_state = SharedState(
    max_context_tokens=800,
    context_strategy=StateContextStrategy.COMPACT,
    llm_client=llm
)

# Worker agents use CLIP (see recent results)
worker_state = SharedState(
    max_context_tokens=1200,
    context_strategy=StateContextStrategy.CLIP
)
```

---

## Troubleshooting

### Problem: "tiktoken not available" warning

**Solution:** Install tiktoken:
```bash
pip install tiktoken
```

### Problem: COMPACT strategy falls back to CLIP

**Cause:** LLM client not provided or summarization failed.

**Solution:**
```python
from openai import AsyncOpenAI

llm_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

state = SharedState(
    llm_client=llm_client,  # Required for COMPACT
    model="gemma3:27b",
    context_strategy=StateContextStrategy.COMPACT
)
```

### Problem: State still exceeds context window

**Cause:** Token limit too high or strategy not working.

**Solution:** Check strategy is enabled and reduce limits:
```python
stats = state.get_state_stats()
print(f"Utilization: {stats.get('utilization', 0):.1%}")

# Reduce limit
state.max_context_tokens = 500

# Or use more aggressive strategy
state.context_strategy = StateContextStrategy.COMPACT
```

### Problem: Summary quality is poor

**Cause:** Model not good at summarization or target tokens too low.

**Solution:**
1. Use a better model (GPT-4 > Gemma > smaller models)
2. Increase target tokens:
```python
state.max_context_tokens = 1000  # More room for details
```

---

## Examples

### Example 1: Multi-Step Data Pipeline

```python
from linus.agents.graph.state import SharedState, StateContextStrategy

state = SharedState(
    max_context_tokens=1000,
    context_strategy=StateContextStrategy.CLIP
)

# Step 1: Data ingestion
state.set("raw_data", "Loaded 10000 records from database", source="step_1")

# Step 2: Data cleaning
state.set("cleaned_data", "Removed 500 duplicates and null values", source="step_2")

# Step 3: Feature engineering
state.set("features", "Created 20 features from raw data", source="step_3")

# ... many more steps ...

# Final step: Only sees recent state due to CLIP strategy
context = state.get_context()
print(f"Context includes most recent {state.count_tokens(context)} tokens")
```

### Example 2: Long-Running Coordinator

```python
from linus.agents.agent.coordinator_agent import CoordinatorAgent
from openai import AsyncOpenAI

llm_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

state = SharedState(
    max_context_tokens=1500,
    llm_client=llm_client,
    model="gemma3:27b",
    context_strategy=StateContextStrategy.COMPACT
)

coordinator = CoordinatorAgent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    available_agents=agents,
    state=state
)

# Coordinator will automatically use COMPACT strategy to summarize
# state from all previous agent executions
result = await coordinator.run("Complete the 20-step data analysis workflow")
```

### Example 3: Testing All Strategies

See [examples/test_state_context_strategies.py](../examples/test_state_context_strategies.py) for a complete test suite demonstrating all three strategies.

---

## API Reference

### SharedState

#### Constructor
```python
SharedState(
    max_context_tokens: Optional[int] = None,
    summary_threshold_tokens: Optional[int] = None,
    llm_client: Optional[Any] = None,
    model: Optional[str] = None,
    encoding_name: str = "cl100k_base",
    context_strategy: StateContextStrategy = StateContextStrategy.FULL
)
```

#### Methods

**`get_context()`**
```python
def get_context(
    strategy: StateContextStrategy = StateContextStrategy.FULL,
    max_tokens: Optional[int] = None,
    include_summary: bool = True
) -> str
```
Returns formatted state context for prompt inclusion.

**`count_tokens()`**
```python
def count_tokens(text: str) -> int
```
Count tokens in text using tiktoken.

**`get_state_stats()`**
```python
def get_state_stats() -> Dict[str, Any]
```
Get state statistics including token usage.

**`clear_summary()`**
```python
def clear_summary() -> None
```
Clear cached summary (forces regeneration on next COMPACT call).

### StateContextStrategy

```python
class StateContextStrategy(str, Enum):
    FULL = "full"      # Include all state
    CLIP = "clip"      # Keep recent entries
    COMPACT = "compact"  # LLM summarization
```

### StateConfig

```python
class StateConfig(BaseModel):
    max_state_context_tokens: Optional[int] = None
    state_context_strategy: StateContextStrategy = StateContextStrategy.FULL
    summary_threshold_tokens: Optional[int] = None
```

---

## Related Documentation

- [Memory Management](./MEMORY.md) - Conversation history management
- [Telemetry](./TELEMETRY.md) - Monitoring token usage
- [Agent Configuration](./AGENT_CONFIG.md) - Complete configuration reference
- [CLAUDE.md](../CLAUDE.md) - Project overview and architecture

---

## Summary

State context management is essential for long-running multi-agent workflows. By using the appropriate strategy (FULL, CLIP, or COMPACT), you can ensure your agents have the context they need without exceeding token limits.

**Quick reference:**
- **FULL**: Use for small state (< 500 tokens)
- **CLIP**: Use for many intermediate results where recent data matters
- **COMPACT**: Use for large but semantically important state

Monitor utilization with `get_state_stats()` and adjust strategies as needed.
