"""Test and demonstrate state context management strategies."""

import asyncio
from openai import AsyncOpenAI
from linus.agents.graph.state import SharedState, StateContextStrategy
from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from loguru import logger


def create_large_shared_state(llm_client=None, model="gemma3:27b") -> SharedState:
    """Create a shared state with many entries to test strategies."""
    state = SharedState(
        max_context_tokens=500,  # Small limit to force strategies
        llm_client=llm_client,
        model=model,
        context_strategy=StateContextStrategy.FULL
    )

    # Add multiple state entries
    state.set("task_1_result", "Completed data validation with 1500 records processed successfully", source="agent_1")
    state.set("task_2_result", "Generated comprehensive report with 50 charts and graphs showing trends", source="agent_2")
    state.set("task_3_result", "Analyzed customer feedback from 3000 reviews using sentiment analysis", source="agent_3")
    state.set("user_preferences", {"language": "en", "format": "json", "detailed": True}, source="coordinator")
    state.set("database_connection", "postgresql://localhost:5432/mydb", source="system")
    state.set("api_endpoint", "https://api.example.com/v1/data", source="config")
    state.set("last_sync_time", "2025-01-20T10:30:00Z", source="sync_service")
    state.set("active_users", 1250, source="analytics")
    state.set("error_log", ["Error 1: Connection timeout", "Error 2: Invalid credentials", "Error 3: Rate limit exceeded"], source="system")
    state.set("feature_flags", {"new_ui": True, "beta_features": False, "analytics": True}, source="config")

    return state


async def test_full_strategy():
    """Test FULL strategy (include all state)."""
    print("\n" + "=" * 80)
    print("TEST 1: FULL STRATEGY (No Truncation)")
    print("=" * 80)

    state = create_large_shared_state()
    state.context_strategy = StateContextStrategy.FULL

    # Get context
    context = state.get_context(strategy=StateContextStrategy.FULL, max_tokens=500)

    # Show statistics
    stats = state.get_state_stats()
    print(f"\nState Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Max allowed: {stats['max_context_tokens']}")
    if 'utilization' in stats:
        print(f"  Utilization: {stats['utilization']:.1%}")

    print(f"\nGenerated Context Preview (first 500 chars):")
    print(context[:500] + "...")


async def test_clip_strategy():
    """Test CLIP strategy (keep most recent entries)."""
    print("\n" + "=" * 80)
    print("TEST 2: CLIP STRATEGY (Most Recent Entries)")
    print("=" * 80)

    state = create_large_shared_state()
    state.context_strategy = StateContextStrategy.CLIP

    # Get context with CLIP
    context = state.get_context(strategy=StateContextStrategy.CLIP, max_tokens=500)

    # Show statistics
    stats = state.get_state_stats()
    print(f"\nState Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Full state tokens: {stats['total_tokens']}")
    print(f"  Max allowed: {stats['max_context_tokens']}")

    context_tokens = state.count_tokens(context)
    print(f"  Clipped context tokens: {context_tokens}")
    print(f"  Reduction: {(1 - context_tokens / stats['total_tokens']):.1%}")

    print(f"\nGenerated Context Preview (first 500 chars):")
    print(context[:500] + "...")


async def test_compact_strategy():
    """Test COMPACT strategy (LLM summarization)."""
    print("\n" + "=" * 80)
    print("TEST 3: COMPACT STRATEGY (LLM Summarization)")
    print("=" * 80)

    # Create LLM client for summarization
    llm_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed"
    )

    state = create_large_shared_state(llm_client=llm_client, model="gemma3:27b")
    state.context_strategy = StateContextStrategy.COMPACT

    # Get context with COMPACT
    print("\nCalling LLM to summarize state...")
    context = state.get_context(strategy=StateContextStrategy.COMPACT, max_tokens=500)

    # Show statistics
    stats = state.get_state_stats()
    print(f"\nState Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Full state tokens: {stats['total_tokens']}")
    print(f"  Max allowed: {stats['max_context_tokens']}")
    print(f"  Has summary: {stats['has_summary']}")
    print(f"  Summary tokens: {stats['summary_tokens']}")

    if stats['summary_tokens'] > 0:
        print(f"  Reduction: {(1 - stats['summary_tokens'] / stats['total_tokens']):.1%}")

    print(f"\nGenerated Context:")
    print(context)


async def test_agent_with_state_strategies():
    """Test agent integration with different state strategies."""
    print("\n" + "=" * 80)
    print("TEST 4: AGENT INTEGRATION")
    print("=" * 80)

    # Create shared state with CLIP strategy
    llm_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed"
    )

    state = SharedState(
        max_context_tokens=800,
        llm_client=llm_client,
        model="gemma3:27b",
        context_strategy=StateContextStrategy.CLIP
    )

    # Populate state
    state.set("step_1_result", "Analyzed 5000 data points and found 3 anomalies", source="step_1")
    state.set("step_2_result", "Generated visualization charts for all anomalies", source="step_2")
    state.set("step_3_result", "Created detailed report with recommendations", source="step_3")
    state.set("config", {"threshold": 0.95, "model": "xgboost"}, source="coordinator")

    # Create agent with this state
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=get_default_tools(),
        state=state,
        verbose=True,
        use_async=True
    )

    print("\nAgent created with CLIP strategy on shared state")
    print(f"State has {len(state._state)} entries")
    print(f"State will be automatically clipped to fit {state.max_context_tokens} tokens\n")

    # Run agent (state context will be automatically managed)
    print("Running agent query...")
    response = await agent.run("What is the sum of 100 + 250?")

    print(f"\nAgent Response: {response.result}")
    print(f"Metrics: {response.metrics}")


async def compare_strategies():
    """Compare all strategies side by side."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    llm_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed"
    )

    state = create_large_shared_state(llm_client=llm_client)

    results = {}

    # Test each strategy
    for strategy in [StateContextStrategy.FULL, StateContextStrategy.CLIP, StateContextStrategy.COMPACT]:
        context = state.get_context(strategy=strategy, max_tokens=500)
        tokens = state.count_tokens(context)
        results[strategy.value] = {
            "tokens": tokens,
            "context_length": len(context)
        }

    # Print comparison
    stats = state.get_state_stats()
    print(f"\nOriginal state: {stats['total_tokens']} tokens, {stats['total_entries']} entries")
    print(f"\nStrategy Results:")
    print(f"  {'Strategy':<15} {'Tokens':<10} {'Context Length':<15} {'Reduction':<10}")
    print(f"  {'-' * 60}")

    for strategy_name, data in results.items():
        reduction = (1 - data['tokens'] / stats['total_tokens']) * 100 if stats['total_tokens'] > 0 else 0
        print(f"  {strategy_name:<15} {data['tokens']:<10} {data['context_length']:<15} {reduction:>5.1f}%")


async def main():
    """Run all tests."""
    print("\n" + "#" * 80)
    print("# STATE CONTEXT MANAGEMENT STRATEGIES - TEST SUITE")
    print("#" * 80)

    try:
        # Test individual strategies
        await test_full_strategy()
        await test_clip_strategy()
        await test_compact_strategy()

        # Compare strategies
        await compare_strategies()

        # Test agent integration
        await test_agent_with_state_strategies()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        print(f"\nERROR: {e}")
        print("\nMake sure Ollama is running with gemma3:27b model:")
        print("  ollama run gemma3:27b")


if __name__ == "__main__":
    asyncio.run(main())
