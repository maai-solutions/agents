"""Example: State compatibility between dict and SharedState."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.graph.state import SharedState
from loguru import logger


def example_dict_state():
    """Example 1: Traditional dict state."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Dict State (Traditional)")
    print("="*80)

    tools = get_default_tools()

    # Create agent with dict state
    state = {
        "user_id": 123,
        "session": "abc-def-123",
        "counter": 0
    }

    agent = create_gemma_agent(
        tools=tools,
        state=state,
        verbose=False,
        max_iterations=2
    )

    print(f"\n📊 Initial state: {agent.state}")

    # Access state
    print(f"User ID: {agent.state['user_id']}")
    print(f"Session: {agent.state['session']}")

    # Modify state
    agent.state["counter"] = agent.state["counter"] + 1
    agent.state["last_action"] = "example_completed"

    print(f"\n📊 Final state: {agent.state}")


def example_shared_state():
    """Example 2: SharedState with metadata and history."""
    print("\n" + "="*80)
    print("EXAMPLE 2: SharedState (Advanced)")
    print("="*80)

    tools = get_default_tools()

    # Create SharedState with metadata
    shared_state = SharedState()
    shared_state.set("user_id", 456, source="init", metadata={"priority": "high"})
    shared_state.set("session", "xyz-789", source="init")

    # Create agent with SharedState
    agent = create_gemma_agent(
        tools=tools,
        state=shared_state,
        verbose=False,
        max_iterations=2
    )

    print(f"\n📊 Agent uses StateWrapper: {type(agent.state).__name__}")

    # Use dict-like interface
    print(f"User ID: {agent.state['user_id']}")
    print(f"Session: {agent.state['session']}")

    # Modify through agent
    agent.state["counter"] = 1
    agent.state["last_action"] = "example_completed"

    print(f"\n📊 State keys: {agent.state.keys()}")
    print(f"📊 State items: {dict(agent.state.items())}")

    # Verify data in underlying SharedState
    print(f"\n✅ Data synchronized with SharedState:")
    print(f"   counter = {shared_state.get('counter')}")
    print(f"   last_action = {shared_state.get('last_action')}")

    # Access history (only available with SharedState)
    history = shared_state.get_history("user_id")
    print(f"\n📜 History for 'user_id': {len(history)} entries")
    for entry in history:
        print(f"   - {entry.value} (source: {entry.source}, time: {entry.timestamp})")


def example_multi_agent_sharing():
    """Example 3: Multiple agents sharing state."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Multi-Agent State Sharing")
    print("="*80)

    tools = get_default_tools()

    # Create shared state
    shared_state = SharedState()
    shared_state.set("task", "process_data", source="coordinator")

    # Create two agents sharing the same state
    agent1 = create_gemma_agent(
        tools=tools,
        state=shared_state,
        verbose=False,
        max_iterations=2
    )

    agent2 = create_gemma_agent(
        tools=tools,
        state=shared_state,
        verbose=False,
        max_iterations=2
    )

    print("\n🤖 Agent 1 sets data:")
    agent1.state["step1_complete"] = True
    agent1.state["step1_result"] = "Data validated"
    print(f"   step1_complete = {agent1.state['step1_complete']}")
    print(f"   step1_result = {agent1.state['step1_result']}")

    print("\n🤖 Agent 2 can see Agent 1's data:")
    print(f"   step1_complete = {agent2.state.get('step1_complete')}")
    print(f"   step1_result = {agent2.state.get('step1_result')}")

    print("\n🤖 Agent 2 adds its own data:")
    agent2.state["step2_complete"] = True
    agent2.state["step2_result"] = "Data processed"
    print(f"   step2_complete = {agent2.state['step2_complete']}")
    print(f"   step2_result = {agent2.state['step2_result']}")

    print("\n🤖 Agent 1 can see Agent 2's data:")
    print(f"   step2_complete = {agent1.state.get('step2_complete')}")
    print(f"   step2_result = {agent1.state.get('step2_result')}")

    print("\n📊 Both agents see complete state:")
    print(f"   Agent 1 keys: {sorted(agent1.state.keys())}")
    print(f"   Agent 2 keys: {sorted(agent2.state.keys())}")
    print(f"   Keys match: {agent1.state.keys() == agent2.state.keys()}")


def example_state_wrapper_features():
    """Example 4: StateWrapper advanced features."""
    print("\n" + "="*80)
    print("EXAMPLE 4: StateWrapper Advanced Features")
    print("="*80)

    tools = get_default_tools()
    shared_state = SharedState()

    agent = create_gemma_agent(
        tools=tools,
        state=shared_state,
        verbose=False,
        max_iterations=2
    )

    # Dict-like operations
    print("\n📝 Dict-like operations:")

    # __setitem__
    agent.state["name"] = "Alice"
    agent.state["age"] = 30
    agent.state["city"] = "San Francisco"

    # __getitem__
    print(f"   name = {agent.state['name']}")

    # get with default
    print(f"   email = {agent.state.get('email', 'not provided')}")

    # __contains__
    print(f"   'name' in state: {'name' in agent.state}")
    print(f"   'email' in state: {'email' in agent.state}")

    # Collection operations
    print("\n📊 Collection operations:")
    print(f"   keys(): {list(agent.state.keys())}")
    print(f"   values(): {list(agent.state.values())}")
    print(f"   items(): {dict(agent.state.items())}")
    print(f"   len(): {len(agent.state)}")

    # Batch update
    print("\n📦 Batch update:")
    agent.state.update({
        "status": "active",
        "last_login": "2024-01-01"
    })
    print(f"   keys after update: {list(agent.state.keys())}")

    # Error handling
    print("\n⚠️  Error handling:")
    try:
        value = agent.state["nonexistent"]
    except KeyError as e:
        print(f"   KeyError caught: {e}")

    safe_value = agent.state.get("nonexistent", "default_value")
    print(f"   get() with default: {safe_value}")


def example_backward_compatibility():
    """Example 5: Backward compatibility - same code works with both."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Backward Compatibility")
    print("="*80)

    def process_with_agent(agent):
        """Process data with agent - works with any state type."""
        # This code doesn't know if state is dict or SharedState
        agent.state["input_received"] = True

        data = agent.state.get("data", [])
        agent.state["data_count"] = len(data)

        agent.state.update({
            "processing_complete": True,
            "output": "processed result"
        })

        return agent.state.get("output")

    tools = get_default_tools()

    # Test with dict state
    print("\n🔹 Using dict state:")
    dict_state = {"data": [1, 2, 3, 4, 5]}
    agent_dict = create_gemma_agent(tools=tools, state=dict_state, verbose=False, max_iterations=2)
    result1 = process_with_agent(agent_dict)
    print(f"   Result: {result1}")
    print(f"   Data count: {agent_dict.state.get('data_count')}")

    # Test with SharedState
    print("\n🔹 Using SharedState:")
    shared = SharedState()
    shared.set("data", [1, 2, 3, 4, 5], source="init")
    agent_shared = create_gemma_agent(tools=tools, state=shared, verbose=False, max_iterations=2)
    result2 = process_with_agent(agent_shared)
    print(f"   Result: {result2}")
    print(f"   Data count: {agent_shared.state.get('data_count')}")

    print("\n✅ Same code works with both state types!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

    print("\n" + "="*80)
    print("STATE COMPATIBILITY EXAMPLES")
    print("="*80)

    try:
        example_dict_state()
        example_shared_state()
        example_multi_agent_sharing()
        example_state_wrapper_features()
        example_backward_compatibility()

        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETE")
        print("="*80)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
