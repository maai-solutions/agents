"""Test state compatibility between agent.py and graph state.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.agent import create_gemma_agent, StateWrapper
from linus.agents.agent.tools import get_default_tools
from linus.agents.graph.state import SharedState
from loguru import logger


def test_dict_state():
    """Test agent with traditional dict state."""
    print("\n" + "="*80)
    print("TEST 1: Agent with Dict State")
    print("="*80)

    tools = get_default_tools()
    state = {"initial_value": 42}

    agent = create_gemma_agent(
        tools=tools,
        state=state,
        verbose=False,
        max_iterations=2
    )

    # Verify state is accessible
    assert agent.state["initial_value"] == 42
    print("✅ Dict state initialized correctly")

    # Modify state
    agent.state["new_value"] = 100
    assert agent.state["new_value"] == 100
    print("✅ Dict state modification works")

    print("\n✅ Dict state test PASSED\n")


def test_shared_state():
    """Test agent with SharedState from graph module."""
    print("="*80)
    print("TEST 2: Agent with SharedState")
    print("="*80)

    tools = get_default_tools()
    shared_state = SharedState()
    shared_state.set("initial_value", 42, source="test")

    agent = create_gemma_agent(
        tools=tools,
        state=shared_state,
        verbose=False,
        max_iterations=2
    )

    # Verify StateWrapper is used
    assert isinstance(agent.state, StateWrapper)
    print("✅ StateWrapper created for SharedState")

    # Verify state is accessible via wrapper
    assert agent.state.get("initial_value") == 42
    print("✅ State access via wrapper works")

    # Test dict-like operations
    agent.state["new_value"] = 100
    assert agent.state.get("new_value") == 100
    print("✅ Dict-like operations work")

    # Verify data is in underlying SharedState
    assert shared_state.get("new_value") == 100
    print("✅ Data synchronized with underlying SharedState")

    # Test keys, values, items
    keys = agent.state.keys()
    assert "initial_value" in keys
    assert "new_value" in keys
    print(f"✅ Keys: {keys}")

    items = agent.state.items()
    print(f"✅ Items: {dict(items)}")

    # Test update
    agent.state.update({"batch_value_1": "a", "batch_value_2": "b"})
    assert agent.state.get("batch_value_1") == "a"
    assert agent.state.get("batch_value_2") == "b"
    print("✅ Batch update works")

    # Test 'in' operator
    assert "initial_value" in agent.state
    print("✅ 'in' operator works")

    # Test len
    length = len(agent.state)
    print(f"✅ State length: {length}")

    print("\n✅ SharedState test PASSED\n")


def test_multi_agent_shared_state():
    """Test multiple agents sharing the same SharedState."""
    print("="*80)
    print("TEST 3: Multiple Agents with Shared State")
    print("="*80)

    tools = get_default_tools()
    shared_state = SharedState()

    # Create first agent
    agent1 = create_gemma_agent(
        tools=tools,
        state=shared_state,
        verbose=False,
        max_iterations=2
    )

    # Create second agent with same state
    agent2 = create_gemma_agent(
        tools=tools,
        state=shared_state,
        verbose=False,
        max_iterations=2
    )

    # Agent 1 sets a value
    agent1.state["from_agent_1"] = "hello from agent 1"

    # Agent 2 should see it
    assert agent2.state.get("from_agent_1") == "hello from agent 1"
    print("✅ Agent 2 can see Agent 1's data")

    # Agent 2 sets a value
    agent2.state["from_agent_2"] = "hello from agent 2"

    # Agent 1 should see it
    assert agent1.state.get("from_agent_2") == "hello from agent 2"
    print("✅ Agent 1 can see Agent 2's data")

    # Both should see all data
    keys = agent1.state.keys()
    assert "from_agent_1" in keys
    assert "from_agent_2" in keys
    print(f"✅ All keys visible to both agents: {keys}")

    print("\n✅ Multi-agent shared state test PASSED\n")


def test_state_wrapper_error_handling():
    """Test StateWrapper error handling."""
    print("="*80)
    print("TEST 4: StateWrapper Error Handling")
    print("="*80)

    shared_state = SharedState()
    wrapper = StateWrapper(shared_state)

    # Test KeyError for missing key
    try:
        value = wrapper["nonexistent_key"]
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"✅ KeyError raised correctly: {e}")

    # Test get with default
    value = wrapper.get("nonexistent_key", "default_value")
    assert value == "default_value"
    print("✅ get() with default works")

    # Test 'in' operator with missing key
    assert "nonexistent_key" not in wrapper
    print("✅ 'in' operator works for missing keys")

    print("\n✅ Error handling test PASSED\n")


def test_state_history_tracking():
    """Test that SharedState history is preserved through wrapper."""
    print("="*80)
    print("TEST 5: State History Tracking")
    print("="*80)

    shared_state = SharedState()
    wrapper = StateWrapper(shared_state)

    # Set values through wrapper
    wrapper["key1"] = "value1"
    wrapper["key2"] = "value2"
    wrapper["key1"] = "updated_value1"  # Update

    # Check history in underlying SharedState
    history = shared_state.get_history("key1")
    assert len(history) == 2  # Original + update
    print(f"✅ History tracked: {len(history)} entries for 'key1'")

    # Verify values
    assert history[0].value == "value1"
    assert history[1].value == "updated_value1"
    print("✅ History values correct")

    # Check sources
    assert all(entry.source == "agent" for entry in history)
    print("✅ Source tracked correctly")

    print("\n✅ History tracking test PASSED\n")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

    print("\n" + "="*80)
    print("STATE COMPATIBILITY TEST SUITE")
    print("="*80)

    try:
        test_dict_state()
    except Exception as e:
        logger.error(f"Dict state test failed: {e}", exc_info=True)

    try:
        test_shared_state()
    except Exception as e:
        logger.error(f"SharedState test failed: {e}", exc_info=True)

    try:
        test_multi_agent_shared_state()
    except Exception as e:
        logger.error(f"Multi-agent test failed: {e}", exc_info=True)

    try:
        test_state_wrapper_error_handling()
    except Exception as e:
        logger.error(f"Error handling test failed: {e}", exc_info=True)

    try:
        test_state_history_tracking()
    except Exception as e:
        logger.error(f"History tracking test failed: {e}", exc_info=True)

    print("="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
