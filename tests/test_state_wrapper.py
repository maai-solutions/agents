"""Test StateWrapper functionality independently."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import only state components, not the full agent
from linus.agents.graph.state import SharedState


# Copy StateWrapper class for isolated testing
class StateWrapper:
    """Wrapper to provide dict-like interface to SharedState."""

    def __init__(self, shared_state):
        self._shared_state = shared_state

    def __getitem__(self, key):
        value = self._shared_state.get(key)
        if value is None:
            raise KeyError(f"Key '{key}' not found in state")
        return value

    def __setitem__(self, key, value):
        self._shared_state.set(key, value, source="agent")

    def __contains__(self, key):
        return self._shared_state.get(key) is not None

    def get(self, key, default=None):
        return self._shared_state.get(key, default=default)

    def keys(self):
        return list(self._shared_state.get_all().keys())

    def values(self):
        return list(self._shared_state.get_all().values())

    def items(self):
        return list(self._shared_state.get_all().items())

    def update(self, other):
        for key, value in other.items():
            self._shared_state.set(key, value, source="agent")

    def __repr__(self):
        return f"StateWrapper({self._shared_state.get_all()})"

    def __len__(self):
        return len(self._shared_state.get_all())


def test_wrapper_basic_operations():
    """Test basic dict-like operations."""
    print("\n" + "="*80)
    print("TEST 1: Basic Operations")
    print("="*80)

    shared_state = SharedState()
    wrapper = StateWrapper(shared_state)

    # Test set via __setitem__
    wrapper["key1"] = "value1"
    assert wrapper.get("key1") == "value1"
    print("✅ __setitem__ and get() work")

    # Test __getitem__
    assert wrapper["key1"] == "value1"
    print("✅ __getitem__ works")

    # Test get with default
    assert wrapper.get("missing", "default") == "default"
    print("✅ get() with default works")

    # Test __contains__
    assert "key1" in wrapper
    assert "missing" not in wrapper
    print("✅ __contains__ works")

    print("\n✅ Basic operations test PASSED\n")


def test_wrapper_collection_operations():
    """Test collection-like operations."""
    print("="*80)
    print("TEST 2: Collection Operations")
    print("="*80)

    shared_state = SharedState()
    wrapper = StateWrapper(shared_state)

    # Add some data
    wrapper["a"] = 1
    wrapper["b"] = 2
    wrapper["c"] = 3

    # Test keys()
    keys = wrapper.keys()
    assert set(keys) == {"a", "b", "c"}
    print(f"✅ keys() works: {keys}")

    # Test values()
    values = wrapper.values()
    assert set(values) == {1, 2, 3}
    print(f"✅ values() works: {values}")

    # Test items()
    items = wrapper.items()
    assert dict(items) == {"a": 1, "b": 2, "c": 3}
    print(f"✅ items() works: {dict(items)}")

    # Test len()
    assert len(wrapper) == 3
    print(f"✅ len() works: {len(wrapper)}")

    # Test update()
    wrapper.update({"d": 4, "e": 5})
    assert wrapper.get("d") == 4
    assert wrapper.get("e") == 5
    assert len(wrapper) == 5
    print("✅ update() works")

    print("\n✅ Collection operations test PASSED\n")


def test_wrapper_error_handling():
    """Test error handling."""
    print("="*80)
    print("TEST 3: Error Handling")
    print("="*80)

    shared_state = SharedState()
    wrapper = StateWrapper(shared_state)

    # Test KeyError for missing key
    try:
        value = wrapper["missing_key"]
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"✅ KeyError raised correctly: {e}")

    # Verify get() doesn't raise
    value = wrapper.get("missing_key", "safe_default")
    assert value == "safe_default"
    print("✅ get() with default doesn't raise")

    print("\n✅ Error handling test PASSED\n")


def test_wrapper_data_sync():
    """Test that wrapper properly syncs with SharedState."""
    print("="*80)
    print("TEST 4: Data Synchronization")
    print("="*80)

    shared_state = SharedState()
    wrapper = StateWrapper(shared_state)

    # Set via wrapper
    wrapper["from_wrapper"] = "wrapper_value"

    # Verify in SharedState
    assert shared_state.get("from_wrapper") == "wrapper_value"
    print("✅ Data set via wrapper visible in SharedState")

    # Set via SharedState
    shared_state.set("from_state", "state_value", source="test")

    # Verify in wrapper
    assert wrapper.get("from_state") == "state_value"
    print("✅ Data set via SharedState visible in wrapper")

    print("\n✅ Data synchronization test PASSED\n")


def test_wrapper_with_multiple_wrappers():
    """Test multiple wrappers sharing same SharedState."""
    print("="*80)
    print("TEST 5: Multiple Wrappers")
    print("="*80)

    shared_state = SharedState()
    wrapper1 = StateWrapper(shared_state)
    wrapper2 = StateWrapper(shared_state)

    # Set via wrapper1
    wrapper1["shared_key"] = "value_from_wrapper1"

    # Verify via wrapper2
    assert wrapper2.get("shared_key") == "value_from_wrapper1"
    print("✅ Wrapper2 sees data from Wrapper1")

    # Update via wrapper2
    wrapper2["shared_key"] = "updated_by_wrapper2"

    # Verify via wrapper1
    assert wrapper1.get("shared_key") == "updated_by_wrapper2"
    print("✅ Wrapper1 sees updates from Wrapper2")

    # Both should see same keys
    assert set(wrapper1.keys()) == set(wrapper2.keys())
    print("✅ Both wrappers see same keys")

    print("\n✅ Multiple wrappers test PASSED\n")


def test_wrapper_repr():
    """Test string representation."""
    print("="*80)
    print("TEST 6: String Representation")
    print("="*80)

    shared_state = SharedState()
    wrapper = StateWrapper(shared_state)

    wrapper["x"] = 10
    wrapper["y"] = 20

    repr_str = repr(wrapper)
    print(f"Wrapper repr: {repr_str}")
    assert "x" in repr_str
    assert "10" in repr_str or "y" in repr_str
    print("✅ __repr__ works")

    print("\n✅ String representation test PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("STATE WRAPPER TEST SUITE")
    print("="*80)

    try:
        test_wrapper_basic_operations()
        test_wrapper_collection_operations()
        test_wrapper_error_handling()
        test_wrapper_data_sync()
        test_wrapper_with_multiple_wrappers()
        test_wrapper_repr()

        print("="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
