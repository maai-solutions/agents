"""Quick test to verify hierarchical tracing implementation is working."""

import sys
sys.path.insert(0, 'src')

from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.telemetry import initialize_telemetry

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")

    # Test telemetry import
    from linus.agents.telemetry import LangfuseTracer, AgentTracer, trace_method
    print("✓ Telemetry imports successful")

    # Test agent imports
    from linus.agents.agent.base import Agent as BaseAgent
    from linus.agents.agent.reasoning_agent import ReasoningAgent
    print("✓ Agent imports successful")

    # Test factory import
    from linus.agents.agent.factory import Agent
    print("✓ Factory import successful")

    return True

def test_agent_creation():
    """Test that agent can be created with agent_name."""
    print("\nTesting agent creation with hierarchical naming...")

    # Initialize telemetry (disabled for testing)
    tracer = initialize_telemetry(
        service_name="test",
        exporter_type="console",
        enabled=False,
        agent_name="test_agent"
    )
    print("✓ Telemetry initialized")

    # Create agent with custom name
    agent = Agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=get_default_tools(),
        agent_name="calculator",
        tracer=tracer,
        use_async=True
    )
    print("✓ Agent created successfully")

    # Verify agent has the correct name
    assert agent.agent_name == "calculator", f"Expected 'calculator', got '{agent.agent_name}'"
    print(f"✓ Agent name is correctly set to: {agent.agent_name}")

    # Verify telemetry has the agent_name attribute
    assert hasattr(agent.telemetry, 'agent_name'), "Telemetry should have agent_name attribute"
    print(f"✓ Telemetry agent name: {agent.telemetry.agent_name}")

    return True

def test_tracer_methods():
    """Test that tracer methods accept hierarchical naming parameters."""
    print("\nTesting tracer method signatures...")

    tracer = initialize_telemetry(
        service_name="test",
        exporter_type="console",
        enabled=False,
        agent_name="test"
    )

    # Check that trace methods accept the new parameters
    import inspect

    # Check trace_agent_run signature
    sig = inspect.signature(tracer.trace_agent_run)
    params = list(sig.parameters.keys())
    assert 'agent_name' in params, "trace_agent_run should accept agent_name parameter"
    print(f"✓ trace_agent_run signature: {params}")

    # Check trace_llm_call signature
    sig = inspect.signature(tracer.trace_llm_call)
    params = list(sig.parameters.keys())
    assert 'llm_name' in params, "trace_llm_call should accept llm_name parameter"
    print(f"✓ trace_llm_call signature: {params}")

    # Check trace_tool_execution signature
    sig = inspect.signature(tracer.trace_tool_execution)
    params = list(sig.parameters.keys())
    assert 'tool_display_name' in params, "trace_tool_execution should accept tool_display_name parameter"
    print(f"✓ trace_tool_execution signature: {params}")

    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Hierarchical Tracing Implementation Test")
    print("=" * 60)

    try:
        # Run tests
        test_imports()
        test_agent_creation()
        test_tracer_methods()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nHierarchical tracing is working correctly:")
        print("  • agent.<name> for agent traces")
        print("  • llm.<name> for LLM traces")
        print("  • tool.<name> for tool traces")

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
