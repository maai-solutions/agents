"""Test DAG orchestration for multi-agent systems."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.graph import AgentDAG, AgentNode, DAGExecutor, SharedState
from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/dag_test.log", rotation="10 MB", level="DEBUG")


def test_simple_dag():
    """Test a simple linear DAG."""
    print("\n" + "="*80)
    print("TEST 1: Simple Linear DAG")
    print("="*80)

    tools = get_default_tools()

    # Create agents
    calculator_agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        max_iterations=3
    )

    search_agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        max_iterations=3
    )

    reporter_agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        max_iterations=3
    )

    # Create DAG
    dag = AgentDAG(name="LinearWorkflow")

    # Add nodes
    dag.add_node(AgentNode(
        name="calculator",
        agent=calculator_agent,
        description="Calculate result",
        input_mapping={"query": "calculation_query"},
        output_key="calculation_result"
    ))

    dag.add_node(AgentNode(
        name="searcher",
        agent=search_agent,
        description="Search for information",
        input_mapping={"query": "search_query"},
        output_key="search_result"
    ))

    dag.add_node(AgentNode(
        name="reporter",
        agent=reporter_agent,
        description="Generate final report",
        output_key="final_report"
    ))

    # Add edges (linear flow)
    dag.add_edge("calculator", "searcher")
    dag.add_edge("searcher", "reporter")

    # Visualize
    print("\n" + dag.visualize())

    # Execute
    executor = DAGExecutor(dag)
    result = executor.execute(initial_state={
        "calculation_query": "Calculate 42 * 17",
        "search_query": "What is the significance of the number 714?"
    })

    # Print results
    print(f"\n📊 Execution Results:")
    print(f"  Status: {result.status}")
    print(f"  Completed: {result.completed_nodes}/{result.total_nodes}")
    print(f"  Time: {result.execution_time_seconds:.2f}s")

    print(f"\n📝 Node Results:")
    for node_name, node_result in result.node_results.items():
        print(f"  {node_name}: {str(node_result)[:100]}...")

    print(f"\n🗂️  Final State:")
    for key, value in result.final_state.items():
        print(f"  {key}: {str(value)[:100]}...")


def test_parallel_dag():
    """Test DAG with parallel execution."""
    print("\n" + "="*80)
    print("TEST 2: Parallel DAG")
    print("="*80)

    tools = get_default_tools()

    # Create multiple agents
    agents = {
        "calc1": create_gemma_agent(tools=tools, verbose=False, max_iterations=2),
        "calc2": create_gemma_agent(tools=tools, verbose=False, max_iterations=2),
        "calc3": create_gemma_agent(tools=tools, verbose=False, max_iterations=2),
        "aggregator": create_gemma_agent(tools=tools, verbose=False, max_iterations=3)
    }

    # Create DAG
    dag = AgentDAG(name="ParallelWorkflow")

    # Add parallel calculation nodes
    dag.add_node(AgentNode(
        name="calc1",
        agent=agents["calc1"],
        description="Calculate first value",
        input_mapping={"query": "calc1_query"},
        output_key="result1"
    ))

    dag.add_node(AgentNode(
        name="calc2",
        agent=agents["calc2"],
        description="Calculate second value",
        input_mapping={"query": "calc2_query"},
        output_key="result2"
    ))

    dag.add_node(AgentNode(
        name="calc3",
        agent=agents["calc3"],
        description="Calculate third value",
        input_mapping={"query": "calc3_query"},
        output_key="result3"
    ))

    # Add aggregator that depends on all calculations
    dag.add_node(AgentNode(
        name="aggregator",
        agent=agents["aggregator"],
        description="Aggregate all results",
        output_key="final_result"
    ))

    # Create diamond pattern (parallel then merge)
    dag.add_edge("calc1", "aggregator")
    dag.add_edge("calc2", "aggregator")
    dag.add_edge("calc3", "aggregator")

    # Visualize
    print("\n" + dag.visualize())

    # Execute
    executor = DAGExecutor(dag)
    result = executor.execute(initial_state={
        "calc1_query": "Calculate 10 * 5",
        "calc2_query": "Calculate 20 + 30",
        "calc3_query": "Calculate 100 / 2"
    })

    # Print results
    print(f"\n📊 Execution Results:")
    print(f"  Status: {result.status}")
    print(f"  Time: {result.execution_time_seconds:.2f}s")
    print(f"\n📝 Results: {result.node_results}")


def test_conditional_dag():
    """Test DAG with conditional edges."""
    print("\n" + "="*80)
    print("TEST 3: Conditional DAG")
    print("="*80)

    tools = get_default_tools()

    # Create agents
    checker_agent = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)
    path_a_agent = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)
    path_b_agent = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)

    # Create DAG
    dag = AgentDAG(name="ConditionalWorkflow")

    dag.add_node(AgentNode(
        name="checker",
        agent=checker_agent,
        description="Check condition",
        output_key="check_result"
    ))

    dag.add_node(AgentNode(
        name="path_a",
        agent=path_a_agent,
        description="Execute path A",
        output_key="path_a_result"
    ))

    dag.add_node(AgentNode(
        name="path_b",
        agent=path_b_agent,
        description="Execute path B",
        output_key="path_b_result"
    ))

    # Add conditional edges
    def check_value_high(state: SharedState) -> bool:
        """Check if value is high."""
        result = state.get("check_result", "")
        # Simple heuristic: if result contains certain keywords
        return "high" in str(result).lower() or "large" in str(result).lower()

    def check_value_low(state: SharedState) -> bool:
        """Check if value is low."""
        return not check_value_high(state)

    dag.add_edge("checker", "path_a", condition=check_value_high)
    dag.add_edge("checker", "path_b", condition=check_value_low)

    # Visualize
    print("\n" + dag.visualize())

    # Execute
    executor = DAGExecutor(dag)
    result = executor.execute(initial_state={
        "value": 100
    })

    print(f"\n📊 Result: {result.status}")
    print(f"📝 Executed nodes: {list(result.node_results.keys())}")


def test_error_handling():
    """Test DAG error handling."""
    print("\n" + "="*80)
    print("TEST 4: Error Handling")
    print("="*80)

    tools = get_default_tools()

    # Create a failing agent (by giving it impossible task)
    failing_agent = create_gemma_agent(tools=tools, verbose=False, max_iterations=1)
    recovery_agent = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)

    # Create DAG
    dag = AgentDAG(name="ErrorHandling")

    # Node that might fail
    dag.add_node(AgentNode(
        name="risky_task",
        agent=failing_agent,
        description="Task that might fail",
        on_error="skip",  # Skip on error instead of failing
        output_key="risky_result"
    ))

    # Recovery node
    dag.add_node(AgentNode(
        name="recovery",
        agent=recovery_agent,
        description="Recovery task",
        output_key="recovery_result"
    ))

    dag.add_edge("risky_task", "recovery")

    # Execute
    executor = DAGExecutor(dag)
    result = executor.execute(initial_state={
        "task": "Do something"
    })

    print(f"\n📊 Status: {result.status}")
    print(f"  Completed: {result.completed_nodes}")
    print(f"  Failed: {result.failed_nodes}")
    print(f"  Skipped: {result.skipped_nodes}")

    if result.errors:
        print(f"\n❌ Errors:")
        for node, error in result.errors.items():
            print(f"  {node}: {error}")


def test_state_management():
    """Test shared state management."""
    print("\n" + "="*80)
    print("TEST 5: State Management")
    print("="*80)

    # Create shared state
    state = SharedState()

    # Set values
    state.set("value1", 42, source="test")
    state.set("value2", "hello", source="test")
    state.set("value3", {"key": "value"}, source="test")

    # Get values
    print(f"\nState values:")
    print(f"  value1: {state.get('value1')}")
    print(f"  value2: {state.get('value2')}")
    print(f"  value3: {state.get('value3')}")

    # Get all
    print(f"\nAll state: {state.get_all()}")

    # History
    print(f"\nHistory:")
    for entry in state.get_history():
        print(f"  {entry.key}: {entry.value} (from {entry.source} at {entry.timestamp})")

    # Update value
    state.set("value1", 100, source="update")

    # Check history for specific key
    print(f"\nHistory for 'value1':")
    for entry in state.get_history("value1"):
        print(f"  {entry.value} at {entry.timestamp}")


def test_dag_visualization():
    """Test DAG visualization."""
    print("\n" + "="*80)
    print("TEST 6: DAG Visualization")
    print("="*80)

    tools = get_default_tools()

    # Create a complex DAG
    dag = AgentDAG(name="ComplexWorkflow")

    agents = {
        f"agent_{i}": create_gemma_agent(tools=tools, verbose=False)
        for i in range(6)
    }

    # Add nodes
    for i in range(6):
        dag.add_node(AgentNode(
            name=f"node_{i}",
            agent=agents[f"agent_{i}"],
            description=f"Task {i}"
        ))

    # Create complex structure
    # node_0 -> node_1, node_2
    # node_1 -> node_3
    # node_2 -> node_3, node_4
    # node_3 -> node_5
    # node_4 -> node_5

    dag.add_edge("node_0", "node_1")
    dag.add_edge("node_0", "node_2")
    dag.add_edge("node_1", "node_3")
    dag.add_edge("node_2", "node_3")
    dag.add_edge("node_2", "node_4")
    dag.add_edge("node_3", "node_5")
    dag.add_edge("node_4", "node_5")

    # Visualize
    print("\n" + dag.visualize())

    # Show execution order
    print(f"\nExecution levels: {dag.get_execution_order()}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DAG ORCHESTRATION TESTS")
    print("Testing multi-agent DAG workflows")
    print("="*80)

    os.makedirs("logs", exist_ok=True)

    # Run tests
    try:
        test_simple_dag()
    except Exception as e:
        logger.error(f"Test 1 failed: {e}", exc_info=True)

    try:
        test_parallel_dag()
    except Exception as e:
        logger.error(f"Test 2 failed: {e}", exc_info=True)

    try:
        test_conditional_dag()
    except Exception as e:
        logger.error(f"Test 3 failed: {e}", exc_info=True)

    try:
        test_error_handling()
    except Exception as e:
        logger.error(f"Test 4 failed: {e}", exc_info=True)

    try:
        test_state_management()
    except Exception as e:
        logger.error(f"Test 5 failed: {e}", exc_info=True)

    try:
        test_dag_visualization()
    except Exception as e:
        logger.error(f"Test 6 failed: {e}", exc_info=True)

    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
