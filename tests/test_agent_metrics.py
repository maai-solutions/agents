"""Test the ReasoningAgent's metrics tracking."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.agent import create_gemma_agent, AgentResponse
from linus.agents.agent.tools import get_default_tools
from loguru import logger
import json

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/agent_metrics_test.log", rotation="10 MB", level="DEBUG")


def print_metrics(response: AgentResponse, test_name: str):
    """Pretty print metrics from agent response."""
    print(f"\n{'='*80}")
    print(f"METRICS for {test_name}")
    print(f"{'='*80}")

    metrics_dict = response.metrics.to_dict()

    print("\n📊 Performance Metrics:")
    print(f"  ⏱️  Execution Time: {metrics_dict['execution_time_seconds']:.3f} seconds")
    print(f"  🔄 Total Iterations: {metrics_dict['total_iterations']}")
    print(f"  ✅ Task Completed: {metrics_dict['task_completed']}")
    if metrics_dict['iterations_to_completion']:
        print(f"  🎯 Iterations to Completion: {metrics_dict['iterations_to_completion']}")

    print("\n🤖 LLM Usage:")
    print(f"  📞 Total LLM Calls: {metrics_dict['llm_calls']}")
    print(f"  🧠 Reasoning Calls: {metrics_dict['reasoning_calls']}")
    print(f"  ✔️  Completion Checks: {metrics_dict['completion_checks']}")

    print("\n🎫 Token Usage:")
    print(f"  📥 Input Tokens: {metrics_dict['total_input_tokens']:,}")
    print(f"  📤 Output Tokens: {metrics_dict['total_output_tokens']:,}")
    print(f"  📊 Total Tokens: {metrics_dict['total_tokens']:,}")
    print(f"  📈 Avg Tokens/Call: {metrics_dict['avg_tokens_per_llm_call']}")

    print("\n🛠️  Tool Usage:")
    print(f"  🔧 Tool Executions: {metrics_dict['tool_executions']}")
    print(f"  ✅ Successful: {metrics_dict['successful_tool_calls']}")
    print(f"  ❌ Failed: {metrics_dict['failed_tool_calls']}")
    if metrics_dict['tool_executions'] > 0:
        print(f"  📊 Success Rate: {metrics_dict['success_rate']:.0%}")

    print("\n📝 Execution History:")
    for i, item in enumerate(response.execution_history, 1):
        print(f"  {i}. [{item['status']}] {item['task']}")
        print(f"     Tool: {item['tool'] or 'None'}")
        print(f"     Result: {item['result'][:100]}...")

    if response.completion_status:
        print(f"\n🎯 Completion Status:")
        print(f"  Complete: {response.completion_status['is_complete']}")
        print(f"  Reasoning: {response.completion_status['reasoning']}")

    print(f"\n💬 Final Result:")
    result_str = str(response.result)
    if len(result_str) > 300:
        print(f"  {result_str[:300]}...")
    else:
        print(f"  {result_str}")

    print(f"\n{'='*80}\n")


def test_simple_calculation_with_metrics():
    """Test simple calculation and track metrics."""
    print("\n" + "="*80)
    print("TEST 1: Simple Calculation with Metrics")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        max_iterations=5
    )

    response = agent.run("Calculate 42 * 17", return_metrics=True)

    assert isinstance(response, AgentResponse), "Should return AgentResponse when return_metrics=True"
    print_metrics(response, "Simple Calculation")


def test_multi_step_with_metrics():
    """Test multi-step task with metrics tracking."""
    print("\n" + "="*80)
    print("TEST 2: Multi-Step Task with Metrics")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        max_iterations=8
    )

    response = agent.run(
        "Calculate 100 + 50, then search for Python, and finally get the current time",
        return_metrics=True
    )

    print_metrics(response, "Multi-Step Task")

    # Verify metrics make sense
    assert response.metrics.total_iterations >= 1, "Should have at least 1 iteration"
    assert response.metrics.llm_calls >= response.metrics.reasoning_calls, "LLM calls should >= reasoning calls"
    assert response.metrics.execution_time_seconds > 0, "Execution time should be positive"


def test_complex_task_metrics():
    """Test complex task and analyze detailed metrics."""
    print("\n" + "="*80)
    print("TEST 3: Complex Task - Detailed Metrics Analysis")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        max_iterations=10
    )

    response = agent.run(
        "First calculate 25 * 4, then search for information about that number, "
        "and finally get the current time to timestamp the result",
        return_metrics=True
    )

    print_metrics(response, "Complex Task")

    # Export metrics to JSON
    metrics_json = json.dumps(response.metrics.to_dict(), indent=2)
    print("\n📄 Metrics JSON:")
    print(metrics_json)

    # Export full response
    full_response_json = json.dumps(response.to_dict(), indent=2, default=str)
    with open("logs/agent_response_example.json", "w") as f:
        f.write(full_response_json)
    print("\n✅ Full response saved to logs/agent_response_example.json")


def test_return_metrics_false():
    """Test that return_metrics=False returns only the result."""
    print("\n" + "="*80)
    print("TEST 4: return_metrics=False")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        max_iterations=5
    )

    result = agent.run("Calculate 10 + 5", return_metrics=False)

    # Should return string or BaseModel, not AgentResponse
    assert not isinstance(result, AgentResponse), "Should NOT return AgentResponse when return_metrics=False"
    print(f"\n✅ Result (no metrics): {result}\n")


def compare_metrics():
    """Compare metrics across different task complexities."""
    print("\n" + "="*80)
    print("TEST 5: Metrics Comparison Across Task Complexities")
    print("="*80)

    tools = get_default_tools()

    tasks = [
        ("Simple", "Calculate 5 + 3"),
        ("Medium", "Calculate 42 * 17 and tell me if it's prime"),
        ("Complex", "Calculate 100/4, then calculate that result * 2, then search for that number"),
    ]

    results = []

    for name, task in tasks:
        agent = create_gemma_agent(tools=tools, verbose=False, max_iterations=10)
        response = agent.run(task, return_metrics=True)
        results.append((name, response))

    print("\n📊 Metrics Comparison:")
    print(f"\n{'Task':<15} {'Iterations':<12} {'LLM Calls':<12} {'Tokens':<12} {'Time (s)':<12} {'Tools':<12}")
    print("-" * 80)

    for name, response in results:
        m = response.metrics
        print(f"{name:<15} {m.total_iterations:<12} {m.llm_calls:<12} {m.total_tokens:<12} {m.execution_time_seconds:<12.3f} {m.tool_executions:<12}")

    print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("REASONING AGENT - METRICS TRACKING TESTS")
    print("Testing comprehensive metrics collection and reporting")
    print("="*80)

    os.makedirs("logs", exist_ok=True)

    # Run tests
    try:
        test_simple_calculation_with_metrics()
    except Exception as e:
        logger.error(f"Test 1 failed: {e}", exc_info=True)

    try:
        test_multi_step_with_metrics()
    except Exception as e:
        logger.error(f"Test 2 failed: {e}", exc_info=True)

    try:
        test_complex_task_metrics()
    except Exception as e:
        logger.error(f"Test 3 failed: {e}", exc_info=True)

    try:
        test_return_metrics_false()
    except Exception as e:
        logger.error(f"Test 4 failed: {e}", exc_info=True)

    try:
        compare_metrics()
    except Exception as e:
        logger.error(f"Test 5 failed: {e}", exc_info=True)

    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
