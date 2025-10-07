"""Example: Async/await support for agents and DAG orchestration."""

import sys
import os
import asyncio
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.graph import AgentDAG, AgentNode, DAGExecutor, SharedState
from loguru import logger


async def example_async_agent():
    """Example 1: Using async agent.arun()"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Async Agent Execution")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        max_iterations=3
    )

    print("\n⏱️  Running agent asynchronously...")
    start_time = time.time()

    # Use async arun instead of sync run
    result = await agent.arun("Calculate 42 * 17 + 256", return_metrics=True)

    execution_time = time.time() - start_time

    print(f"\n✅ Result: {result.result}")
    print(f"⏱️  Execution time: {execution_time:.2f}s")
    print(f"📊 Metrics:")
    print(f"   - Iterations: {result.metrics.total_iterations}")
    print(f"   - Total tokens: {result.metrics.total_tokens}")
    print(f"   - LLM calls: {result.metrics.llm_calls}")


async def example_parallel_agents():
    """Example 2: Running multiple agents in parallel"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Parallel Agent Execution")
    print("="*80)

    tools = get_default_tools()

    # Create multiple agents
    agent1 = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)
    agent2 = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)
    agent3 = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)

    # Define tasks
    tasks = [
        ("Agent 1", agent1.arun("Calculate 10 + 20", return_metrics=False)),
        ("Agent 2", agent2.arun("Calculate 50 * 2", return_metrics=False)),
        ("Agent 3", agent3.arun("Calculate 100 / 4", return_metrics=False))
    ]

    print("\n⏱️  Running 3 agents in parallel...")
    start_time = time.time()

    # Execute all agents in parallel
    results = await asyncio.gather(*[task for _, task in tasks])

    execution_time = time.time() - start_time

    print(f"\n✅ Results (completed in {execution_time:.2f}s):")
    for (name, _), result in zip(tasks, results):
        print(f"   {name}: {result}")


async def example_async_dag_sequential():
    """Example 3: Async DAG with sequential execution"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Async DAG - Sequential Execution")
    print("="*80)

    tools = get_default_tools()

    # Create agents
    agent1 = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)
    agent2 = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)
    agent3 = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)

    # Build DAG: agent1 -> agent2 -> agent3
    dag = AgentDAG(name="SequentialPipeline")

    dag.add_node(AgentNode(name="step1", agent=agent1, output_key="result1"))
    dag.add_node(AgentNode(name="step2", agent=agent2, output_key="result2"))
    dag.add_node(AgentNode(name="step3", agent=agent3, output_key="result3"))

    dag.add_edge("step1", "step2")
    dag.add_edge("step2", "step3")

    print(f"\n📊 DAG Structure:")
    print(dag.visualize())

    # Execute with async (sequential within each level)
    executor = DAGExecutor(dag)

    print("\n⏱️  Executing DAG asynchronously (sequential)...")
    start_time = time.time()

    result = await executor.aexecute(
        initial_state={"input": "Process this data"},
        parallel=False  # Sequential execution
    )

    execution_time = time.time() - start_time

    print(f"\n✅ Execution complete:")
    print(f"   Status: {result.status}")
    print(f"   Completed: {result.completed_nodes}/{result.total_nodes} nodes")
    print(f"   Time: {execution_time:.2f}s")
    print(f"\n📊 Final state keys: {list(result.final_state.keys())}")


async def example_async_dag_parallel():
    """Example 4: Async DAG with parallel execution"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Async DAG - Parallel Execution")
    print("="*80)

    tools = get_default_tools()

    # Create agents for parallel processing
    ingestion = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)
    stats = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)
    patterns = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)
    report = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)

    # Build DAG with parallel processing
    #          ┌→ stats ┐
    # ingest ──┼→ patterns ┼→ report
    #          └→ (parallel) ┘

    dag = AgentDAG(name="ParallelPipeline")

    dag.add_node(AgentNode(name="ingest", agent=ingestion, output_key="raw_data"))
    dag.add_node(AgentNode(name="stats", agent=stats, output_key="stats_result"))
    dag.add_node(AgentNode(name="patterns", agent=patterns, output_key="pattern_result"))
    dag.add_node(AgentNode(name="report", agent=report, output_key="final_report"))

    dag.add_edge("ingest", "stats")
    dag.add_edge("ingest", "patterns")
    dag.add_edge("stats", "report")
    dag.add_edge("patterns", "report")

    print(f"\n📊 DAG Structure:")
    print(dag.visualize())

    # Execute with async and parallel processing
    executor = DAGExecutor(dag)

    print("\n⏱️  Executing DAG asynchronously (parallel)...")
    start_time = time.time()

    result = await executor.aexecute(
        initial_state={"input": "Sales data for Q4"},
        parallel=True  # Enable parallel execution
    )

    execution_time = time.time() - start_time

    print(f"\n✅ Execution complete:")
    print(f"   Status: {result.status}")
    print(f"   Completed: {result.completed_nodes}/{result.total_nodes} nodes")
    print(f"   Time: {execution_time:.2f}s")
    print(f"\n📝 Node execution order demonstrates parallelism:")
    print(f"   Level 1: [ingest] - runs first")
    print(f"   Level 2: [stats, patterns] - run in parallel")
    print(f"   Level 3: [report] - runs after both complete")


async def example_performance_comparison():
    """Example 5: Compare sync vs async performance"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Performance Comparison (Sync vs Async)")
    print("="*80)

    tools = get_default_tools()

    # Create DAG for testing
    def create_test_dag():
        dag = AgentDAG(name="TestDAG")

        # Create 3 parallel branches
        for i in range(3):
            agent = create_gemma_agent(tools=tools, verbose=False, max_iterations=2)
            dag.add_node(AgentNode(name=f"task{i}", agent=agent, output_key=f"result{i}"))

        return dag

    # Test 1: Sync execution
    print("\n⏱️  Test 1: Synchronous execution...")
    dag_sync = create_test_dag()
    executor_sync = DAGExecutor(dag_sync)

    start_sync = time.time()
    result_sync = executor_sync.execute(
        initial_state={"input": "test"},
        parallel=False
    )
    time_sync = time.time() - start_sync

    print(f"   ✅ Sync time: {time_sync:.2f}s")

    # Test 2: Async sequential
    print("\n⏱️  Test 2: Async execution (sequential)...")
    dag_async_seq = create_test_dag()
    executor_async_seq = DAGExecutor(dag_async_seq)

    start_async_seq = time.time()
    result_async_seq = await executor_async_seq.aexecute(
        initial_state={"input": "test"},
        parallel=False
    )
    time_async_seq = time.time() - start_async_seq

    print(f"   ✅ Async sequential time: {time_async_seq:.2f}s")

    # Test 3: Async parallel
    print("\n⏱️  Test 3: Async execution (parallel)...")
    dag_async_par = create_test_dag()
    executor_async_par = DAGExecutor(dag_async_par)

    start_async_par = time.time()
    result_async_par = await executor_async_par.aexecute(
        initial_state={"input": "test"},
        parallel=True
    )
    time_async_par = time.time() - start_async_par

    print(f"   ✅ Async parallel time: {time_async_par:.2f}s")

    # Summary
    print(f"\n📊 Performance Summary:")
    print(f"   Sync execution:        {time_sync:.2f}s (baseline)")
    print(f"   Async sequential:      {time_async_seq:.2f}s ({time_async_seq/time_sync*100:.1f}% of sync)")
    print(f"   Async parallel:        {time_async_par:.2f}s ({time_async_par/time_sync*100:.1f}% of sync)")
    if time_async_par < time_sync:
        speedup = time_sync / time_async_par
        print(f"\n   🚀 Speedup with parallel: {speedup:.2f}x faster!")


async def main():
    """Run all async examples"""
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

    print("\n" + "="*80)
    print("ASYNC/AWAIT EXAMPLES")
    print("="*80)

    try:
        await example_async_agent()
        await example_parallel_agents()
        await example_async_dag_sequential()
        await example_async_dag_parallel()
        await example_performance_comparison()

        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETE")
        print("="*80)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
