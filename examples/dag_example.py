"""Example: Multi-agent DAG orchestration for data processing pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.graph import AgentDAG, AgentNode, DAGExecutor, SharedState
from linus.agents.agent.agent import Agent
from linus.agents.agent.tools import get_default_tools
from loguru import logger


def example_data_pipeline():
    """
    Example: Data processing pipeline with multiple agents.

    Pipeline:
    1. Data Ingestion - Load and validate data
    2. Data Analysis - Analyze loaded data (parallel)
       a. Statistical Analysis
       b. Pattern Detection
    3. Report Generation - Combine analyses into report
    """

    print("="*80)
    print("DATA PROCESSING PIPELINE")
    print("="*80)

    tools = get_default_tools()

    # Create specialized agents
    ingestion_agent = Agent(
        tools=tools,
        verbose=False,
        max_iterations=3
    )

    stats_agent = Agent(
        tools=tools,
        verbose=False,
        max_iterations=5
    )

    pattern_agent = Agent(
        tools=tools,
        verbose=False,
        max_iterations=5
    )

    report_agent = Agent(
        tools=tools,
        verbose=False,
        max_iterations=3
    )

    # Build DAG
    dag = AgentDAG(name="DataPipeline")

    # 1. Ingestion node
    dag.add_node(AgentNode(
        name="ingest",
        agent=ingestion_agent,
        description="Ingest and validate data",
        input_mapping={"data_source": "data_source"},
        output_key="raw_data"
    ))

    # 2a. Statistical analysis (runs in parallel with 2b)
    dag.add_node(AgentNode(
        name="stats_analysis",
        agent=stats_agent,
        description="Perform statistical analysis",
        input_mapping={"data": "raw_data"},
        output_key="stats_results"
    ))

    # 2b. Pattern detection (runs in parallel with 2a)
    dag.add_node(AgentNode(
        name="pattern_detection",
        agent=pattern_agent,
        description="Detect patterns in data",
        input_mapping={"data": "raw_data"},
        output_key="pattern_results"
    ))

    # 3. Report generation (waits for both analyses)
    dag.add_node(AgentNode(
        name="generate_report",
        agent=report_agent,
        description="Generate comprehensive report",
        output_key="final_report"
    ))

    # Define edges
    dag.add_edge("ingest", "stats_analysis")
    dag.add_edge("ingest", "pattern_detection")
    dag.add_edge("stats_analysis", "generate_report")
    dag.add_edge("pattern_detection", "generate_report")

    # Visualize DAG
    print("\n" + dag.visualize())

    # Execute pipeline
    print("\n" + "="*80)
    print("EXECUTING PIPELINE")
    print("="*80)

    executor = DAGExecutor(dag)
    result = executor.execute(initial_state={
        "data_source": "Sales data for Q4 2024 showing revenue trends"
    })

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\n📊 Pipeline Status: {result.status.upper()}")
    print(f"⏱️  Execution Time: {result.execution_time_seconds:.2f}s")
    print(f"✅ Completed: {result.completed_nodes}/{result.total_nodes} nodes")

    if result.errors:
        print(f"\n❌ Errors:")
        for node, error in result.errors.items():
            print(f"  {node}: {error}")

    print(f"\n📝 Node Results:")
    for node_name, node_result in result.node_results.items():
        print(f"\n  [{node_name}]")
        print(f"  {str(node_result)[:200]}...")

    print(f"\n🗂️  Final State Keys: {list(result.final_state.keys())}")

    return result


def example_conditional_workflow():
    """
    Example: Conditional workflow based on data characteristics.

    Workflow:
    1. Analyze data characteristics
    2. Route to appropriate processing:
       - If numeric: Statistical processing
       - If text: NLP processing
    3. Generate specialized report
    """

    print("\n\n" + "="*80)
    print("CONDITIONAL WORKFLOW")
    print("="*80)

    tools = get_default_tools()

    # Create agents
    analyzer = Agent(tools=tools, verbose=False, max_iterations=3)
    numeric_processor = Agent(tools=tools, verbose=False, max_iterations=4)
    text_processor = Agent(tools=tools, verbose=False, max_iterations=4)
    report_generator = Agent(tools=tools, verbose=False, max_iterations=3)

    # Build DAG
    dag = AgentDAG(name="ConditionalWorkflow")

    # Analyzer node
    dag.add_node(AgentNode(
        name="analyzer",
        agent=analyzer,
        description="Analyze data type",
        output_key="data_type"
    ))

    # Processing paths
    dag.add_node(AgentNode(
        name="numeric_processor",
        agent=numeric_processor,
        description="Process numeric data",
        output_key="processed_data"
    ))

    dag.add_node(AgentNode(
        name="text_processor",
        agent=text_processor,
        description="Process text data",
        output_key="processed_data"
    ))

    # Report generator
    dag.add_node(AgentNode(
        name="reporter",
        agent=report_generator,
        description="Generate report",
        output_key="report"
    ))

    # Conditional edges
    def is_numeric_data(state: SharedState) -> bool:
        data_type = str(state.get("data_type", "")).lower()
        return "numeric" in data_type or "number" in data_type

    def is_text_data(state: SharedState) -> bool:
        return not is_numeric_data(state)

    dag.add_edge("analyzer", "numeric_processor", condition=is_numeric_data)
    dag.add_edge("analyzer", "text_processor", condition=is_text_data)
    dag.add_edge("numeric_processor", "reporter")
    dag.add_edge("text_processor", "reporter")

    # Visualize
    print("\n" + dag.visualize())

    # Execute with numeric data
    print("\n--- Execution with numeric data ---")
    executor = DAGExecutor(dag)
    result = executor.execute(initial_state={
        "input_data": "Dataset contains numeric values: 10, 20, 30, 40"
    })
    print(f"Status: {result.status}, Nodes executed: {list(result.node_results.keys())}")

    # Reset and execute with text data
    print("\n--- Execution with text data ---")
    dag.reset()
    executor = DAGExecutor(dag)
    result = executor.execute(initial_state={
        "input_data": "Dataset contains customer reviews and feedback"
    })
    print(f"Status: {result.status}, Nodes executed: {list(result.node_results.keys())}")


def example_error_recovery():
    """
    Example: DAG with error handling and recovery.

    Workflow:
    1. Primary processing task
    2. If task fails: Route to recovery task
    3. Validation task
    """

    print("\n\n" + "="*80)
    print("ERROR RECOVERY WORKFLOW")
    print("="*80)

    tools = get_default_tools()

    # Create agents
    primary = Agent(tools=tools, verbose=False, max_iterations=2)
    recovery = Agent(tools=tools, verbose=False, max_iterations=3)
    validator = Agent(tools=tools, verbose=False, max_iterations=2)

    # Build DAG
    dag = AgentDAG(name="ErrorRecovery")

    # Primary task (might fail)
    dag.add_node(AgentNode(
        name="primary_task",
        agent=primary,
        description="Primary processing (might fail)",
        on_error="skip",  # Skip to next node on error
        output_key="result"
    ))

    # Recovery task (if primary fails)
    dag.add_node(AgentNode(
        name="recovery_task",
        agent=recovery,
        description="Recovery processing",
        output_key="result"
    ))

    # Validator (always runs)
    dag.add_node(AgentNode(
        name="validator",
        agent=validator,
        description="Validate results",
        output_key="validation"
    ))

    # Setup edges
    dag.add_edge("primary_task", "validator")
    dag.add_edge("recovery_task", "validator")

    # Note: In a real scenario, you'd add logic to trigger recovery_task
    # only when primary_task fails, possibly through conditional edges

    print("\n" + dag.visualize())

    # Execute
    executor = DAGExecutor(dag)
    result = executor.execute(initial_state={
        "input": "Test data for processing"
    })

    print(f"\nStatus: {result.status}")
    print(f"Results: {result.node_results.keys()}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    # Run examples
    print("\n" + "="*80)
    print("MULTI-AGENT DAG ORCHESTRATION EXAMPLES")
    print("="*80)

    try:
        example_data_pipeline()
    except Exception as e:
        logger.exception(f"Data pipeline example failed: {e}", exc_info=True)

    try:
        example_conditional_workflow()
    except Exception as e:
        logger.exception(f"Conditional workflow example failed: {e}", exc_info=True)

    try:
        example_error_recovery()
    except Exception as e:
        logger.exception(f"Error recovery example failed: {e}", exc_info=True)

    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
