#!/usr/bin/env python3
"""
Simple LLM Usage Example with Metrics, Logging, and Tracing

This script demonstrates how to use the linus module with:
1. Rich logging for beautiful console output
2. Telemetry/tracing for observability (Langfuse or OpenTelemetry)
3. Metrics tracking for performance monitoring

Usage:
    python src/run_llm.py
    python src/run_llm.py --query "What is 42 * 17?"
    python src/run_llm.py --no-trace  # Disable tracing
    python src/run_llm.py --exporter console  # Use console exporter instead of Langfuse
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.logging_config import (
    setup_rich_logging,
    log_with_panel,
    log_with_table,
    log_metrics,
    log_tree
)
from linus.agents.telemetry import initialize_telemetry
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple LLM usage with metrics, logging, and tracing"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is the result of 42 * 17? Also search for information about Python.",
        help="Query to send to the LLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LLM_MODEL", "gemma3:27b"),
        help="Model name to use"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        help="API base URL"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        help="Temperature for LLM"
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable telemetry/tracing"
    )
    parser.add_argument(
        "--exporter",
        type=str,
        default=os.getenv("TELEMETRY_EXPORTER", "langfuse"),
        choices=["langfuse", "console", "otlp", "jaeger"],
        help="Telemetry exporter type"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for grouping traces (optional)"
    )
    return parser.parse_args()


async def run_llm_example(
    query: str,
    model: str,
    api_base: str,
    temperature: float,
    enable_tracing: bool = True,
    exporter_type: str = "langfuse",
    session_id: Optional[str] = None
):
    """
    Run LLM example with metrics, logging, and tracing.

    Args:
        query: User query to process
        model: LLM model name
        api_base: API base URL
        temperature: LLM temperature
        enable_tracing: Whether to enable telemetry/tracing
        exporter_type: Type of telemetry exporter
        session_id: Optional session ID for trace grouping
    """
    # 1. SETUP RICH LOGGING
    console = setup_rich_logging(level="INFO")

    log_with_panel(
        "LLM Example with Metrics, Logging, and Tracing",
        title="🚀 Starting Run",
        border_style="bold blue"
    )

    # Display configuration
    config_data = [
        {"Setting": "Model", "Value": model},
        {"Setting": "API Base", "Value": api_base},
        {"Setting": "Temperature", "Value": temperature},
        {"Setting": "Tracing Enabled", "Value": enable_tracing},
        {"Setting": "Exporter", "Value": exporter_type if enable_tracing else "N/A"},
        {"Setting": "Session ID", "Value": session_id or "Not set"}
    ]

    log_with_table(
        config_data,
        title="⚙️ Configuration"
    )

    # 2. INITIALIZE TELEMETRY/TRACING
    tracer = None
    if enable_tracing:
        log_with_panel(
            f"Initializing {exporter_type} telemetry...",
            title="📊 Telemetry Setup",
            border_style="cyan"
        )

        try:
            tracer = initialize_telemetry(
                service_name="run_llm_example",
                exporter_type=exporter_type,
                enabled=True
            )
            logger.info(f"✅ Telemetry initialized with {exporter_type} exporter")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize telemetry: {e}")
            logger.info("Continuing without tracing...")
            tracer = None
    else:
        logger.info("Tracing disabled by user")

    # 3. CREATE AGENT WITH ALL FEATURES
    log_with_panel(
        "Creating agent with default tools...",
        title="🤖 Agent Setup",
        border_style="green"
    )

    tools = get_default_tools()
    tool_names = [tool.__class__.__name__ for tool in tools]

    log_tree(
        {
            "Agent Configuration": {
                "Model": model,
                "Temperature": temperature,
                "Tools": tool_names,
                "Async Mode": True,
                "Tracing": "Enabled" if tracer else "Disabled"
            }
        },
        title="📋 Agent Details"
    )

    agent = Agent(
        api_base=api_base,
        model=model,
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=temperature,
        max_tokens=2048,
        top_k=40,
        tools=tools,
        verbose=True,
        use_async=True,
        tracer=tracer,
        session_id=session_id
    )

    logger.info("✅ Agent created successfully")

    # 4. RUN THE QUERY
    log_with_panel(
        query,
        title="💬 User Query",
        border_style="bold yellow"
    )

    start_time = datetime.now()
    logger.info(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Execute the agent
        response = await agent.run(query)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 5. DISPLAY RESULTS
        log_with_panel(
            response.result,
            title="✨ Agent Response",
            border_style="bold green"
        )

        # 6. DISPLAY METRICS
        if response.metrics:
            log_metrics(
                response.metrics,
                title="📊 Performance Metrics"
            )

            # Additional calculated metrics
            success_rate = (response.metrics.successful_tool_calls / response.metrics.tool_executions) if response.metrics.tool_executions > 0 else 0
            additional_metrics = [
                {"Metric": "Wall Clock Time (s)", "Value": f"{duration:.2f}"},
                {"Metric": "Tokens/Second", "Value": f"{response.metrics.total_tokens / duration:.2f}" if duration > 0 else "N/A"},
                {"Metric": "Success Rate", "Value": f"{success_rate * 100:.1f}%"}
            ]

            log_with_table(
                additional_metrics,
                title="🎯 Additional Metrics"
            )

        # 7. DISPLAY EXECUTION HISTORY
        if response.execution_history:
            log_with_panel(
                "Task Execution Timeline",
                title="📜 Execution History",
                border_style="cyan"
            )

            history_data = []
            for idx, step in enumerate(response.execution_history, 1):
                history_data.append({
                    "Step": idx,
                    "Task": step.get("task", "N/A"),
                    "Tool": step.get("tool", "N/A"),
                    "Status": step.get("status", "N/A"),
                    "Result Preview": str(step.get("result", ""))[:50] + "..."
                })

            if history_data:
                # Display as tree for better readability
                history_tree = {}
                for item in history_data:
                    step_key = f"Step {item['Step']}: {item['Task']}"
                    history_tree[step_key] = {
                        "Tool": item['Tool'],
                        "Status": item['Status'],
                        "Result": item['Result Preview']
                    }

                log_tree(history_tree, title="🔄 Execution Steps")

        # 8. DISPLAY CITATIONS
        if response.citations:
            log_with_panel(
                f"Found {len(response.citations)} citation(s)",
                title="📚 Citations",
                border_style="blue"
            )

            citation_data = []
            for idx, citation in enumerate(response.citations, 1):
                citation_data.append({
                    "Index": idx,
                    "Document": citation.document_id,
                    "Chunk": citation.chunk_number,
                    "Score": f"{citation.score:.4f}" if citation.score else "N/A",
                    "Preview": citation.content_preview[:50] + "..." if len(citation.content_preview) > 50 else citation.content_preview
                })

            log_with_table(
                citation_data,
                title="📑 Citation Details"
            )
        else:
            logger.info("No citations found in response")

        # 9. DISPLAY COMPLETION STATUS
        if response.completion_status:
            status_style = "green" if response.completion_status.get("is_complete") else "yellow"
            log_with_panel(
                f"Complete: {response.completion_status.get('is_complete')}\n"
                f"Reasoning: {response.completion_status.get('reasoning', 'N/A')}",
                title="✅ Completion Status",
                border_style=status_style
            )

        logger.info(f"⏱️ Total execution time: {duration:.2f}s")

    except Exception as e:
        logger.error(f"❌ Error during execution: {e}", exc_info=True)
        raise

    finally:
        # 10. FLUSH TRACES
        if tracer:
            log_with_panel(
                "Flushing traces to telemetry backend...",
                title="🔄 Cleanup",
                border_style="cyan"
            )
            try:
                tracer.flush()
                logger.info("✅ Traces flushed successfully")

                if exporter_type == "langfuse":
                    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                    logger.info(f"📊 View traces at: {langfuse_host}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to flush traces: {e}")

    log_with_panel(
        "Run completed successfully! 🎉",
        title="✅ Done",
        border_style="bold green"
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Generate session ID if not provided
    session_id = args.session_id or f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    asyncio.run(
        run_llm_example(
            query=args.query,
            model=args.model,
            api_base=args.api_base,
            temperature=args.temperature,
            enable_tracing=not args.no_trace,
            exporter_type=args.exporter,
            session_id=session_id
        )
    )


if __name__ == "__main__":
    main()
