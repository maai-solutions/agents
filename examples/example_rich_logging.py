#!/usr/bin/env python3
"""Example demonstrating rich logging in the ReasoningAgent framework."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from linus.agents.logging_config import (
    setup_rich_logging,
    log_with_panel,
    log_with_table,
    log_metrics,
    log_tree
)
from linus.agents.agent import create_gemma_agent, get_default_tools
from loguru import logger
from dotenv import load_dotenv

# Load environment
load_dotenv()


def main():
    """Demonstrate rich logging features."""

    # Setup rich logging
    console = setup_rich_logging(
        level="INFO",
        log_file="logs/rich_demo.log",
        rich_tracebacks=True
    )

    # Display welcome message in a panel
    log_with_panel(
        "[bold green]ReasoningAgent Framework[/bold green]\n"
        "Demonstration of Rich Logging Features",
        title="🚀 Welcome",
        border_style="green",
        console=console
    )

    # Log some basic messages
    logger.info("[bold blue]Starting agent initialization...[/bold blue]")
    logger.debug("Loading environment variables")

    # Display configuration as a tree
    config_data = {
        "LLM": {
            "api_base": os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
            "model": os.getenv("LLM_MODEL", "gemma3:27b"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7"))
        },
        "Agent": {
            "verbose": os.getenv("AGENT_VERBOSE", "true"),
            "max_iterations": 10
        }
    }
    log_tree(config_data, title="⚙️  Agent Configuration", console=console)

    # Initialize agent
    try:
        tools = get_default_tools()

        # Display available tools in a table
        tools_data = [
            {"Tool": tool.name, "Description": tool.description}
            for tool in tools
        ]
        log_with_table(tools_data, title="🛠️  Available Tools", console=console)

        agent = create_gemma_agent(
            api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
            model=os.getenv("LLM_MODEL", "gemma3:27b"),
            tools=tools,
            verbose=True
        )

        logger.info("[green]✓[/green] Agent initialized successfully")

        # Run a test query
        log_with_panel(
            "What is 42 * 17?",
            title="📝 Test Query",
            border_style="cyan",
            console=console
        )

        # Execute the query
        logger.info("[bold yellow]Executing query...[/bold yellow]")
        response = agent.run("What is 42 * 17?", return_metrics=True)

        # Display result
        log_with_panel(
            f"[bold green]{response.result}[/bold green]",
            title="✅ Result",
            border_style="green",
            console=console
        )

        # Metrics are already displayed by the agent with rich formatting
        # But we can also display execution history
        if response.execution_history:
            history_data = [
                {
                    "Iteration": item["iteration"],
                    "Task": item["task"][:50] + "..." if len(item["task"]) > 50 else item["task"],
                    "Tool": item.get("tool", "None"),
                    "Status": item["status"]
                }
                for item in response.execution_history
            ]
            log_with_table(history_data, title="📊 Execution History", console=console)

        # Display completion status
        if response.completion_status:
            status_data = {
                "Complete": "✅ Yes" if response.completion_status["is_complete"] else "❌ No",
                "Reasoning": response.completion_status["reasoning"],
                "Next Action": response.completion_status.get("next_action", "N/A")
            }

            console.print("\n")
            log_tree(status_data, title="🎯 Completion Status", console=console)

        logger.info("[bold green]Demo complete![/bold green]")

    except Exception as e:
        logger.exception(f"[bold red]Error:[/bold red] {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
