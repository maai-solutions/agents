"""Rich logging configuration for the agent framework."""

import sys
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from typing import Optional


def setup_rich_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    show_path: bool = True,
    show_time: bool = True,
    rich_tracebacks: bool = True,
    console: Optional[Console] = None,
    use_stderr: bool = False
) -> Console:
    """Setup rich logging with loguru.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for detailed logs
        show_path: Show file path in console logs
        show_time: Show timestamp in console logs
        rich_tracebacks: Enable rich tracebacks with syntax highlighting
        console: Optional Rich Console instance (creates new if None)
        use_stderr: Force output to stderr instead of stdout (required for MCP servers)

    Returns:
        Console instance used for logging
    """
    # Create or use provided console
    if console is None:
        # Force stderr for MCP servers to avoid interfering with JSON-RPC protocol
        console = Console(stderr=use_stderr)

    # Install rich tracebacks globally
    if rich_tracebacks:
        install_rich_traceback(
            show_locals=True,
            width=console.width,
            extra_lines=3,
            theme="monokai",
            word_wrap=True,
            console=console
        )

    # Remove default loguru handlers
    logger.remove()

    # Add rich console handler
    logger.add(
        RichHandler(
            console=console,
            rich_tracebacks=rich_tracebacks,
            tracebacks_show_locals=True,
            markup=True,
            show_time=show_time,
            show_level=True,
            show_path=show_path
        ),
        format="{message}",
        level=level
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )

    return console


def log_with_panel(
    message: str,
    title: str = "",
    console: Optional[Console] = None,
    border_style: str = "blue"
):
    """Log a message in a rich panel for better visibility.

    Args:
        message: Message to display
        title: Panel title
        console: Console instance (creates new if None)
        border_style: Border color/style
    """
    from rich.panel import Panel

    if console is None:
        console = Console()

    console.print(Panel(message, title=title, border_style=border_style))


def log_with_table(data: list, title: str = "", console: Optional[Console] = None):
    """Log data in a rich table format.

    Args:
        data: List of dictionaries with table data
        title: Table title
        console: Console instance (creates new if None)
    """
    from rich.table import Table

    if console is None:
        console = Console()

    if not data:
        console.print("[yellow]No data to display[/yellow]")
        return

    # Create table with columns from first row
    table = Table(title=title, show_header=True, header_style="bold magenta")

    # Add columns
    for key in data[0].keys():
        table.add_column(str(key), style="cyan")

    # Add rows
    for row in data:
        table.add_row(*[str(v) for v in row.values()])

    console.print(table)


def log_metrics(metrics: dict, title: str = "Metrics", console: Optional[Console] = None):
    """Log metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
        console: Console instance (creates new if None)
    """
    from rich.table import Table

    if console is None:
        console = Console()

    table = Table(title=title, show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        # Format the key (convert snake_case to Title Case)
        formatted_key = key.replace("_", " ").title()
        table.add_row(formatted_key, str(value))

    console.print(table)


def log_progress(description: str, total: int = 100):
    """Create a rich progress bar for long-running tasks.

    Args:
        description: Task description
        total: Total number of steps

    Returns:
        Rich Progress instance
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )

    return progress


def log_tree(data: dict, title: str = "Data Structure", console: Optional[Console] = None):
    """Display nested data as a tree structure.

    Args:
        data: Nested dictionary to display
        title: Tree title
        console: Console instance (creates new if None)
    """
    from rich.tree import Tree

    if console is None:
        console = Console()

    def add_to_tree(tree, data):
        """Recursively add data to tree."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    branch = tree.add(f"[bold cyan]{key}[/bold cyan]")
                    add_to_tree(branch, value)
                else:
                    tree.add(f"[cyan]{key}[/cyan]: [yellow]{value}[/yellow]")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    branch = tree.add(f"[bold cyan]Item {i}[/bold cyan]")
                    add_to_tree(branch, item)
                else:
                    tree.add(f"[yellow]{item}[/yellow]")
        else:
            tree.add(f"[yellow]{data}[/yellow]")

    tree = Tree(f"[bold magenta]{title}[/bold magenta]")
    add_to_tree(tree, data)
    console.print(tree)


# Example usage
if __name__ == "__main__":
    # Setup logging
    console = setup_rich_logging(level="DEBUG", log_file="logs/test.log")

    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test panel logging
    log_with_panel("This is an important message!", title="Alert", border_style="red")

    # Test table logging
    data = [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "LA"}
    ]
    log_with_table(data, title="Users")

    # Test metrics logging
    metrics = {
        "total_requests": 1500,
        "success_rate": 0.95,
        "avg_response_time": 120.5,
        "errors": 75
    }
    log_metrics(metrics, title="API Metrics")

    # Test tree logging
    tree_data = {
        "agent": {
            "name": "ReasoningAgent",
            "tools": ["search", "calculator", "file_reader"],
            "config": {
                "temperature": 0.7,
                "max_tokens": 2048
            }
        }
    }
    log_tree(tree_data, title="Agent Configuration")
