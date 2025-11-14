from typing import Optional
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

class RichLogger:
    def __init__(
        self,
        level: str = "INFO",
        log_file: Optional[str] = None,
        show_path: bool = True,
        show_time: bool = True,
        rich_tracebacks: bool = True,
        console: Optional[Console] = None,
        use_stderr: bool = False
    ):

        console = console or Console(stderr=use_stderr)

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
            
        self.console = console

    def log_with_panel(
        self,
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

        if self.console is None:
            self.console = Console()

        self.console.print(Panel(message, title=title, border_style=border_style))

    def log_with_table(self, data: list, title: str = "", console: Optional[Console] = None):
        """Log data in a rich table format.

        Args:
            data: List of dictionaries with table data
            title: Table title
            console: Console instance (creates new if None)
        """
        from rich.table import Table

        if self.console is None:
            self.console = Console()

        if not data:
            self.console.print("[yellow]No data to display[/yellow]")
            return

        # Create table with columns from first row
        table = Table(title=title, show_header=True, header_style="bold magenta")

        # Add columns
        for key in data[0].keys():
            table.add_column(str(key), style="cyan")

        # Add rows
        for row in data:
            table.add_row(*[str(v) for v in row.values()])

        self.console.print(table)

    def log_metrics(self, metrics, title: str = "Metrics", console: Optional[Console] = None):
        """Log metrics in a formatted table.

        Args:
            metrics: Dictionary of metrics or AgentMetrics object
            title: Title for the metrics display
            console: Console instance (creates new if None)
        """
        from rich.table import Table

        if self.console is None:
            self.console = Console()

        # Convert AgentMetrics to dict if needed
        if hasattr(metrics, 'to_dict'):
            metrics = metrics.to_dict()

        table = Table(title=title, show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for key, value in metrics.items():
            # Format the key (convert snake_case to Title Case)
            formatted_key = key.replace("_", " ").title()
            table.add_row(formatted_key, str(value))

        self.console.print(table)

    def log_progress(self, description: str, total: int = 100):
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

    def log_tree(self, data: dict, title: str = "Data Structure", console: Optional[Console] = None):
        """Display nested data as a tree structure.

        Args:
            data: Nested dictionary to display
            title: Tree title
            console: Console instance (creates new if None)
        """
        from rich.tree import Tree

        if self.console is None:
            self.console = Console()

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
        self.console.print(tree)

    def debug(self, message: str, **kwargs) -> None:
        logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        logger.warning(message, **kwargs)   

    def error(self, message: str, **kwargs) -> None:
        logger.error(message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        logger.exception(message, **kwargs)

    

