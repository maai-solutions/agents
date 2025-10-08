# Rich Logging Guide

This guide explains how to use Rich logging features in the ReasoningAgent framework.

## Overview

The framework uses [Rich](https://rich.readthedocs.io/) library to provide beautiful, formatted console output with:

- 🎨 Colored and styled text
- 📊 Tables for structured data
- 📦 Panels for important messages
- 🌳 Tree views for nested data
- 📈 Progress bars for long operations
- 🐛 Enhanced tracebacks with syntax highlighting

## Installation

Rich is included in the requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Setup

```python
from linus.agents.logging_config import setup_rich_logging

# Setup rich logging with default settings
console = setup_rich_logging(
    level="INFO",
    log_file="logs/app.log",
    rich_tracebacks=True
)
```

### Using Rich Logging in Your Code

```python
from loguru import logger

# Rich markup is supported in log messages
logger.info("[bold green]Operation successful![/bold green]")
logger.warning("[yellow]This is a warning[/yellow]")
logger.error("[bold red]Error occurred[/bold red]")
```

## Rich Logging Utilities

### Panel Logging

Display important messages in a bordered panel:

```python
from linus.agents.logging_config import log_with_panel

log_with_panel(
    "This is an important message!",
    title="Alert",
    border_style="red"
)
```

### Table Logging

Display structured data in tables:

```python
from linus.agents.logging_config import log_with_table

data = [
    {"name": "Alice", "role": "Engineer", "tasks": 42},
    {"name": "Bob", "role": "Designer", "tasks": 28}
]

log_with_table(data, title="Team Members")
```

### Metrics Display

Format and display metrics nicely:

```python
from linus.agents.logging_config import log_metrics

metrics = {
    "total_requests": 1500,
    "success_rate": 0.95,
    "avg_response_time": 120.5,
    "errors": 75
}

log_metrics(metrics, title="API Performance")
```

### Tree View

Display nested data structures:

```python
from linus.agents.logging_config import log_tree

config = {
    "agent": {
        "name": "ReasoningAgent",
        "tools": ["search", "calculator"],
        "config": {
            "temperature": 0.7,
            "max_tokens": 2048
        }
    }
}

log_tree(config, title="Configuration")
```

## Agent Integration

The ReasoningAgent automatically uses Rich formatting when available:

```python
from linus.agents.agent import create_gemma_agent, get_default_tools

agent = create_gemma_agent(
    api_base="http://localhost:11434/v1",
    model="gemma3:27b",
    tools=get_default_tools(),
    verbose=True
)

# Metrics will be displayed in a beautiful table
response = agent.run("Calculate 42 * 17", return_metrics=True)
```

### Rich Metrics Display

When `RICH` is available, agent metrics are displayed as formatted tables with:
- Total iterations and tokens
- Execution time
- LLM and tool call statistics
- Success rates
- Visual indicators (✅/❌)

## FastAPI Integration

The FastAPI app (`src/app.py`) is configured with rich logging by default:

```python
# Rich console handler is automatically configured
python src/app.py
```

This provides:
- Colored HTTP request logs
- Formatted exception tracebacks
- Better error visibility

## Console Customization

### Custom Console

Create a custom console with specific settings:

```python
from rich.console import Console

console = Console(
    width=120,           # Console width
    force_terminal=True, # Force color output
    color_system="auto", # Auto-detect color support
    theme=my_theme      # Custom theme
)
```

### Custom Themes

Define custom color themes:

```python
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green"
})

console = Console(theme=custom_theme)
```

## Progress Bars

For long-running operations:

```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("[cyan]Processing...", total=100)

    for i in range(100):
        # Do work
        progress.update(task, advance=1)
```

## Advanced Features

### Syntax Highlighting

Rich automatically highlights code in tracebacks:

```python
# Errors will show syntax-highlighted code
raise ValueError("Something went wrong")
```

### Live Display

Create live-updating displays:

```python
from rich.live import Live
from rich.table import Table

with Live(auto_refresh=False) as live:
    for i in range(10):
        table = Table()
        table.add_column("Iteration")
        table.add_column("Status")
        table.add_row(str(i), "Processing...")
        live.update(table, refresh=True)
```

### Markdown Rendering

Render markdown in the console:

```python
from rich.markdown import Markdown

markdown = Markdown("""
# Title
- Item 1
- Item 2
""")

console.print(markdown)
```

## Example Usage

See the complete example in:

```bash
python example_rich_logging.py
```

This demonstrates:
- Panel messages for headers
- Tables for tools and execution history
- Tree views for configuration
- Metrics display
- Rich tracebacks

## Configuration Options

### Logging Levels

```python
setup_rich_logging(level="DEBUG")  # Show all logs
setup_rich_logging(level="INFO")   # Default
setup_rich_logging(level="WARNING") # Warnings and errors only
```

### File Logging

```python
setup_rich_logging(
    log_file="logs/app.log",  # Enable file logging
    level="INFO"               # Console level
)
# File always logs at DEBUG level
```

### Disable Rich Features

If you prefer plain logging:

```python
# Don't import rich logging utilities
# Use standard loguru without rich handler

from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")
```

## Troubleshooting

### Rich Not Available

If Rich is not installed, the agent falls back to standard logging:

```bash
pip install rich
```

### Color Not Showing

Check terminal color support:

```python
from rich.console import Console

console = Console()
print(f"Color system: {console.color_system}")
print(f"Is terminal: {console.is_terminal}")
```

Force colors:

```python
console = Console(force_terminal=True)
```

### Width Issues

Adjust console width:

```python
console = Console(width=120)  # Set explicit width
```

## Best Practices

1. **Use markup sparingly**: Don't overuse colors/styles
2. **Table for data**: Use tables for structured data
3. **Panels for important info**: Use panels for alerts/headers
4. **Tree for nested data**: Use trees for hierarchical data
5. **Progress for long tasks**: Show progress for operations > 5s

## Environment Variables

Control rich logging via environment:

```bash
# Disable rich features
export RICH_ENABLED=false

# Force color output
export FORCE_COLOR=1

# Set console width
export COLUMNS=120
```

## API Reference

### `setup_rich_logging()`

Setup rich logging with loguru.

**Parameters:**
- `level` (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `log_file` (str, optional): Path to log file
- `show_path` (bool): Show file path in logs
- `show_time` (bool): Show timestamp
- `rich_tracebacks` (bool): Enable rich tracebacks
- `console` (Console, optional): Custom console instance

**Returns:** Console instance

### `log_with_panel()`

Display message in a panel.

**Parameters:**
- `message` (str): Message to display
- `title` (str): Panel title
- `console` (Console, optional): Console instance
- `border_style` (str): Border color/style

### `log_with_table()`

Display data in a table.

**Parameters:**
- `data` (list): List of dicts with table data
- `title` (str): Table title
- `console` (Console, optional): Console instance

### `log_metrics()`

Display metrics in a table.

**Parameters:**
- `metrics` (dict): Metrics dictionary
- `title` (str): Title for metrics
- `console` (Console, optional): Console instance

### `log_tree()`

Display nested data as tree.

**Parameters:**
- `data` (dict): Nested dictionary
- `title` (str): Tree title
- `console` (Console, optional): Console instance

## Additional Resources

- [Rich Documentation](https://rich.readthedocs.io/)
- [Loguru Documentation](https://loguru.readthedocs.io/)
- [Rich GitHub](https://github.com/Textualize/rich)

## Examples

See example implementations:
- `example_rich_logging.py` - Rich logging demo
- `src/app.py` - FastAPI with rich logging
- `src/linus/agents/logging_config.py` - Logging utilities
- `src/linus/agents/agent/reasoning_agent.py` - Agent with rich metrics
