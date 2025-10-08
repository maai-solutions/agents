#!/usr/bin/env python3
"""Quick test of Rich logging features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from linus.agents.logging_config import (
    setup_rich_logging,
    log_with_panel,
    log_with_table,
    log_metrics,
    log_tree
)
from loguru import logger

# Setup rich logging
console = setup_rich_logging(level="INFO")

# Test panel
log_with_panel(
    "[bold green]Rich Logging Test[/bold green]\nThis demonstrates beautiful console output!",
    title="🎨 Welcome",
    border_style="blue"
)

# Test colored logs
logger.info("[bold blue]Testing colored info message[/bold blue]")
logger.warning("[yellow]Testing warning message[/yellow]")
logger.success("[bold green]Testing success message[/bold green]")

# Test metrics
metrics = {
    "total_tokens": 1245,
    "execution_time": 2.34,
    "success_rate": 0.95,
    "iterations": 3
}
log_metrics(metrics, title="📊 Test Metrics")

# Test table
data = [
    {"Feature": "Colored Output", "Status": "✅ Working"},
    {"Feature": "Tables", "Status": "✅ Working"},
    {"Feature": "Panels", "Status": "✅ Working"},
    {"Feature": "Metrics", "Status": "✅ Working"}
]
log_with_table(data, title="🧪 Feature Status")

# Test tree
tree_data = {
    "Rich Logging": {
        "Console": ["Colors", "Formatting", "Emojis"],
        "Structures": ["Tables", "Panels", "Trees"],
        "Features": {
            "Tracebacks": "Enhanced",
            "Progress": "Bars & Spinners"
        }
    }
}
log_tree(tree_data, title="🌳 Feature Tree")

logger.info("[bold green]✅ All Rich logging features are working![/bold green]")
