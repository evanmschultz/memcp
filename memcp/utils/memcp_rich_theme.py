"""Defines the Rich theme for MemCP."""

import rich.theme

GRAPHITI_THEME = rich.theme.Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "danger": "bold red",
        "success": "bold green",
        "shutdown": "bold blue",
        "task": "dim magenta",
        "highlight": "bold cyan",
        "normal": "white",
        "step.success": "green",
        "step.warning": "yellow",
    }
)
