"""Logging utilities for MemCP."""

from memcp.utils.memcp_rich_theme import GRAPHITI_THEME

import logging

from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback


# Configure the root logger with rich handler
def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging with rich handler.

    Args:
        level: Logging level to use
    """
    # Create console for rich output
    logging_console = Console(color_system="auto", theme=GRAPHITI_THEME)

    # Install rich traceback handler
    install_rich_traceback(console=logging_console, show_locals=False)
    # # Configure logging with Rich to minimize noise and add colors
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=logging_console,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                markup=True,
                show_time=False,
                show_path=False,
                enable_link_path=False,  # Disable clickable file paths
                highlighter=ReprHighlighter(),
            )
        ],
    )


# Create a function to get a logger
def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Name for the logger

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
