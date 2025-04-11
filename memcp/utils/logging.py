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

    # Install rich traceback handler with custom suppress options
    install_rich_traceback(
        console=logging_console,
        show_locals=False,
        # Suppress specific modules from traceback to silence uvicorn/asyncio noise
        suppress=["uvicorn", "asyncio", "anyio"],
    )

    # Configure logging with Rich to minimize noise and add colors
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=logging_console,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                markup=True,
                show_time=False,
                show_path=True,
                highlighter=ReprHighlighter(),
                # Suppress specific modules to hide their tracebacks
                tracebacks_suppress=["uvicorn", "asyncio", "anyio"],
            )
        ],
    )

    # Disable uvicorn loggers to prevent duplicate output
    uvicorn_loggers = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.asgi",
    ]

    for logger_name in uvicorn_loggers:
        logger = logging.getLogger(logger_name)
        logger.handlers = []  # Remove default handlers
        logger.propagate = False  # Don't propagate to root logger
        logger.disabled = True  # Disable the logger completely

    # Also disable other loggers that might be noisy during normal operation
    other_loggers = ["asyncio", "starlette.lifespan", "starlette.middleware"]
    for logger_name in other_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)  # Only show error+ level messages


# Create a function to get a logger
def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Name for the logger

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
