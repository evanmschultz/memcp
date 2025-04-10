"""Command-line interface for MemCP."""

from memcp.cli.create_server import create_server
from memcp.config import MemCPConfig
from memcp.config.errors import ConfigError, MissingCredentialsError
from memcp.utils import get_logger

import sys

from rich.console import Console
from rich.panel import Panel

# Get a logger for this module
logger = get_logger(__name__)


async def run_server(config: MemCPConfig) -> None:
    """Create and run the MemCP server asynchronously.

    Args:
        config (MemCPConfig): The MemCP configuration object
    """
    console = Console()

    try:
        server = await create_server(config)
        await server.run()
    except MissingCredentialsError as e:
        console.print(
            Panel(f"[bold yellow]Missing Credentials:[/] {str(e)}", title="Configuration Error", border_style="yellow")
        )
        logger.error(f"Missing credentials: {str(e)}")
        sys.exit(1)
    except ConfigError as e:
        console.print(
            Panel(f"[bold orange]Configuration Error:[/] {str(e)}", title="Configuration Error", border_style="orange")
        )
        logger.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/] {str(e)}", title="Error", border_style="red"))
        logger.error(f"Error running MemCP server: {str(e)}")
        sys.exit(1)
