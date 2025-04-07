#!/usr/bin/env python3
"""Command-line interface for MemCP."""

from memcp.app_builder import create_server
from memcp.config.settings import ConfigError, MemCPConfig, MissingCredentialsError, SecurityError
from memcp.utils import configure_logging, get_logger

import asyncio
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import CliApp
from rich.console import Console
from rich.panel import Panel

# Load environment variables from multiple possible locations
found_dotenv = find_dotenv(usecwd=True)
if found_dotenv:
    load_dotenv(found_dotenv)
else:
    # Try app directory
    app_dir = Path(__file__).parent.parent
    app_env = app_dir / ".env"
    if app_env.exists():
        load_dotenv(str(app_env))

# Configure logging
configure_logging()

# Get a logger for this module
logger = get_logger(__name__)


async def run_server(config_path: str | None = None) -> None:
    """Create and run the MemCP server asynchronously.

    Args:
        config_path: Optional path to TOML config file
    """
    console = Console()

    try:
        server = await create_server(config_path)
        await server.run()
    except SecurityError as e:
        console.print(Panel(f"[bold red]Security Error:[/] {str(e)}", title="Security Error", border_style="red"))
        logger.error(f"Security error: {str(e)}")
        sys.exit(1)
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


def main() -> None:
    """Main entry point for the MemCP CLI."""
    try:
        # Use CliApp to get configuration
        config = CliApp.run(MemCPConfig)

        # Get TOML config path if specified
        toml_path = config.config_path

        # Run the server with the configuration
        asyncio.run(run_server(toml_path))
    except Exception as e:
        # Handle any unexpected errors during startup
        console = Console()
        console.print(f"[bold red]Error during startup:[/] {str(e)}")
        logger.error(f"Error during startup: {str(e)}")
        sys.exit(1)
    finally:
        # Restore stderr to its original state (for Rich console)
        sys.stderr = sys.__stderr__
