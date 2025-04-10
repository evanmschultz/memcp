"""Command-line interface for MemCP."""

from memcp.cli import run_server
from memcp.config import MemCPConfigBuilder
from memcp.utils import configure_logging, get_logger

import asyncio
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import CliApp
from rich.console import Console

# Load environment variables from multiple possible locations
found_dotenv = find_dotenv(usecwd=True)
if found_dotenv:
    load_dotenv(found_dotenv)
    print(f"Found .env file at: {found_dotenv}")
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


def main() -> None:
    """Main entry point for the MemCP CLI."""
    try:
        # Use CliApp to get configuration
        cli_return = CliApp.run(MemCPConfigBuilder)
        memcp_config = cli_return.to_memcp_config()

        # Run the server with the configuration
        asyncio.run(run_server(memcp_config))
    except Exception as e:
        # Handle any unexpected errors during startup
        console = Console()
        console.print(f"[bold red]Error during startup:[/] {str(e)}")
        logger.error(f"Error during startup: {str(e)}")
        sys.exit(1)
    finally:
        # Restore stderr to its original state (for Rich console)
        sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
