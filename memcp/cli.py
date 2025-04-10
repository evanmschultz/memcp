#!/usr/bin/env python3
"""Command-line interface for MemCP."""

from memcp.api.memcp_server import MemCPServer
from memcp.config import MemCPConfig, MemCPConfigBuilder, MissingCredentialsError
from memcp.config.config_errors import ConfigError
from memcp.console import DisplayManager, QueueProgressDisplay
from memcp.llm.llm_factory import LLMClientFactory
from memcp.queue import QueueManager, QueueStatsTracker
from memcp.utils import configure_logging, get_logger
from memcp.utils.shutdown import ShutdownManager

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


async def create_server(config: MemCPConfig) -> MemCPServer:
    """Create the MemCP server with all dependencies.

    Args:
        config (MemCPConfig): The MemCP configuration object

    Returns:
        Configured MemCP server
    """
    try:
        shutdown_manager = ShutdownManager()
        display_manager = DisplayManager()

        server = MemCPServer(
            config=config,
            display_manager=display_manager,
            shutdown_manager=shutdown_manager,
            logger=logger,
        )

        queue_stats_tracker = QueueStatsTracker()
        queue_manager = QueueManager(queue_stats_tracker)
        queue_progress_display = QueueProgressDisplay(
            display_manager.get_main_console(),
            queue_stats_tracker,
        )
        server.initialize_queue_components(
            queue_stats_tracker=queue_stats_tracker,
            queue_manager=queue_manager,
            queue_progress_display=queue_progress_display,
        )

        llm_factory = LLMClientFactory(config=config)
        llm_client = llm_factory.create_client()

        await server.initialize_graphiti(llm_client, config.destroy_graph)
        server.initialize_mcp()

        return server
    except ConfigError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error creating server: {str(e)}")
        raise


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
