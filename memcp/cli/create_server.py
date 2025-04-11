"""Command-line interface for MemCP."""

from memcp.api.memcp_server import MemCPServer
from memcp.config import MemCPConfig
from memcp.config.errors import ConfigError
from memcp.console import DisplayManager, QueueProgressDisplay
from memcp.llm.llm_factory import LLMClientFactory
from memcp.queue import QueueManager, QueueStatsTracker, ShutdownManager
from memcp.utils import get_logger

import rich.console

# Get a logger for this module
logger = get_logger(__name__)


async def create_server(config: MemCPConfig, console: rich.console.Console) -> MemCPServer:
    """Create the MemCP server with all dependencies.

    Args:
        config (MemCPConfig): The MemCP configuration object
        console (rich.console.Console): The rich console object

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
