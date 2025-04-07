"""Application builder for creating and configuring the application components."""

from memcp.api.memcp_server import MemCPServer
from memcp.config import ConfigManager, MemCPConfig
from memcp.config.settings import ConfigError, SecurityError
from memcp.console import DisplayManager, QueueProgressDisplay
from memcp.core.queue import QueueManager, QueueStatsTracker
from memcp.llm.llm_factory import LLMClientFactory
from memcp.utils import get_logger
from memcp.utils.shutdown import ShutdownManager

from graphiti_core.llm_client import LLMClient


class ApplicationBuilder:
    """Builder for creating and wiring application components.

    This builder is responsible for:
    1. Configuring the application from various sources
    2. Creating and configuring all application components
    3. Wiring dependencies together
    4. Providing a configured MemCPServer
    """

    __slots__ = (
        "logger",
        "_config_manager",
        "_memcp_config",
        "_mcp_config",
        "_llm_factory",
        "_llm_client",
        "_display_manager",
        "_shutdown_manager",
        "_queue_stats_tracker",
        "_queue_manager",
        "_queue_progress_display",
    )

    def __init__(self) -> None:
        """Initialize the application builder."""
        self.logger = get_logger(__name__)
        # Other attributes will be created on demand via properties

    def configure(self, config_path: str | None = None) -> None:
        """Configure the application.

        Args:
            config_path: Optional path to TOML config file
        """
        # Create and configure the config manager
        self._config_manager = ConfigManager.create_default(config_path)

        # Create the configuration objects
        self._memcp_config = self._config_manager.create_memcp_config()
        # self._mcp_config = self._config_manager.create_mcp_config()

    @property
    def config_manager(self) -> ConfigManager:
        """Get the configuration manager, creating it if necessary."""
        if not hasattr(self, "_config_manager"):
            self.configure()
        return self._config_manager

    @property
    def memcp_config(self) -> MemCPConfig:
        """Get the MemCP configuration, creating it if necessary."""
        if not hasattr(self, "_memcp_config"):
            self.configure()
        return self._memcp_config

    @property
    def llm_factory(self) -> LLMClientFactory:
        """Get the LLM factory, creating it if necessary."""
        if not hasattr(self, "_llm_factory"):
            self._llm_factory = LLMClientFactory()
        return self._llm_factory

    @property
    def llm_client(self) -> LLMClient:
        """Get the LLM client, creating it if necessary."""
        if not hasattr(self, "_llm_client"):
            self._llm_client = self.llm_factory.create_openai_client(
                api_key=self.memcp_config.openai.api_key.get_secret_value(), model=self.memcp_config.openai.model_name
            )
        return self._llm_client

    @property
    def display_manager(self) -> DisplayManager:
        """Get the display manager, creating it if necessary."""
        if not hasattr(self, "_display_manager"):
            self._display_manager = DisplayManager()
        return self._display_manager

    @property
    def shutdown_manager(self) -> ShutdownManager:
        """Get the shutdown manager, creating it if necessary."""
        if not hasattr(self, "_shutdown_manager"):
            self._shutdown_manager = ShutdownManager(self.display_manager.get_shutdown_console())
        return self._shutdown_manager

    @property
    def queue_stats_tracker(self) -> QueueStatsTracker:
        """Get the queue stats tracker, creating it if necessary."""
        if not hasattr(self, "_queue_stats_tracker"):
            self._queue_stats_tracker = QueueStatsTracker()
        return self._queue_stats_tracker

    @property
    def queue_manager(self) -> QueueManager:
        """Get the queue manager, creating it if necessary."""
        if not hasattr(self, "_queue_manager"):
            self._queue_manager = QueueManager(self.queue_stats_tracker)
        return self._queue_manager

    @property
    def queue_progress_display(self) -> QueueProgressDisplay:
        """Get the queue progress display, creating it if necessary."""
        if not hasattr(self, "_queue_progress_display"):
            self._queue_progress_display = QueueProgressDisplay(
                self.display_manager.get_main_console(),
                self.queue_stats_tracker,
            )
        return self._queue_progress_display

    async def build(self) -> MemCPServer:
        """Build the MemCP server with all components wired together.

        Returns:
            MemCPServer: Configured MemCPServer
        """
        server = MemCPServer(
            config=self.memcp_config,
            display_manager=self.display_manager,
            shutdown_manager=self.shutdown_manager,
            logger=self.logger,
        )

        destroy_graph = self.config_manager.should_destroy_graph()
        await server.initialize_graphiti(self.llm_client, destroy_graph=destroy_graph)
        server.initialize_queue_components(
            queue_stats_tracker=self.queue_stats_tracker,
            queue_manager=self.queue_manager,
            queue_progress_display=self.queue_progress_display,
        )
        server.initialize_mcp()

        return server


async def create_server(config_path: str | None = None) -> MemCPServer:
    """Create the MemCP server with all dependencies.

    Args:
        config_path: Optional path to TOML config file

    Returns:
        Configured MemCP server
    """
    logger = get_logger(__name__)

    try:
        # Load configuration
        config_manager = ConfigManager(toml_path=config_path)
        config = config_manager.create_memcp_config()

        # Create display manager
        shutdown_manager = ShutdownManager()
        display_manager = DisplayManager()

        # Create the server
        server = MemCPServer(
            config=config,
            display_manager=display_manager,
            shutdown_manager=shutdown_manager,
            logger=logger,
        )

        # Initialize queue components
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

        # Create LLM client
        llm_client = LLMClientFactory.create_openai_client(
            api_key=config.openai.api_key.get_secret_value(),
            model=config.openai.model_name,
        )

        # Initialize Graphiti
        await server.initialize_graphiti(llm_client, config.destroy_graph)

        # Initialize MCP
        server.initialize_mcp()

        return server
    except SecurityError as e:
        logger.error(f"Security error: {str(e)}")
        raise
    except ConfigError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error creating server: {str(e)}")
        raise
