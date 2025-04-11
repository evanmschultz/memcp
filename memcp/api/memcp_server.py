"""Graphiti MCP Server implementation."""

from memcp.api.api_errors import ServerInitializationError, UnsupportedTransportError
from memcp.api.mcp_tools import MCPToolsRegistry, register_tools
from memcp.config.memcp_config import MemCPConfig
from memcp.console.display_manager import DisplayManager
from memcp.console.queue_display import QueueProgressDisplay
from memcp.queue import QueueManager, QueueStatsTracker, ShutdownManager
from memcp.utils.uvicorn_config import get_uvicorn_config

import asyncio
import os
from logging import Logger

# from typing import NoReturn
from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.fastmcp import FastMCP


class MemCPServer:
    """Graphiti MCP Server with proper dependency injection.

    Exposes Graphiti functionality through the Model Context Protocol (MCP)
    with proper dependency injection and no global variables.
    """

    def __init__(
        self,
        config: MemCPConfig,
        display_manager: DisplayManager,
        shutdown_manager: ShutdownManager,
        logger: Logger,
    ) -> None:
        """Initialize the server with all dependencies.

        Args:
            config: Configuration for MemCP
            display_manager: Manager for console displays
            shutdown_manager: Manager for shutdown operations
            logger: Logger instance
        """
        self.config = config
        self.display_manager = display_manager
        self.shutdown_manager = shutdown_manager
        self.logger = logger

        # Components to be initialized later
        self.graphiti_client: Graphiti | None = None
        self.queue_manager: QueueManager | None = None
        self.queue_stats_tracker: QueueStatsTracker | None = None
        self.queue_progress_display: QueueProgressDisplay | None = None
        self.mcp: FastMCP | None = None
        self.tools_registry: MCPToolsRegistry | None = None

    async def initialize_graphiti(self, llm_client: LLMClient, destroy_graph: bool = False) -> None:
        """Initialize the Graphiti client.

        Args:
            llm_client: LLMClient for LLM operations
            destroy_graph: Whether to destroy existing graphs
        """
        if not self.config.neo4j.uri or not self.config.neo4j.user or not self.config.neo4j.password:
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")

        self.graphiti_client = Graphiti(
            uri=self.config.neo4j.uri,
            user=self.config.neo4j.user,
            password=self.config.neo4j.password.get_secret_value(),
            llm_client=llm_client,
        )

        if destroy_graph:
            self.logger.info("Destroying graph...")
            await clear_data(self.graphiti_client.driver)

        # Initialize the graph database with Graphiti's indices
        await self.graphiti_client.build_indices_and_constraints()
        self.logger.info("Graphiti client initialized successfully")

    def initialize_queue_components(
        self,
        queue_stats_tracker: QueueStatsTracker,
        queue_manager: QueueManager,
        queue_progress_display: QueueProgressDisplay,
    ) -> None:
        """Initialize queue-related components.

        Args:
            queue_stats_tracker: Statistics tracker for queues
            queue_manager: Manager for processing queues
            queue_progress_display: Display for queue progress
        """
        self.queue_stats_tracker = queue_stats_tracker
        self.queue_manager = queue_manager
        self.queue_progress_display = queue_progress_display

        # Add the display manager's update method as a callback
        # Use a wrapper function that doesn't return anything to fix type error
        def update_callback() -> None:
            asyncio.create_task(self.display_manager.debounced_update())

        self.queue_manager.add_state_change_callback(update_callback)

    def initialize_mcp(self) -> None:
        """Initialize the MCP server instance."""
        # Get the Uvicorn settings with proper logging configuration
        uvicorn_settings = get_uvicorn_config(
            host=self.config.server.host,
            port=self.config.server.port,
        )

        # Create the MCP server instance with custom settings
        self.mcp = FastMCP(
            self.config.mcp.name,
            instructions=self.config.mcp.instructions,
            settings=uvicorn_settings,
        )

        # Make sure we have necessary components initialized
        if not self.graphiti_client or not self.queue_manager:
            raise ServerInitializationError("GraphitiServer not fully initialized. Call initialize methods first.")

        # Create the tools registry
        self.tools_registry = MCPToolsRegistry(
            graphiti_client=self.graphiti_client,
            queue_manager=self.queue_manager,
            config=self.config,
        )

        # Register all MCP tools
        register_tools(self.mcp, self.tools_registry)

        self.logger.info("MCP server initialized and tools registered successfully")

    async def _run_mcp_server(self) -> None:
        """Run the MCP server with the specified transport."""
        if not self.mcp:
            raise ServerInitializationError("MCP not initialized. Call initialize_mcp first.")

        if self.config.server.transport == "stdio":
            await self.mcp.run_stdio_async()
        elif self.config.server.transport == "sse":
            await self.mcp.run_sse_async()
        else:
            raise UnsupportedTransportError(f"Unsupported transport: {self.config.server.transport}")

    async def run(self) -> None:
        """Run the server."""
        if not self.mcp or not self.graphiti_client or not self.queue_manager or not self.queue_progress_display:
            raise ValueError("Server not properly initialized. Call initialize methods first.")

        # Set up signal handlers
        self.shutdown_manager.setup_signal_handlers(
            self.queue_manager,
            self.queue_progress_display,
            self.graphiti_client,
            self.logger,
            self.display_manager,
        )

        # Initialize the queue progress display once before entering Live context
        if self.queue_progress_display:
            self.queue_progress_display.render()

        # Display the process ID and server info
        pid = os.getpid()

        self.display_manager.show_server_info(
            self.config.server,
            self.config.graph.id,
            pid,
            self.config.neo4j.user,
            self.config.llm.model_name,
        )

        # Run the server
        try:
            # Create status object
            status_obj = self.display_manager.create_status_display(self.config.server)

            # Ensure we have a queue progress display
            if not self.queue_progress_display:
                raise ValueError("Queue progress display not initialized")

            # Run the appropriate server with the Live display
            await self.display_manager.run_with_live_display(
                self.queue_progress_display,
                status_obj,
                lambda: self._run_mcp_server(),
            )

        except asyncio.CancelledError:
            self.logger.info("[success]Server shutdown initiated - tasks being cancelled (this is normal)[/success]")
            # Don't re-raise the exception - let the graceful shutdown complete
            # The ShutdownManager.graceful_shutdown method will handle this
            return
