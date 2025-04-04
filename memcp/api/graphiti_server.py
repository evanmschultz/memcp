"""Graphiti MCP Server implementation."""

from memcp.api.mcp_tools import MCPToolsRegistry, register_tools
from memcp.config.settings import GraphitiConfig, MCPConfig
from memcp.console.display_manager import DisplayManager
from memcp.console.queue_display import QueueProgressDisplay
from memcp.core.queue import QueueManager, QueueStatsTracker
from memcp.utils.shutdown import ShutdownManager

import asyncio
import os
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.fastmcp import FastMCP

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Welcome to Graphiti MCP - a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to
capture relationships between concepts, entities, and information. The system organizes data as episodes
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic,
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_episode tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations.
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality.
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid
API keys are provided for any language model operations.
"""  # noqa: E501


class GraphitiServer:
    """Graphiti MCP Server with proper dependency injection.

    Exposes Graphiti functionality through the Model Context Protocol (MCP)
    with proper dependency injection and no global variables.
    """

    def __init__(
        self,
        graphiti_config: GraphitiConfig,
        mcp_config: MCPConfig,
        display_manager: DisplayManager,
        shutdown_manager: ShutdownManager,
        logger: Any,
    ) -> None:
        """Initialize the server with all dependencies.

        Args:
            graphiti_config: Configuration for Graphiti
            mcp_config: Configuration for MCP server
            display_manager: Manager for console displays
            shutdown_manager: Manager for shutdown operations
            logger: Logger instance
        """
        self.graphiti_config = graphiti_config
        self.mcp_config = mcp_config
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
        if (
            not self.graphiti_config.neo4j_uri
            or not self.graphiti_config.neo4j_user
            or not self.graphiti_config.neo4j_password
        ):
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")

        self.graphiti_client = Graphiti(
            uri=self.graphiti_config.neo4j_uri,
            user=self.graphiti_config.neo4j_user,
            password=self.graphiti_config.neo4j_password,
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
        # This will be triggered whenever queue state changes
        # Use a debounced callback to prevent too many updates
        self.queue_manager.add_state_change_callback(
            # Simple lambda that only calls update_display with the queue_progress_display
            # We let the DisplayManager handle accessing the status object
            lambda: self.display_manager.update_display()
        )

    def initialize_mcp(self) -> None:
        """Initialize the MCP server instance."""
        # Create the MCP server instance
        self.mcp = FastMCP(
            "graphiti",
            instructions=GRAPHITI_MCP_INSTRUCTIONS,
            settings={"host": self.mcp_config.host, "port": self.mcp_config.port},
        )

        # Make sure we have necessary components initialized
        if not self.graphiti_client or not self.queue_manager:
            raise ValueError("GraphitiServer not fully initialized. Call initialize methods first.")

        # Create the tools registry
        self.tools_registry = MCPToolsRegistry(
            graphiti_client=self.graphiti_client,
            queue_manager=self.queue_manager,
            config=self.graphiti_config,
        )

        # Register all MCP tools
        register_tools(self.mcp, self.tools_registry)

        self.logger.info("MCP server initialized and tools registered successfully")

    async def _run_mcp_server(self, transport: str) -> None:
        """Run the MCP server with the specified transport.

        Args:
            transport: Transport type ("stdio" or "sse")
        """
        if not self.mcp:
            raise ValueError("MCP not initialized. Call initialize_mcp first.")

        if transport == "stdio":
            await self.mcp.run_stdio_async()
        elif transport == "sse":
            await self.mcp.run_sse_async()
        else:
            raise ValueError(f"Unsupported transport: {transport}")

    async def run(self) -> None:
        """Run the server."""
        if not self.mcp or not self.graphiti_client or not self.queue_manager:
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
        self.display_manager.show_server_info(self.mcp_config, self.graphiti_config.graph_id, pid)

        # Run the server
        try:
            # Create status object
            status_obj = self.display_manager.create_status_display(self.mcp_config)

            # Ensure we have a queue progress display
            if not self.queue_progress_display:
                raise ValueError("Queue progress display not initialized")

            # Run the appropriate server with the Live display
            await self.display_manager.run_with_live_display(
                self.queue_progress_display,
                status_obj,
                lambda: self._run_mcp_server(self.mcp_config.transport),
            )

        except asyncio.CancelledError:
            self.logger.info(
                "[success]Server shutdown initiated - tasks being cancelled (this is normal)[/success]"
            )
            # Let the exception propagate for proper shutdown
