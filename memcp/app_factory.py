"""Application factory for creating and configuring the application components."""

from memcp.api.graphiti_server import GraphitiServer
from memcp.config.settings import GraphitiConfig, MCPConfig
from memcp.console.display_manager import DisplayManager
from memcp.console.queue_display import QueueProgressDisplay
from memcp.core.queue import QueueManager, QueueStatsTracker
from memcp.utils import get_logger
from memcp.utils.shutdown import ShutdownManager

import argparse
import uuid

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient


class AppFactory:
    """Factory for creating and wiring application components.

    This factory is responsible for:
    1. Parsing command line arguments
    2. Creating and configuring all application components
    3. Wiring dependencies together
    4. Providing a configured GraphitiServer
    """

    def __init__(self) -> None:
        """Initialize the application factory."""
        self.logger = get_logger(__name__)

    def create_llm_client(self, api_key: str, model: str | None = None) -> LLMClient:
        """Create an OpenAI LLM client.

        Args:
            api_key: API key for the OpenAI service
            model: Model name to use

        Returns:
            An instance of the OpenAI LLM client
        """
        # Create config with provided API key and model
        llm_config = LLMConfig(api_key=api_key)

        # Set model if provided
        if model:
            llm_config.model = model

        # Create and return the client
        return OpenAIClient(config=llm_config)

    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments.

        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="Run the Graphiti MCP server with optional LLM client"
        )
        parser.add_argument(
            "--graph-name",
            help="Namespace for the graph. This is an arbitrary string used to organize related "
            "data. If not provided, a random UUID will be generated.",
        )
        parser.add_argument(
            "--transport",
            choices=["sse", "stdio"],
            default="sse",
            help="Transport to use for communication with the client. Default: sse",
        )
        parser.add_argument(
            "--model",
            default="gpt-4o-mini",
            help="Model name to use with the LLM client. Default: gpt-4o-mini",
        )
        parser.add_argument(
            "--destroy-graph", action="store_true", help="Destroy all Graphiti graphs"
        )
        parser.add_argument(
            "--use-custom-entities",
            action="store_true",
            help="Enable entity extraction using the predefined MEMCP_ENTITIES",
        )
        parser.add_argument(
            "--graph-id",
            help="Graph ID for the knowledge graph. If not provided, "
            "a random graph ID will be generated.",
        )
        parser.add_argument(
            "--host",
            default="0.0.0.0",
            help="Host address to bind the server to when using SSE transport. Default: 0.0.0.0",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port number to bind the server to when using SSE transport. Default: 8000",
        )

        return parser.parse_args()

    async def create_server(self) -> tuple[GraphitiServer, bool]:
        """Create and configure a GraphitiServer instance with all dependencies.

        Returns:
            Tuple of (configured GraphitiServer, destroy_graph flag)
        """
        # Parse arguments
        args = self.parse_args()

        # Create GraphitiConfig from environment
        graphiti_config = GraphitiConfig.from_env()

        # Set graph_id from args or generate a random one
        if args.graph_id:
            graphiti_config.graph_id = args.graph_id
            self.logger.info(f"Using provided graph_id: {graphiti_config.graph_id}")
        else:
            graphiti_config.graph_id = f"graph_{uuid.uuid4().hex[:8]}"
            self.logger.info(f"Generated random graph_id: {graphiti_config.graph_id}")

        # Set custom entities flag
        if args.use_custom_entities:
            graphiti_config.use_custom_entities = True
            self.logger.info("Entity extraction enabled using predefined MEMCP_ENTITIES")
        else:
            self.logger.info("Entity extraction disabled (no custom entities will be used)")

        # Override model from args if specified
        if args.model:
            graphiti_config.model_name = args.model

        # Create MCP config
        mcp_config = MCPConfig(transport=args.transport, host=args.host, port=args.port)

        # Create display components
        display_manager = DisplayManager()

        # Create shutdown manager
        shutdown_manager = ShutdownManager(display_manager.get_shutdown_console())

        # Create queue components
        queue_stats_tracker = QueueStatsTracker()
        queue_manager = QueueManager(queue_stats_tracker)
        queue_progress_display = QueueProgressDisplay(
            display_manager.get_main_console(), queue_stats_tracker
        )

        # Create LLM client
        if not graphiti_config.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        llm_client = self.create_llm_client(
            api_key=graphiti_config.openai_api_key, model=graphiti_config.model_name
        )

        # Create server instance
        server = GraphitiServer(
            graphiti_config=graphiti_config,
            mcp_config=mcp_config,
            display_manager=display_manager,
            shutdown_manager=shutdown_manager,
            logger=self.logger,
        )

        # Initialize server components
        await server.initialize_graphiti(llm_client, destroy_graph=args.destroy_graph)
        server.initialize_queue_components(
            queue_stats_tracker=queue_stats_tracker,
            queue_manager=queue_manager,
            queue_progress_display=queue_progress_display,
        )
        server.initialize_mcp()

        return server, args.destroy_graph


async def create_server() -> GraphitiServer:
    """Create a fully configured GraphitiServer instance.

    Returns:
        Configured GraphitiServer
    """
    factory = AppFactory()
    server, _ = await factory.create_server()
    return server
