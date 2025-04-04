"""Server configuration for MemCP."""

from memcp.config.settings import GraphitiConfig, MCPConfig
from memcp.utils import get_logger

import argparse
import uuid

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient


class ServerConfig:
    """Configuration manager for the MemCP server.

    Handles command-line argument parsing, environment configuration,
    and LLM client creation.
    """

    def __init__(self) -> None:
        """Initialize the server configuration."""
        self.logger = get_logger(__name__)
        self.graphiti_config = GraphitiConfig.from_env()
        self.mcp_config: MCPConfig | None = None

    def create_llm_client(self, api_key: str | None = None, model: str | None = None) -> LLMClient:
        """Create an OpenAI LLM client.

        Args:
            api_key: API key for the OpenAI service
            model: Model name to use

        Returns:
            An instance of the OpenAI LLM client
        """
        # Create config with provided API key and model
        llm_config = LLMConfig(api_key=api_key or self.graphiti_config.openai_api_key or "")

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

    def configure_from_args(self, args: argparse.Namespace) -> tuple[MCPConfig, LLMClient | None]:
        """Configure the server from parsed arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            Tuple of (MCP config, LLM client)
        """
        # Set the graph_id from CLI argument or generate a random one
        if args.graph_id:
            self.graphiti_config.graph_id = args.graph_id
            self.logger.info(f"Using provided graph_id: {self.graphiti_config.graph_id}")
        else:
            self.graphiti_config.graph_id = f"graph_{uuid.uuid4().hex[:8]}"
            self.logger.info(f"Generated random graph_id: {self.graphiti_config.graph_id}")

        # Set use_custom_entities flag if specified
        if args.use_custom_entities:
            self.graphiti_config.use_custom_entities = True
            self.logger.info("Entity extraction enabled using predefined MEMCP_ENTITIES")
        else:
            self.logger.info("Entity extraction disabled (no custom entities will be used)")

        llm_client = None

        # Create OpenAI client if model is specified or if OPENAI_API_KEY is available
        if args.model or self.graphiti_config.openai_api_key:
            # Override model from command line if specified
            self.graphiti_config.model_name = args.model

            # Create the OpenAI client
            llm_client = self.create_llm_client(
                api_key=self.graphiti_config.openai_api_key, model=self.graphiti_config.model_name
            )

        # Create the MCP config
        self.mcp_config = MCPConfig(transport=args.transport, host=args.host, port=args.port)

        return self.mcp_config, llm_client
