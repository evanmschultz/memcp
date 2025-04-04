#!/usr/bin/env python3
"""Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)."""

from memcp.config.settings import GraphitiConfig, MCPConfig
from memcp.console.queue_display import QueueProgressDisplay
from memcp.core.queue import QueueManager, QueueStatsTracker
from memcp.memcp_typings import MEMCP_ENTITIES
from memcp.models.responses import (
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeResult,
    NodeSearchResponse,
    StatusResponse,
    SuccessResponse,
)
from memcp.utils import GRAPHITI_THEME, configure_logging, get_logger

import argparse
import asyncio
import atexit
import contextlib
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, cast

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

load_dotenv()


# Configure logging
configure_logging()

# Get a logger for this module
logger = get_logger(__name__)


shutdown_console = Console(stderr=True, theme=GRAPHITI_THEME)
progress_console = Console(theme=GRAPHITI_THEME)  # New console for progress display


# Set up a global variable to track if we're shutting down
_original_excepthook = sys.excepthook
_shutdown_mode = "NONE"  # Can be "NONE", "GRACEFUL", or "FORCE"


# Create a custom excepthook that will intercept unhandled exceptions
def custom_excepthook(exc_type, exc_value, exc_traceback) -> None:
    """Custom excepthook that intercepts unhandled exceptions and shows a final message."""
    # First, call the original excepthook to show the traceback
    _original_excepthook(exc_type, exc_value, exc_traceback)

    # If this is a shutdown-related exception, show our final message
    if _shutdown_mode != "NONE" and issubclass(exc_type, asyncio.CancelledError):
        # Wait a moment to ensure traceback is fully printed
        time.sleep(0.1)
        show_final_message()


# Register our custom excepthook
sys.excepthook = custom_excepthook


def show_final_message() -> None:
    """Display a final reassuring message after all tracebacks."""
    print("\n\n" + "-" * 80)
    if _shutdown_mode == "GRACEFUL":
        shutdown_console.print(
            "[bold green]✅ GRAPHITI SERVER SHUTDOWN SUCCESSFULLY[/bold green]\n"
        )
        shutdown_console.print(
            "\n[info]The asyncio CancelledError tracebacks above are normal and expected "
            "during shutdown.[/info]"
        )
        shutdown_console.print(
            "\n[info]All tasks were properly cancelled and resources were released "
            "cleanly.[/info]\n"
        )
    elif _shutdown_mode == "FORCE":
        shutdown_console.print("[bold red]⚠️ GRAPHITI SERVER FORCE KILLED[/bold red]\n")
        shutdown_console.print(
            "\n[dim warning]No cleanup was performed. Some resources may not have been properly "
            "released.[/dim warning]"
        )
    else:
        shutdown_console.print("[bold green]GRAPHITI SERVER SHUTDOWN COMPLETE[/bold green]\n")
    print("-" * 80 + "\n")


# Register an atexit handler that will show our message as the very last thing
atexit.register(show_final_message)


# Configure root logger to ignore asyncio cancellation errors during shutdown
logging.getLogger("asyncio").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Create global config instance
config: GraphitiConfig = GraphitiConfig.from_env()

# Global references to important components
graphiti_client: Graphiti | None = None
queue_manager: QueueManager | None = None


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


# MCP server instance
mcp = FastMCP(
    "graphiti",
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
    settings={"host": "0.0.0.0", "port": 8000},
)


async def initialize_graphiti(
    llm_client: LLMClient | None = None, destroy_graph: bool = False
) -> None:
    """Initialize the Graphiti client with the provided settings.

    Args:
        llm_client: Optional LLMClient instance to use for LLM operations
        destroy_graph: Optional boolean to destroy all Graphiti graphs
    """
    global graphiti_client

    # If no client is provided, create a default OpenAI client
    if not llm_client:
        if config.openai_api_key:
            llm_config = LLMConfig(api_key=config.openai_api_key)
            if config.openai_base_url:
                llm_config.base_url = config.openai_base_url
            if config.model_name:
                llm_config.model = config.model_name
            llm_client = OpenAIClient(config=llm_config)
        else:
            raise ValueError("OPENAI_API_KEY must be set when not using a custom LLM client")

    if not config.neo4j_uri or not config.neo4j_user or not config.neo4j_password:
        raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")

    graphiti_client = Graphiti(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        llm_client=llm_client,
    )

    if destroy_graph:
        logger.info("Destroying graph...")
        await clear_data(graphiti_client.driver)

    # Initialize the graph database with Graphiti's indices
    await graphiti_client.build_indices_and_constraints()
    logger.info("Graphiti client initialized successfully")


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    return edge.model_dump(
        mode="json",
        exclude={
            "fact_embedding",
        },
    )


@mcp.tool()
async def add_episode(
    name: str,
    episode_body: str,
    graph_id: str | None = None,
    source: str = "text",
    source_description: str = "",
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    r"""Add an episode to the Graphiti knowledge graph. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode. When source='json', this must be a properly escaped JSON string,
                           not a raw Python dictionary. The JSON data will be automatically processed
                           to extract entities and relationships.
        graph_id (str, optional): A unique ID for this graph. If not provided, uses the default graph_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

    Examples:
        # Adding plain text content
        add_episode(
            name="Company News",
            episode_body="Acme Corp announced a new product line today.",
            source="text",
            source_description="news article",
            graph_id="some_arbitrary_string"
        )

        # Adding structured JSON data
        # NOTE: episode_body must be a properly escaped JSON string. Note the triple backslashes
        add_episode(
            name="Customer Profile",
            episode_body="{\\\"company\\\": {\\\"name\\\": \\\"Acme Technologies\\\"}, \\\"products\\\": [{\\\"id\\\": \\\"P001\\\", \\\"name\\\": \\\"CloudSync\\\"}, {\\\"id\\\": \\\"P002\\\", \\\"name\\\": \\\"DataMiner\\\"}]}",
            source="json",
            source_description="CRM data"
        )

        # Adding message-style content
        add_episode(
            name="Customer Conversation",
            episode_body="user: What's your return policy?\nassistant: You can return items within 30 days.",
            source="message",
            source_description="chat transcript",
            graph_id="some_arbitrary_string"
        )

    Notes:
        When using source='json':
        - The JSON must be a properly escaped string, not a raw Python dictionary
        - The JSON will be automatically processed to extract entities and relationships
        - Complex nested structures are supported (arrays, nested objects, mixed data types), but keep nesting to a minimum
        - Entities will be created from appropriate JSON properties
        - Relationships between entities will be established based on the JSON structure
    """  # noqa: E501
    global graphiti_client, queue_manager
    if graphiti_client is None:
        return {"error": "Graphiti client not initialized"}

    if queue_manager is None:
        return {"error": "Queue manager not initialized"}

    try:
        # Map string source to EpisodeType enum
        source_type = EpisodeType.text
        if source.lower() == "message":
            source_type = EpisodeType.message
        elif source.lower() == "json":
            source_type = EpisodeType.json

        # Use the provided graph_id or fall back to the default from config
        effective_graph_id = graph_id if graph_id is not None else config.graph_id

        # Cast graph_id to str to satisfy type checker
        # The Graphiti client expects a str for graph_id, not Optional[str]
        graph_id_str = str(effective_graph_id) if effective_graph_id is not None else ""

        client = cast(Graphiti, graphiti_client)

        # Define the episode processing function
        async def process_episode() -> None:
            try:
                logger.info(
                    f"Processing queued episode '[highlight]{name}[/highlight]' for graph_id: [highlight]{graph_id_str}[/highlight]"
                )
                # Use all entity types if use_custom_entities is enabled, otherwise use empty dict
                entity_types = MEMCP_ENTITIES if config.use_custom_entities else {}

                # Cast to dict[str, BaseModel] to match the expected type in Graphiti
                # This is safe because our MemCPEntityTypes satisfies this interface
                await client.add_episode(
                    name=name,
                    episode_body=episode_body,
                    source=source_type,
                    source_description=source_description,
                    group_id=graph_id_str,  # Using the string version of graph_id
                    uuid=uuid,
                    reference_time=datetime.now(timezone.utc),
                    entity_types=cast("dict[str, BaseModel]", entity_types),
                )
                logger.info(
                    f"Episode '[highlight]{name}[/highlight]' [success]added successfully[/success]"
                )

                logger.info(f"Building communities after episode '[highlight]{name}[/highlight]'")
                await client.build_communities()

                logger.info(
                    f"Episode '[highlight]{name}[/highlight]' [success]processed successfully[/success]"
                )
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"[danger]Error[/danger] processing episode '[highlight]{name}[/highlight]' for graph_id [highlight]{graph_id_str}[/highlight]: [danger]{error_msg}[/danger]"
                )

        # Enqueue the task for processing
        await queue_manager.enqueue_task(graph_id_str, process_episode)

        # Return immediately with a success message
        # Use the queue size from the queue manager's queue for this group_id
        queue_size = 0
        if graph_id_str in queue_manager.episode_queues:
            queue_size = queue_manager.episode_queues[graph_id_str].qsize()

        return {
            "message": f"Episode '[highlight]{name}[/highlight]' queued for processing (position: [highlight]{queue_size}[/highlight])"
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[danger]Error queuing episode task[/danger]: {error_msg}")
        return {"error": f"Error queuing episode task: {error_msg}"}


@mcp.tool()
async def search_nodes(
    query: str,
    graph_ids: list[str] | None = None,
    max_nodes: int = 10,
    center_node_uuid: str | None = None,
    entity: str = "",  # cursor seems to break with None
) -> NodeSearchResponse | ErrorResponse:
    """Search the Graphiti knowledge graph for relevant node summaries.

    These contain a summary of all of a node's relationships with other nodes.

    Note: entity is a single entity type to filter results (permitted: "Preference", "Procedure").

    Args:
        query: The search query
        graph_ids: Optional list of group IDs to filter results
        max_nodes: Maximum number of nodes to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
        entity: Optional single entity type to filter results (permitted: "Preference", "Procedure")
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    try:
        # Use the provided graph_ids or fall back to the default from config if none provided
        effective_graph_ids = (
            graph_ids if graph_ids is not None else [config.graph_id] if config.graph_id else []
        )

        # Configure the search
        if center_node_uuid is not None:
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
        else:
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        filters = SearchFilters()
        if entity != "":
            filters.node_labels = [entity]

        client = cast(Graphiti, graphiti_client)

        # Perform the search using the _search method
        search_results = await client._search(
            query=query,
            config=search_config,
            group_ids=effective_graph_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )

        if not search_results.nodes:
            return NodeSearchResponse(message="No relevant nodes found", nodes=[])

        # Format the node results
        formatted_nodes: list[NodeResult] = [
            {
                "uuid": node.uuid,
                "name": node.name,
                "summary": node.summary if hasattr(node, "summary") else "",
                "labels": node.labels if hasattr(node, "labels") else [],
                "group_id": node.group_id,
                "created_at": node.created_at.isoformat(),
                "attributes": node.attributes if hasattr(node, "attributes") else {},
            }
            for node in search_results.nodes
        ]

        return NodeSearchResponse(message="Nodes retrieved successfully", nodes=formatted_nodes)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error searching nodes: {error_msg}")
        return ErrorResponse(error=f"Error searching nodes: {error_msg}")


@mcp.tool()
async def search_facts(
    query: str,
    graph_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the Graphiti knowledge graph for relevant facts.

    Args:
        query: The search query
        graph_ids: Optional list of group IDs to filter results
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    global graphiti_client

    if graphiti_client is None:
        return {"error": "Graphiti client not initialized"}

    try:
        # Use the provided graph_ids or fall back to the default from config if none provided
        effective_graph_ids = (
            graph_ids if graph_ids is not None else [config.graph_id] if config.graph_id else []
        )

        client = cast(Graphiti, graphiti_client)

        relevant_edges = await client.search(
            group_ids=effective_graph_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return {"message": "No relevant facts found", "facts": []}

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return {"message": "Facts retrieved successfully", "facts": facts}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error searching facts: {error_msg}")
        return {"error": f"Error searching facts: {error_msg}"}


@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the Graphiti knowledge graph.

    Args:
        uuid: UUID of the entity edge to delete
    """
    global graphiti_client

    if graphiti_client is None:
        return {"error": "Graphiti client not initialized"}

    try:
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return {"message": f"Entity edge with UUID {uuid} deleted successfully"}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error deleting entity edge: {error_msg}")
        return {"error": f"Error deleting entity edge: {error_msg}"}


@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the Graphiti knowledge graph.

    Args:
        uuid: UUID of the episode to delete
    """
    global graphiti_client

    if graphiti_client is None:
        return {"error": "Graphiti client not initialized"}

    try:
        client = cast(Graphiti, graphiti_client)

        # Get the episodic node by UUID - EpisodicNode is already imported at the top
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return {"message": f"Episode with UUID {uuid} deleted successfully"}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error deleting episode: {error_msg}")
        return {"error": f"Error deleting episode: {error_msg}"}


@mcp.tool()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the Graphiti knowledge graph by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    global graphiti_client

    if graphiti_client is None:
        return {"error": "Graphiti client not initialized"}

    try:
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting entity edge: {error_msg}")
        return {"error": f"Error getting entity edge: {error_msg}"}


@mcp.tool()
async def get_episodes(
    graph_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent episodes for a specific group.

    Args:
        graph_id: ID of the group to retrieve episodes from. If not provided, uses the default
            graph_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    global graphiti_client

    if graphiti_client is None:
        return {"error": "Graphiti client not initialized"}

    try:
        # Use the provided graph_id or fall back to the default from config
        effective_graph_id = graph_id if graph_id is not None else config.graph_id

        if not isinstance(effective_graph_id, str):
            return {"error": "Group ID must be a string"}

        client = cast(Graphiti, graphiti_client)

        episodes = await client.retrieve_episodes(
            group_ids=[effective_graph_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
        )

        if not episodes:
            return {"message": f"No episodes found for group {effective_graph_id}", "episodes": []}

        # Use Pydantic's model_dump method for EpisodicNode serialization
        formatted_episodes = [
            # Use mode='json' to handle datetime serialization
            episode.model_dump(mode="json")
            for episode in episodes
        ]

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting episodes: {error_msg}")
        return {"error": f"Error getting episodes: {error_msg}"}


@mcp.tool()
async def clear_graph() -> SuccessResponse | ErrorResponse:
    """Clear all data from the Graphiti knowledge graph and rebuild indices."""
    global graphiti_client

    if graphiti_client is None:
        return {"error": "Graphiti client not initialized"}

    try:
        client = cast(Graphiti, graphiti_client)

        # clear_data is already imported at the top
        await clear_data(client.driver)
        await client.build_indices_and_constraints()
        return {"message": "Graph cleared successfully and indices rebuilt"}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error clearing graph: {error_msg}")
        return {"error": f"Error clearing graph: {error_msg}"}


@mcp.resource("http://graphiti/status")
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    global graphiti_client

    if graphiti_client is None:
        return {"status": "error", "message": "Graphiti client not initialized"}

    try:
        client = cast(Graphiti, graphiti_client)

        # Test Neo4j connection
        await client.driver.verify_connectivity()
        return {"status": "ok", "message": "Graphiti MCP server is running and connected to Neo4j"}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error checking Neo4j connection: {error_msg}")
        return {
            "status": "error",
            "message": f"Graphiti MCP server is running but Neo4j connection failed: {error_msg}",
        }


def create_llm_client(api_key: str | None = None, model: str | None = None) -> LLMClient:
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


async def initialize_server() -> tuple[MCPConfig, FastMCP]:
    """Parse args and create both config and MCP instance."""
    parser = argparse.ArgumentParser(
        description="Run the Graphiti MCP server with optional LLM client"
    )
    parser.add_argument(
        "--graph-name",
        help="Namespace for the graph. This is an arbitrary string used to organize related data. "
        "If not provided, a random UUID will be generated.",
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default="sse",
        help="Transport to use for communication with the client. Default: sse",
    )
    # OpenAI is the only supported LLM client
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name to use with the LLM client. Default: gpt-4o-mini",
    )
    parser.add_argument("--destroy-graph", action="store_true", help="Destroy all Graphiti graphs")
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

    args: argparse.Namespace = parser.parse_args()

    # Set the graph_id from CLI argument or generate a random one
    if args.graph_id:
        config.graph_id = args.graph_id
        logger.info(f"Using provided graph_id: {config.graph_id}")
    else:
        config.graph_id = f"graph_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated random graph_id: {config.graph_id}")

    # Set use_custom_entities flag if specified
    if args.use_custom_entities:
        config.use_custom_entities = True
        logger.info("Entity extraction enabled using predefined MEMCP_ENTITIES")
    else:
        logger.info("Entity extraction disabled (no custom entities will be used)")

    llm_client = None

    # Create OpenAI client if model is specified or if OPENAI_API_KEY is available
    if args.model or config.openai_api_key:
        # Override model from command line if specified
        config.model_name = args.model

        # Create the OpenAI client
        llm_client = create_llm_client(api_key=config.openai_api_key, model=config.model_name)

    # Initialize Graphiti with the specified LLM client
    await initialize_graphiti(llm_client, destroy_graph=args.destroy_graph)

    # Create the config
    mcp_config = MCPConfig(transport=args.transport, host=args.host, port=args.port)

    # Create the MCP instance with proper settings from config
    mcp_instance = FastMCP(
        "graphiti",
        instructions=GRAPHITI_MCP_INSTRUCTIONS,
        settings={"host": mcp_config.host, "port": mcp_config.port},
    )

    return mcp_config, mcp_instance


# Add a shutdown flag to track shutdown state
_shutdown_in_progress = False


async def graceful_shutdown(
    queue_manager: QueueManager, queue_progress_display: QueueProgressDisplay, timeout: float = 5.0
) -> None:
    """Perform graceful shutdown of the MCP server and all resources.

    Args:
        queue_manager: The queue manager to shut down
        queue_progress_display: The progress display to use for the shutdown visualization
        timeout: Maximum time to wait for tasks to complete gracefully
    """
    global graphiti_client, _shutdown_in_progress, _shutdown_mode
    # Prevent multiple shutdown attempts
    if _shutdown_in_progress:
        return
    _shutdown_in_progress = True

    # Set the shutdown mode for our message
    _shutdown_mode = "GRACEFUL"

    # Stop the queue progress display (cleanup resources)
    if queue_progress_display:
        queue_progress_display.stop()

    # Suppress standard error output during shutdown to avoid showing cancellation errors
    sys.stderr = open(os.devnull, "w")

    # Clear the screen to prepare for clean shutdown display
    shutdown_console.clear()

    # Create a table for tracking shutdown progress with custom styling
    table = Table(
        box=box.ROUNDED,
        border_style="shutdown",
        expand=False,
        show_header=True,
        highlight=True,
        title="[success]Graphiti MCP Server Shutdown[/success]",
    )
    table.add_column("[shutdown]Step[/shutdown]", style="info")
    table.add_column("[shutdown]Status[/shutdown]", style="info")

    # Create a status for shutdown process
    with shutdown_console.status(
        "[shutdown]Performing graceful shutdown...[/shutdown]", spinner="dots"
    ) as status:
        # Add initial table
        table.add_row("Shutdown initiated", "[step.success]✓[/step.success]")
        status.update(Panel(table, border_style="shutdown"))
        await asyncio.sleep(0.1)  # Small pause to ensure UI updates

        # 1. Cancel queue workers
        active_worker_count = queue_manager.cancel_all_workers()

        table.add_row(f"Queue workers ({active_worker_count})", "[step.success]✓[/step.success]")
        status.update(Panel(table, border_style="shutdown"))
        await asyncio.sleep(0.1)  # Small pause to ensure UI updates

        # 2. Close Neo4j connection if client exists
        neo4j_success = True
        if graphiti_client is not None:
            try:
                await graphiti_client.driver.close()
                logger.info("[success]Neo4j connection closed successfully[/success]")
            except Exception:
                neo4j_success = False
                logger.warning("[warning]Could not close Neo4j connection cleanly[/warning]")

        table.add_row(
            "Neo4j connection",
            "[step.success]✓[/step.success]" if neo4j_success else "[step.warning]⚠[/step.warning]",
        )
        status.update(Panel(table, border_style="shutdown"))
        await asyncio.sleep(0.1)  # Small pause to ensure UI updates

        # 3. Cancel all remaining tasks except the current one
        current_task = asyncio.current_task()
        remaining_tasks = []
        for task in [t for t in asyncio.all_tasks() if t is not current_task]:
            if task.get_name() != "shutdown_task":
                task.cancel()
                remaining_tasks.append(task)
                # Add a colorful log entry indicating this cancellation is expected
                logger.info(
                    f"[task]Cancelling task {task.get_name()} - [success]Expected![/success][/task]"
                )

        table.add_row(f"Tasks cancelled ({len(remaining_tasks)})", "[step.success]✓[/step.success]")
        status.update(Panel(table, border_style="shutdown"))
        await asyncio.sleep(0.1)  # Small pause to ensure UI updates

        # 4. Wait for all tasks to complete with timeout
        completion_success = True
        if remaining_tasks:
            try:
                # Suppress cancellation errors during shutdown
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(
                        asyncio.gather(*remaining_tasks, return_exceptions=True), timeout
                    )
                    logger.info("[success]All tasks completed gracefully[/success]")
            except asyncio.TimeoutError:
                completion_success = False
                logger.warning(f"[warning]Shutdown timed out after {timeout}s[/warning]")

        table.add_row(
            "Task completion",
            "[step.success]✓[/step.success]"
            if completion_success
            else "[step.warning]⚠[/step.warning]",
        )
        status.update(Panel(table, border_style="shutdown"))
        await asyncio.sleep(0.1)  # Small pause to ensure UI updates

        # Final status row
        table.add_row("Server shutdown", "[step.success]✓[/step.success]")
        status.update(Panel(table, border_style="shutdown"))

    # Final message - outside the status context
    shutdown_console.print(
        Panel(
            "[success]Graphiti MCP Server shutdown complete[/success]\n"
            "[normal]All resources have been released and tasks cancelled properly.[/normal]",
            title="[success]Shutdown Complete[/success]",
            border_style="success",
            padding=(1, 2),
        )
    )

    # Reset stderr before exit
    sys.stderr = sys.__stderr__

    # Create a final message file that main() will detect after shutdown
    with open(".shutdown_message.txt", "w") as f:
        f.write("GRACEFUL_SHUTDOWN_COMPLETE")

    # Exit the process immediately
    sys.exit(0)


def force_kill(queue_progress_display: QueueProgressDisplay) -> None:
    """Force kill the process immediately without cleanup.

    Args:
        queue_progress_display: The progress display to use for the episode
    """
    global _shutdown_mode

    # Set the shutdown mode for our message
    _shutdown_mode = "FORCE"

    # Stop the queue progress display
    if queue_progress_display:
        queue_progress_display.stop()

    # Suppress standard error output during shutdown to avoid showing errors
    sys.stderr = open(os.devnull, "w")

    # Clear the console for a clean display
    shutdown_console.clear()

    shutdown_console.print(
        Panel(
            "[danger]Force killing Graphiti MCP Server![/danger]\n"
            "[normal]No cleanup was performed. Some resources may not be properly released.[/normal]",
            title="[danger]Emergency Shutdown[/danger]",
            border_style="danger",
            padding=(1, 2),
        )
    )

    # Reset stderr before exit
    sys.stderr = sys.__stderr__

    # Brief pause to ensure message is displayed
    shutdown_console.print()

    # Use sys.exit instead of os._exit to allow atexit handlers to run
    sys.exit(1)


# Register the SIGTERM handler - move this after main is defined
def setup_signal_handlers(
    queue_manager: QueueManager, queue_progress_display: QueueProgressDisplay
) -> None:
    """Set up signal handlers for graceful shutdown and force kill."""
    loop = asyncio.get_running_loop()

    # Register SIGHUP (1) for graceful shutdown
    loop.add_signal_handler(
        signal.SIGHUP,
        lambda: asyncio.create_task(
            graceful_shutdown(queue_manager, queue_progress_display), name="shutdown_task"
        ),
    )

    # Register SIGINT (2/Ctrl+C) for graceful shutdown as well
    loop.add_signal_handler(
        signal.SIGINT,
        lambda: asyncio.create_task(
            graceful_shutdown(queue_manager, queue_progress_display), name="shutdown_task"
        ),
    )

    # Register SIGQUIT (3) for force kill
    loop.add_signal_handler(signal.SIGQUIT, lambda: force_kill(queue_progress_display))

    # SIGTERM is handled separately at the process level with signal.signal()

    logger.info("[shutdown]Signal handlers registered:[/shutdown]")
    logger.info("  - [info]SIGHUP (1)[/info]: [success]Graceful shutdown[/success] (kill -1 <pid>)")
    logger.info("  - [info]SIGINT (2)[/info]: [success]Graceful shutdown[/success] (Ctrl+C)")
    logger.info("  - [info]SIGQUIT (3)[/info]: [danger]Force kill[/danger] (emergency only)")
    logger.info("  - [info]SIGTERM (15)[/info]: [success]Graceful shutdown[/success] (docker/k8s)")


async def _refresh_display_periodically(live, get_display_fn, interval):
    """Periodically refresh the live display with fresh content.

    Args:
        live: The Live display instance
        get_display_fn: Function that returns a fresh renderable
        interval: Time between updates in seconds
    """
    try:
        while True:
            # Update the live display with fresh content
            live.update(get_display_fn())
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        # Exit cleanly when cancelled
        pass


async def run_mcp_server() -> None:
    """Run the MCP server in the current event loop."""
    global queue_manager

    # Get both config and server instance
    mcp_config, mcp = await initialize_server()

    # Create global instances for queue tracking and display
    console = Console(theme=GRAPHITI_THEME)
    queue_stats_tracker = QueueStatsTracker()
    queue_manager = QueueManager(queue_stats_tracker)
    queue_progress_display = QueueProgressDisplay(console, queue_stats_tracker)

    # Add the display as a state change callback for the queue manager
    queue_manager.add_state_change_callback(queue_progress_display.update)

    # Set up signal handlers for shutdown
    setup_signal_handlers(queue_manager, queue_progress_display)

    # Initialize the queue progress display
    queue_progress_display.create_progress_tasks()  # Initialize the tasks

    # Display the current process ID for easier signal sending
    pid = os.getpid()

    # Create a nice server info panel
    server_table = Table(show_header=False, box=box.SIMPLE, expand=False)
    server_table.add_column("Property", style="info")
    server_table.add_column("Value", style="success")
    server_table.add_row("Status", "Running")
    server_table.add_row("PID", str(pid))
    server_table.add_row("Transport", mcp_config.transport)
    server_table.add_row("Graph ID", config.graph_id or "None")

    if mcp_config.transport == "sse":
        server_table.add_row("Address", f"{mcp_config.host}:{mcp_config.port}")

    console.print(
        Panel(
            server_table, title="[highlight]Graphiti MCP Server[/highlight]", border_style="success"
        )
    )

    # Show commands in a separate panel
    command_table = Table(show_header=False, box=box.SIMPLE, expand=False)
    command_table.add_column("Command", style="highlight")
    command_table.add_column("Description", style="normal")
    command_table.add_row("Ctrl+C", "[success]Graceful shutdown[/success]")
    command_table.add_row(f"kill -1 {pid}", "[success]Graceful shutdown[/success]")
    command_table.add_row(f"kill -3 {pid}", "[danger]Force kill[/danger] (emergency only)")

    console.print(
        Panel(
            command_table,
            title="[normal]Control Commands[/normal]",
            border_style="normal",
            padding=(1, 2),
        )
    )

    def create_status_obj() -> Status:
        # Create and start the Status object once, before the Live display
        if mcp_config.transport == "stdio":
            status_obj = Status(
                Text(" STDIO Server is running...", style="normal"),
                spinner="arc",
                spinner_style="success",
            )
        else:
            status_obj = Status(
                Text(f" SSE Server running on {mcp_config.host}:{mcp_config.port}"),
                spinner="arc",
                spinner_style="green",
            )
        return status_obj

    def get_current_display(status_obj: Status) -> Group:
        # Handle potential None case
        progress_panel: Panel = queue_progress_display.get_renderable()
        return Group(progress_panel, status_obj)

    # Run the server with appropriate transport and dynamic display
    try:
        status_obj: Status = create_status_obj()
        # Get the initial display renderable
        initial_display = get_current_display(status_obj)

        if mcp_config.transport == "stdio":
            with Live(
                initial_display,
                console=console,
                refresh_per_second=4,
                transient=False,
            ) as live:
                # Create a refresh task that updates the display periodically
                refresh_task = asyncio.create_task(
                    _refresh_display_periodically(
                        live, lambda: get_current_display(status_obj), 0.25
                    )  # 4 times per second
                )
                try:
                    await mcp.run_stdio_async()
                finally:
                    # Cancel the refresh task when the server stops
                    refresh_task.cancel()
        elif mcp_config.transport == "sse":
            with Live(
                initial_display,
                console=console,
                refresh_per_second=4,
                transient=False,
            ) as live:
                # Create a refresh task that updates the display periodically
                refresh_task = asyncio.create_task(
                    _refresh_display_periodically(
                        live, lambda: get_current_display(status_obj), 0.25
                    )  # 4 times per second
                )
                try:
                    await mcp.run_sse_async()
                finally:
                    # Cancel the refresh task when the server stops
                    refresh_task.cancel()
    except asyncio.CancelledError:
        # Mark this as an expected event with color coding
        logger.info(
            "[success]Server shutdown initiated - tasks being cancelled (this is normal)[/success]"
        )
        pass


# Add a trap for SIGTERM to handle docker/kubernetes style shutdowns
def sigterm_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM (15) by starting graceful shutdown."""
    global _shutdown_mode
    _shutdown_mode = "GRACEFUL"
    print("\nSIGTERM received. Initiating graceful shutdown...")
    sys.exit(0)  # This will trigger our atexit handler


def main() -> None:
    """Main function to run the Graphiti MCP server."""
    # Register the SIGTERM handler at the process level
    signal.signal(signal.SIGTERM, sigterm_handler)

    # Add a try-finally to ensure our message is shown
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except Exception as e:
        # Ensure we show an error but also our final message
        logger.error(f"Error initializing Graphiti MCP server: {str(e)}")
        _shutdown_mode = "GRACEFUL"  # Use graceful message even for errors
        # Let the exception propagate so the traceback is shown
        raise
    finally:
        # Ensure we reset stderr
        sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
