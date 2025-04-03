#!/usr/bin/env python3
"""Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)."""

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

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
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

# Install rich traceback handler with custom filters to hide CancelledError
install_rich_traceback(show_locals=False, suppress=["asyncio.exceptions.CancelledError"])

load_dotenv()

# Initialize Rich consoles - one for main output and one for shutdown process
console = Console()
# Use a dedicated console for shutdown to avoid mixing with other output
shutdown_console = Console(stderr=True)


class Requirement(BaseModel):
    """A Requirement represents a specific need, feature, or functionality that a product or
    service must fulfill.

    Always ensure an edge is created between the requirement and the project it belongs to,
    and clearly indicate on the edge that the requirement is a requirement.

    Instructions for identifying and extracting requirements:
    1. Look for explicit statements of needs or necessities ("We need X", "X is required",
    "X must have Y")
    2. Identify functional specifications that describe what the system should do
    3. Pay attention to non-functional requirements like performance, security, or usability
    criteria
    4. Extract constraints or limitations that must be adhered to
    5. Focus on clear, specific, and measurable requirements rather than vague wishes
    6. Capture the priority or importance if mentioned ("critical", "high priority", etc.)
    7. Include any dependencies between requirements when explicitly stated
    8. Preserve the original intent and scope of the requirement
    9. Categorize requirements appropriately based on their domain or function
    """

    project_name: str = Field(
        ...,
        description="The name of the project to which the requirement belongs.",
    )
    description: str = Field(
        ...,
        description="Description of the requirement. Only use information mentioned "
        "in the context to write this description.",
    )


class Preference(BaseModel):
    """A Preference represents a user's expressed like, dislike, or preference for something.

    Instructions for identifying and extracting preferences:
    1. Look for explicit statements of preference such as "I like/love/enjoy/prefer X"
    or "I don't like/hate/dislike X"
    2. Pay attention to comparative statements ("I prefer X over Y")
    3. Consider the emotional tone when users mention certain topics
    4. Extract only preferences that are clearly expressed, not assumptions
    5. Categorize the preference appropriately based on its domain (food, music, brands, etc.)
    6. Include relevant qualifiers (e.g., "likes spicy food" rather than just "likes food")
    7. Only extract preferences directly stated by the user, not preferences of others they mention
    8. Provide a concise but specific description that captures the nature of the preference
    """

    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')",
    )
    description: str = Field(
        ...,
        description="Brief description of the preference. Only use information mentioned "
        "in the context to write this description.",
    )


class Procedure(BaseModel):
    """A Procedure informing the agent what actions to take or how to perform in certain scenarios.
    Procedures are typically composed of several steps.

    Instructions for identifying and extracting procedures:
    1. Look for sequential instructions or steps ("First do X, then do Y")
    2. Identify explicit directives or commands ("Always do X when Y happens")
    3. Pay attention to conditional statements ("If X occurs, then do Y")
    4. Extract procedures that have clear beginning and end points
    5. Focus on actionable instructions rather than general information
    6. Preserve the original sequence and dependencies between steps
    7. Include any specified conditions or triggers for the procedure
    8. Capture any stated purpose or goal of the procedure
    9. Summarize complex procedures while maintaining critical details
    """

    description: str = Field(
        ...,
        description="Brief description of the procedure. Only use information mentioned "
        "in the context to write this description.",
    )


class MemCPEntities:
    """Container for MemCP entity types, used to enforce type safety when adding entities.

    EntityTypes:
        - Requirement
        - Preference
        - Procedure
    """

    Requirement = Requirement
    Preference = Preference
    Procedure = Procedure

    @classmethod
    def as_dict(cls) -> dict[str, type[BaseModel]]:
        """Get entities as a dictionary.

        Returns:
            dict[str, type[BaseModel]]: A dictionary of entity types

        Examples:
            ```Python
            MemCPEntities.as_dict()
            {
                "Requirement": Requirement,
                "Preference": Preference,
                "Procedure": Procedure,
            }
            ```
        """
        return {
            "Requirement": Requirement,
            "Preference": Preference,
            "Procedure": Procedure,
        }


MEMCP_ENTITIES: dict[str, type[BaseModel]] = MemCPEntities.as_dict()


# Type definitions for API responses
class ErrorResponse(TypedDict):
    """Represents an error response from the MCP server."""

    error: str


class SuccessResponse(TypedDict):
    """Represents a successful response from the MCP server."""

    message: str


class NodeResult(TypedDict):
    """Represents a node result from the MCP server."""

    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    """Represents a node search response from the MCP server."""

    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    """Represents a fact search response from the MCP server."""

    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    """Represents an episode search response from the MCP server."""

    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    """Represents a status response from the MCP server."""

    status: str
    message: str


# Server configuration classes
class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client.

    Centralizes all configuration parameters for the Graphiti client,
    including database connection details and LLM settings.
    """

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    model_name: str | None = None
    graph_id: str | None = None
    use_custom_entities: bool = False

    @classmethod
    def from_env(cls) -> "GraphitiConfig":
        """Create a configuration instance from environment variables."""
        neo4j_password = os.environ.get("NEO4J_PASSWORD")
        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD must be set")

        return cls(
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=neo4j_password,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_base_url=os.environ.get("OPENAI_BASE_URL"),
            model_name=os.environ.get("MODEL_NAME"),
        )


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str


# Configure logging with Rich to minimize noise
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
            show_time=False,
            show_path=False,
            enable_link_path=False,  # Disable clickable file paths
        )
    ],
)

# Configure root logger to ignore asyncio cancellation errors during shutdown
logging.getLogger("asyncio").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Create global config instance
config: GraphitiConfig = GraphitiConfig.from_env()

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
)


# Initialize Graphiti client
graphiti_client: Graphiti | None = None


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


# Dictionary to store queues for each group_id
# Each queue is a list of tasks to be processed sequentially
episode_queues: dict[str, asyncio.Queue] = {}
# Dictionary to track if a worker is running for each group_id
queue_workers: dict[str, bool] = {}


async def process_episode_queue(group_id: str) -> None:
    """Process episodes for a specific group_id sequentially.

    This function runs as a long-lived task that processes episodes
    from the queue one at a time.
    """
    global queue_workers

    # Name the task to be able to identify it during shutdown
    current_task = asyncio.current_task()
    if current_task:
        current_task.set_name(f"queue_worker_{group_id}")

    logger.info(f"Starting episode queue worker for group_id: {group_id}")
    queue_workers[group_id] = True

    try:
        while True:
            # Get the next episode processing function from the queue
            # This will wait if the queue is empty
            process_func = await episode_queues[group_id].get()

            try:
                # Process the episode
                await process_func()
            except Exception as e:
                logger.error(f"Error processing queued episode for group_id {group_id}: {str(e)}")
            finally:
                # Mark the task as done regardless of success/failure
                episode_queues[group_id].task_done()
    except asyncio.CancelledError:
        logger.info(f"Episode queue worker for group_id {group_id} was cancelled")
    except Exception as e:
        logger.error(f"Unexpected error in queue worker for group_id {group_id}: {str(e)}")
    finally:
        queue_workers[group_id] = False
        logger.info(f"Stopped episode queue worker for group_id: {group_id}")


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
    global graphiti_client, episode_queues, queue_workers

    if graphiti_client is None:
        return {"error": "Graphiti client not initialized"}

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
                logger.info(f"Processing queued episode '{name}' for graph_id: {graph_id_str}")
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
                logger.info(f"Episode '{name}' added successfully")

                logger.info(f"Building communities after episode '{name}'")
                await client.build_communities()

                logger.info(f"Episode '{name}' processed successfully")
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error processing episode '{name}' for graph_id {graph_id_str}: {error_msg}"
                )

        # Initialize queue for this graph_id if it doesn't exist
        if graph_id_str not in episode_queues:
            episode_queues[graph_id_str] = asyncio.Queue()

        # Add the episode processing function to the queue
        await episode_queues[graph_id_str].put(process_episode)

        # Start a worker for this queue if one isn't already running
        if not queue_workers.get(graph_id_str, False):
            asyncio.create_task(process_episode_queue(graph_id_str))

        # Return immediately with a success message
        return {
            "message": f"Episode '{name}' queued for processing (position: "
            f"{episode_queues[graph_id_str].qsize()})"
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error queuing episode task: {error_msg}")
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


async def initialize_server() -> MCPConfig:
    """Initialize the Graphiti server with the specified LLM client."""
    global config

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

    args = parser.parse_args()

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

    return MCPConfig(transport=args.transport)


# Add a shutdown flag to track shutdown state
_shutdown_in_progress = False


async def graceful_shutdown(timeout: float = 5.0) -> None:
    """Perform graceful shutdown of the MCP server and all resources.

    Args:
        timeout: Maximum time to wait for tasks to complete gracefully
    """
    global graphiti_client, queue_workers, _shutdown_in_progress

    # Prevent multiple shutdown attempts
    if _shutdown_in_progress:
        return
    _shutdown_in_progress = True

    # Suppress standard error output during shutdown to avoid showing cancellation errors
    sys.stderr = open(os.devnull, "w")

    # Clear the screen to prepare for clean shutdown display
    shutdown_console.clear()

    # Create a table for tracking shutdown progress
    table = Table(box=box.ROUNDED, border_style="blue", expand=False, show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Status", style="yellow")

    # Create a status for shutdown process with transient=True for cleaner exit
    with shutdown_console.status(
        "[bold blue]Performing graceful shutdown...", spinner="dots"
    ) as status:
        # Add initial table
        table.add_row("Shutdown initiated", "✓")
        status.update(
            Panel(table, title="[bold]Graceful Shutdown Progress[/bold]", border_style="blue")
        )
        await asyncio.sleep(0.1)  # Small pause to ensure UI updates

        # 1. Cancel episode queue workers
        tasks = []
        for group_id in list(queue_workers.keys()):
            if queue_workers.get(group_id, False):
                for task in asyncio.all_tasks():
                    if task.get_name().startswith(f"queue_worker_{group_id}"):
                        tasks.append(task)
                        task.cancel()

        table.add_row(f"Queue workers ({len(tasks)})", "✓")
        status.update(
            Panel(table, title="[bold]Graceful Shutdown Progress[/bold]", border_style="blue")
        )
        await asyncio.sleep(0.1)  # Small pause to ensure UI updates

        # 2. Close Neo4j connection if client exists
        neo4j_success = True
        if graphiti_client is not None:
            try:
                await graphiti_client.driver.close()
            except Exception:
                neo4j_success = False

        table.add_row("Neo4j connection", "✓" if neo4j_success else "⚠")
        status.update(
            Panel(table, title="[bold]Graceful Shutdown Progress[/bold]", border_style="blue")
        )
        await asyncio.sleep(0.1)  # Small pause to ensure UI updates

        # 3. Cancel all remaining tasks except the current one
        current_task = asyncio.current_task()
        remaining_tasks = []
        for task in [t for t in asyncio.all_tasks() if t is not current_task]:
            if task.get_name() != "shutdown_task":
                task.cancel()
                remaining_tasks.append(task)

        all_tasks = tasks + remaining_tasks
        table.add_row(f"Tasks cancelled ({len(all_tasks)})", "✓")
        status.update(
            Panel(table, title="[bold]Graceful Shutdown Progress[/bold]", border_style="blue")
        )
        await asyncio.sleep(0.1)  # Small pause to ensure UI updates

        # 4. Wait for all tasks to complete with timeout
        completion_success = True
        if all_tasks:
            try:
                # Suppress cancellation errors during shutdown
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(
                        asyncio.gather(*all_tasks, return_exceptions=True), timeout
                    )
            except asyncio.TimeoutError:
                completion_success = False

        table.add_row("Wait for task completion", "✓" if completion_success else "⚠")
        status.update(
            Panel(table, title="[bold]Graceful Shutdown Progress[/bold]", border_style="blue")
        )

    # Final message - outside the status context
    shutdown_console.print(
        Panel.fit(
            "[bold green]Graphiti MCP Server shutdown complete[/bold green]",
            title="Shutdown",
            border_style="green",
        )
    )

    # Reset stderr before exit
    sys.stderr = sys.__stderr__

    # Use a cleaner exit approach that avoids traceback display
    os._exit(0)  # This is cleaner for our use case than sys.exit(0)


def force_kill() -> None:
    """Force kill the process immediately without cleanup."""
    # Suppress standard error output during shutdown to avoid showing errors
    sys.stderr = open(os.devnull, "w")

    # Clear the console for a clean display
    shutdown_console.clear()

    shutdown_console.print(
        Panel.fit(
            "[bold red]Force killing Graphiti MCP Server![/bold red]",
            title="Emergency Shutdown",
            border_style="red",
        )
    )

    # Brief pause to ensure message is displayed
    shutdown_console.print()
    os._exit(1)  # Immediate exit without any cleanup


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown and force kill."""
    loop = asyncio.get_running_loop()

    # Register SIGHUP (1) for graceful shutdown
    loop.add_signal_handler(
        signal.SIGHUP, lambda: asyncio.create_task(graceful_shutdown(), name="shutdown_task")
    )

    # Register SIGINT (2/Ctrl+C) for graceful shutdown as well
    loop.add_signal_handler(
        signal.SIGINT, lambda: asyncio.create_task(graceful_shutdown(), name="shutdown_task")
    )

    # Register SIGQUIT (3) for force kill
    loop.add_signal_handler(signal.SIGQUIT, force_kill)

    logger.info("Signal handlers registered:")
    logger.info("  - SIGHUP (1): Graceful shutdown (kill -1 <pid>)")
    logger.info("  - SIGINT (2): Graceful shutdown (Ctrl+C)")
    logger.info("  - SIGQUIT (3): Force kill (emergency only)")


async def run_mcp_server() -> None:
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Set up signal handlers for shutdown
    setup_signal_handlers()

    # Clear any previous output for clean display
    # console.clear()

    # Display the current process ID for easier signal sending
    pid = os.getpid()

    # Create a nice server info panel
    server_table = Table(show_header=False, box=box.SIMPLE, expand=False)
    server_table.add_column("Property", style="cyan")
    server_table.add_column("Value", style="green")
    server_table.add_row("Status", "Running")
    server_table.add_row("PID", str(pid))
    server_table.add_row("Transport", mcp_config.transport)
    server_table.add_row("Graph ID", config.graph_id or "None")

    if mcp_config.transport == "sse":
        server_table.add_row("Address", f"{mcp.settings.host}:{mcp.settings.port}")

    console.print(
        Panel(server_table, title="[bold]Graphiti MCP Server[/bold]", border_style="green")
    )

    # Show commands in a separate panel
    command_table = Table(show_header=False, box=box.SIMPLE, expand=False)
    command_table.add_column("Command", style="yellow")
    command_table.add_column("Description", style="white")
    command_table.add_row("Ctrl+C", "Graceful shutdown")
    command_table.add_row(f"kill -1 {pid}", "Graceful shutdown")
    command_table.add_row(f"kill -3 {pid}", "Force kill (emergency only)")

    console.print(
        Panel(
            command_table,
            title="[bold]Control Commands[/bold]",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    # Run the server with appropriate transport
    try:
        if mcp_config.transport == "stdio":
            with console.status("[bold green]Server running...", spinner="dots"):
                await mcp.run_stdio_async()
        elif mcp_config.transport == "sse":
            with console.status(
                "[bold green]Server running on " + f"{mcp.settings.host}:{mcp.settings.port}",
                spinner="dots",
            ):
                await mcp.run_sse_async()
    except asyncio.CancelledError:
        # This is expected during shutdown, suppress the error message
        pass


def main() -> None:
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except Exception as e:
        logger.error(f"Error initializing Graphiti MCP server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
