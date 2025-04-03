#!/usr/bin/env python3
"""Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)."""

# Import our queue tracking and display classes
from queue_stats import QueueStatsTracker

import asyncio
import atexit
import logging
import os
import sys
import time
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
from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback

# Custom theme for our console
GRAPHITI_THEME = Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "danger": "bold red",
        "success": "bold green",
        "shutdown": "bold blue",
        "task": "dim magenta",
        "highlight": "bold cyan",
        "normal": "white",
        "step.success": "green",
        "step.warning": "yellow",
    }
)

# Install rich traceback handler with custom filters to hide CancelledError
install_rich_traceback(
    show_locals=False,
    suppress=["asyncio.exceptions.CancelledError"],
)

load_dotenv()

# Initialize Rich consoles with our custom theme
console = Console(theme=GRAPHITI_THEME)
shutdown_console = Console(stderr=True, theme=GRAPHITI_THEME)
progress_console = Console(theme=GRAPHITI_THEME)  # New console for progress display

# Create global instances for queue tracking and display
queue_stats_tracker = QueueStatsTracker()
queue_progress_display = None  # Will be initialized in the run_mcp_server function

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
    host: str = Field(
        default="127.0.0.1", description="Host address for the server when transport is 'sse'"
    )
    port: int = Field(
        default=8000, description="Port number for the server when transport is 'sse'"
    )


# Configure logging with Rich to minimize noise and add colors
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
            highlighter=ReprHighlighter(),
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


# Initialize Graphiti client
graphiti_client: Graphiti | None = None


class MCPFactory:
    """Factory for creating MCP instances.

    This class provides a factory method to create MCP instances based on the configuration.
    It supports both SSE and stdio transports.

    Attributes:
        name: Name of the MCP instance
        instructions: Instructions for the MCP
        config: Configuration for the MCP

    Examples:
        ```Python
        sse_mcp_config = MCPConfig(transport="sse", host="127.0.0.1", port=8000)

        sse_mcp_instance = MCPFactory.create_mcp(
            "graphiti",
            instructions=GRAPHITI_MCP_INSTRUCTIONS,
            config=sse_mcp_config
        )

        stdio_mcp_config = MCPConfig(transport="stdio")

        stdio_mcp_instance = MCPFactory.create_mcp(
            "graphiti",
            instructions=GRAPHITI_MCP_INSTRUCTIONS,
            config=stdio_mcp_config
        )
        ```
    """

    @staticmethod
    def create_mcp(name: str, instructions: str, config: MCPConfig) -> FastMCP:
        """Create an MCP instance with the given configuration.

        Args:
            name: Name of the MCP instance
            instructions: Instructions for the MCP
            config: Configuration for the MCP

        Returns:
            An initialized FastMCP instance
        """

        def create_sse_mcp() -> FastMCP:
            return FastMCP(
                name,
                instructions=instructions,
                settings={"host": config.host, "port": config.port},
            )

        def create_stdio_mcp() -> FastMCP:
            return FastMCP(
                name,
                instructions=instructions,
            )

        return create_sse_mcp() if config.transport == "sse" else create_stdio_mcp()


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
    global queue_workers, queue_stats_tracker, queue_progress_display

    # Name the task to be able to identify it during shutdown
    current_task = asyncio.current_task()
    if current_task:
        current_task.set_name(f"queue_worker_{group_id}")

    logger.info(f"Starting episode queue worker for group_id: [highlight]{group_id}[/highlight]")
    queue_workers[group_id] = True

    try:
        while True:
            # Get the next episode processing function from the queue
            # This will wait if the queue is empty
            process_func = await episode_queues[group_id].get()

            # Generate a unique task ID for tracking
            task_id = str(uuid.uuid4())

            try:
                # Record that processing has started
                queue_stats_tracker.start_processing(group_id, task_id)

                # Update the progress display
                if queue_progress_display:
                    queue_progress_display.update()

                # Process the episode
                await process_func()

                # Record successful completion
                queue_stats_tracker.complete_task(group_id, task_id, success=True)
            except Exception as e:
                # Record failed completion
                queue_stats_tracker.complete_task(group_id, task_id, success=False)

                logger.error(
                    f"Error processing queued episode for group_id [highlight]{group_id}"
                    f"[/highlight]: [danger]{str(e)}[/danger]"
                )
            finally:
                # Mark the task as done regardless of success/failure
                episode_queues[group_id].task_done()

                # Update the progress display
                if queue_progress_display:
                    queue_progress_display.update()
    except asyncio.CancelledError:
        logger.info(
            f"Episode queue worker for group_id [highlight]{group_id}[/highlight] was "
            "[success]cancelled[/success]"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in queue worker for group_id [highlight]{group_id}[/highlight]: "
            f"[danger]{str(e)}[/danger]"
        )
    finally:
        queue_workers[group_id] = False
        logger.info(
            f"[success]Stopped[/success] episode queue worker for group_id: [highlight]{group_id}"
            "[/highlight]"
        )


class GraphitiToolRegistry:
    """Registry for all Graphiti MCP tools.

    This class acts as a coordinator between the MCP instance and the various
    service classes that implement the actual functionality.
    """

    def __init__(
        self,
        mcp_instance: FastMCP,
        graphiti_client: Graphiti,
        config: GraphitiConfig,
    ) -> None:
        """Initialize the tool registry with an MCP instance.

        Args:
            mcp_instance: The MCP instance to register tools with
            graphiti_client: The Graphiti client to use for the tools
            config: The configuration for the Graphiti client
        """
        self.mcp = mcp_instance
        self.graphiti_client = graphiti_client
        self.config = config

    def _register_tools(self) -> None:
        """Register all tools with the MCP instance."""
        # Episode tools
        self.mcp.tool()(self.add_episode)
        self.mcp.tool()(self.delete_episode)
        self.mcp.tool()(self.get_episodes)

        # Search tools
        self.mcp.tool()(self.search_nodes)
        self.mcp.tool()(self.search_facts)

        # Entity tools
        self.mcp.tool()(self.delete_entity_edge)
        self.mcp.tool()(self.get_entity_edge)

        # Graph management tools
        self.mcp.tool()(self.clear_graph)

        # Status resource
        self.mcp.resource("http://graphiti/status")(self.get_status)

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        graph_id: str | None = None,
        source: str = "text",
        source_description: str = "",
        uuid: str | None = None,
    ) -> SuccessResponse | ErrorResponse:
        r"""Add an episode to the Graphiti knowledge graph. This is the primary way to add
        information to the graph.

        This function returns immediately and processes the episode addition in the background.
        Episodes for the same group_id are processed sequentially to avoid race conditions.

        Args:
            name (str): Name of the episode
            episode_body (str): The content of the episode. When source='json', this must be a
            properly escaped JSON string,
                            not a raw Python dictionary. The JSON data will be automatically
                            processed to extract entities and relationships.
            graph_id (str, optional): A unique ID for this graph. If not provided, uses the default
                                    graph_id from CLI or a generated one.
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
                episode_body="{\\\"company\\\": {\\\"name\\\": \\\"Acme Technologies\\\"}, "
                "\\\"products\\\": [{\\\"id\\\": \\\"P001\\\", \\\"name\\\": \\\"CloudSync\\\"}, "
                "{\\\"id\\\": \\\"P002\\\", \\\"name\\\": \\\"DataMiner\\\"}]}",
                source="json",
                source_description="CRM data"
            )

            # Adding message-style content
            add_episode(
                name="Customer Conversation",
                episode_body="user: What's your return policy?\nassistant: You can return items "
                "within 30 days.",
                source="message",
                source_description="chat transcript",
                graph_id="some_arbitrary_string",
            )

        Notes:
            When using source='json':
            - The JSON must be a properly escaped string, not a raw Python dictionary
            - The JSON will be automatically processed to extract entities and relationships
            - Complex nested structures are supported (arrays, nested objects, mixed data types),
                but keep nesting to a minimum
            - Entities will be created from appropriate JSON properties
            - Relationships between entities will be established based on the JSON structure
        """
        global episode_queues, queue_workers, queue_stats_tracker, queue_progress_display

        if self.graphiti_client is None:
            return {"error": "Graphiti client not initialized"}

        try:
            # Map string source to EpisodeType enum
            source_type = EpisodeType.text
            if source.lower() == "message":
                source_type = EpisodeType.message
            elif source.lower() == "json":
                source_type = EpisodeType.json

            # Use the provided graph_id or fall back to the default from config
            effective_graph_id = graph_id if graph_id is not None else self.config.graph_id

            # Cast graph_id to str to satisfy type checker
            # The Graphiti client expects a str for graph_id, not Optional[str]
            graph_id_str = str(effective_graph_id) if effective_graph_id is not None else ""

            client = cast(Graphiti, self.graphiti_client)

            # Define the episode processing function
            async def process_episode() -> None:
                try:
                    logger.info(
                        f"Processing queued episode '[highlight]{name}[/highlight]' for graph_id: "
                        f"[highlight]{graph_id_str}[/highlight]"
                    )
                    # Use all entity types if use_custom_entities is enabled, otherwise use empty
                    # dict
                    entity_types = MEMCP_ENTITIES if self.config.use_custom_entities else {}

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
                        f"Episode '[highlight]{name}[/highlight]' [success]added successfully"
                        "[/success]"
                    )

                    logger.info(
                        f"Building communities after episode '[highlight]{name}[/highlight]'"
                    )
                    await client.build_communities()

                    logger.info(
                        f"Episode '[highlight]{name}[/highlight]' [success]processed successfully"
                        "[/success]"
                    )
                except Exception as e:
                    error_msg = str(e)
                    logger.error(
                        f"[danger]Error[/danger] processing episode '[highlight]{name}[/highlight]'"
                        f" for graph_id [highlight]{graph_id_str}[/highlight]: [danger]{error_msg}"
                        "[/danger]"
                    )

            # Initialize queue for this graph_id if it doesn't exist
            if graph_id_str not in episode_queues:
                episode_queues[graph_id_str] = asyncio.Queue()

            # Track the new task in our stats tracker
            queue_stats_tracker.add_task(graph_id_str)

            # Update the progress display immediately
            if queue_progress_display:
                queue_progress_display.update()

            # Add the episode processing function to the queue
            await episode_queues[graph_id_str].put(process_episode)

            # Start a worker for this queue if one isn't already running
            if not queue_workers.get(graph_id_str, False):
                asyncio.create_task(process_episode_queue(graph_id_str))

            # Return immediately with a success message
            return {
                "message": f"Episode '[highlight]{name}[/highlight]' queued for processing "
                f"(position: [highlight]{episode_queues[graph_id_str].qsize()}[/highlight])"
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[danger]Error queuing episode task[/danger]: {error_msg}")
            return {"error": f"Error queuing episode task: {error_msg}"}

    async def search_nodes(
        self,
        query: str,
        graph_ids: list[str] | None = None,
        max_nodes: int = 10,
        center_node_uuid: str | None = None,
        entity: str = "",  # cursor seems to break with None
    ) -> NodeSearchResponse | ErrorResponse:
        """Search the Graphiti knowledge graph for relevant node summaries.

        These contain a summary of all of a node's relationships with other nodes.

        Note: entity is a single entity type to filter results (permitted: "Preference",
        "Procedure").

        Args:
            query: The search query
            graph_ids: Optional list of group IDs to filter results
            max_nodes: Maximum number of nodes to return (default: 10)
            center_node_uuid: Optional UUID of a node to center the search around
            entity: Optional single entity type to filter results (permitted: "Preference",
            "Procedure")
        """
        if self.graphiti_client is None:
            return ErrorResponse(error="Graphiti client not initialized")

        try:
            # Use the provided graph_ids or fall back to the default from config if none provided
            effective_graph_ids = (
                graph_ids
                if graph_ids is not None
                else [self.config.graph_id]
                if self.config.graph_id
                else []
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

            client = cast(Graphiti, self.graphiti_client)

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

    async def search_facts(
        self,
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
        if self.graphiti_client is None:
            return {"error": "Graphiti client not initialized"}

        try:
            # Use the provided graph_ids or fall back to the default from config if none provided
            effective_graph_ids = (
                graph_ids
                if graph_ids is not None
                else [self.config.graph_id]
                if self.config.graph_id
                else []
            )

            client = cast(Graphiti, self.graphiti_client)

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

    async def delete_entity_edge(self, uuid: str) -> SuccessResponse | ErrorResponse:
        """Delete an entity edge from the Graphiti knowledge graph.

        Args:
            uuid: UUID of the entity edge to delete
        """
        if self.graphiti_client is None:
            return {"error": "Graphiti client not initialized"}

        try:
            client = cast(Graphiti, self.graphiti_client)

            # Get the entity edge by UUID
            entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
            # Delete the edge using its delete method
            await entity_edge.delete(client.driver)
            return {"message": f"Entity edge with UUID {uuid} deleted successfully"}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error deleting entity edge: {error_msg}")
            return {"error": f"Error deleting entity edge: {error_msg}"}

    async def delete_episode(self, uuid: str) -> SuccessResponse | ErrorResponse:
        """Delete an episode from the Graphiti knowledge graph.

        Args:
            uuid: UUID of the episode to delete
        """
        if self.graphiti_client is None:
            return {"error": "Graphiti client not initialized"}

        try:
            client = cast(Graphiti, self.graphiti_client)

            # Get the episodic node by UUID - EpisodicNode is already imported at the top
            episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
            # Delete the node using its delete method
            await episodic_node.delete(client.driver)
            return {"message": f"Episode with UUID {uuid} deleted successfully"}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error deleting episode: {error_msg}")
            return {"error": f"Error deleting episode: {error_msg}"}

    async def get_entity_edge(self, uuid: str) -> dict[str, Any] | ErrorResponse:
        """Get an entity edge from the Graphiti knowledge graph by its UUID.

        Args:
            uuid: UUID of the entity edge to retrieve
        """
        if self.graphiti_client is None:
            return {"error": "Graphiti client not initialized"}

        try:
            client = cast(Graphiti, self.graphiti_client)

            # Get the entity edge directly using the EntityEdge class method
            entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

            # Use the format_fact_result function to serialize the edge
            # Return the Python dict directly - MCP will handle serialization
            return format_fact_result(entity_edge)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting entity edge: {error_msg}")
            return {"error": f"Error getting entity edge: {error_msg}"}

    async def get_episodes(
        self, graph_id: str | None = None, last_n: int = 10
    ) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
        """Get the most recent episodes for a specific group.

        Args:
            graph_id: ID of the group to retrieve episodes from. If not provided, uses the default
                graph_id.
            last_n: Number of most recent episodes to retrieve (default: 10)
        """
        if self.graphiti_client is None:
            return {"error": "Graphiti client not initialized"}

        try:
            # Use the provided graph_id or fall back to the default from config
            effective_graph_id = graph_id if graph_id is not None else self.config.graph_id

            if not isinstance(effective_graph_id, str):
                return {"error": "Group ID must be a string"}

            client = cast(Graphiti, self.graphiti_client)

            episodes = await client.retrieve_episodes(
                group_ids=[effective_graph_id],
                last_n=last_n,
                reference_time=datetime.now(timezone.utc),
            )

            if not episodes:
                return {
                    "message": f"No episodes found for group {effective_graph_id}",
                    "episodes": [],
                }

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

    async def clear_graph(self) -> SuccessResponse | ErrorResponse:
        """Clear all data from the Graphiti knowledge graph and rebuild indices."""
        if self.graphiti_client is None:
            return {"error": "Graphiti client not initialized"}

        try:
            client = cast(Graphiti, self.graphiti_client)

            # clear_data is already imported at the top
            await clear_data(client.driver)
            await client.build_indices_and_constraints()
            return {"message": "Graph cleared successfully and indices rebuilt"}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error clearing graph: {error_msg}")
            return {"error": f"Error clearing graph: {error_msg}"}

    async def get_status(self) -> StatusResponse:
        """Get the status of the Graphiti MCP server and Neo4j connection."""
        if self.graphiti_client is None:
            return {"status": "error", "message": "Graphiti client not initialized"}

        try:
            client = cast(Graphiti, self.graphiti_client)

            # Test Neo4j connection
            await client.driver.verify_connectivity()
            return {
                "status": "ok",
                "message": "Graphiti MCP server is running and connected to Neo4j",
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error checking Neo4j connection: {error_msg}")
            return {
                "status": "error",
                "message": f"Graphiti MCP server is running but Neo4j connection failed: "
                f"{error_msg}",
            }
