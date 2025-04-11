"""MCP tools for Graphiti Server."""

from memcp.api.api_errors import EpisodeError
from memcp.config.memcp_config import MemCPConfig
from memcp.models.entities import MEMCP_ENTITIES
from memcp.models.responses import (
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeResult,
    NodeSearchResponse,
    StatusResponse,
    SuccessResponse,
)
from memcp.queue import QueueError, QueueManager, QueueProcessingError
from memcp.utils import get_logger

from datetime import datetime, timezone
from typing import Any, cast

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Get a logger for this module
logger = get_logger(__name__)


def _format_fact_result(edge: EntityEdge) -> dict[str, Any]:
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


class MCPToolsRegistry:
    """Registry for MCP tools used in the Graphiti server.

    This class follows the dependency injection principle by accepting
    dependencies through the constructor rather than creating them internally.
    """

    def __init__(
        self,
        graphiti_client: Graphiti,
        queue_manager: QueueManager,
        config: MemCPConfig,
    ) -> None:
        """Initialize the MCP tools registry.

        Args:
            graphiti_client: Graphiti client instance
            queue_manager: Queue manager for processing tasks
            config: Configuration settings
        """
        self.graphiti_client: Graphiti = graphiti_client
        self.queue_manager: QueueManager = queue_manager
        self.config: MemCPConfig = config

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        graph_id: str | None = None,
        source: str = "text",
        source_description: str = "",
        uuid: str | None = None,
    ) -> SuccessResponse | ErrorResponse:
        r"""Add an episode to the Graphiti knowledge graph.

        This function returns immediately and processes the episode addition in the background.
        Episodes for the same group_id are processed sequentially to avoid race conditions.

        Args:
            name: Name of the episode
            episode_body: The content of the episode
            graph_id: Optional graph ID for this episode
            source: Source type (text, json, message)
            source_description: Description of the source
            uuid: Optional UUID for the episode

        Returns:
            SuccessResponse or ErrorResponse
        """
        try:
            # Map string source to EpisodeType enum
            source_type = EpisodeType.text
            if source.lower() == "message":
                source_type = EpisodeType.message
            elif source.lower() == "json":
                source_type = EpisodeType.json

            # Use the provided graph_id or fall back to the default from config
            effective_graph_id = graph_id if graph_id is not None else self.config.graph.id

            # Cast graph_id to str to satisfy type checker
            graph_id_str = str(effective_graph_id) if effective_graph_id is not None else ""

            # Define the episode processing function
            async def process_episode() -> None:
                try:
                    logger.info(
                        f"Processing queued episode '[highlight]{name}[/highlight]' for graph_id: [highlight]"
                        f"{graph_id_str}[/highlight]"
                    )
                    # Use all entity types if use_custom_entities is enabled, otherwise use empty dict
                    entity_types = MEMCP_ENTITIES if self.config.graph.use_memcp_entities else {}

                    # Cast to dict[str, BaseModel] to match the expected type in Graphiti
                    await self.graphiti_client.add_episode(
                        name=name,
                        episode_body=episode_body,
                        source=source_type,
                        source_description=source_description,
                        group_id=graph_id_str,
                        uuid=uuid,
                        reference_time=datetime.now(timezone.utc),
                        entity_types=cast("dict[str, BaseModel]", entity_types),
                    )
                    logger.info(f"Episode '[highlight]{name}[/highlight]' [success]added successfully[/success]")

                    logger.info(f"Building communities after episode '[highlight]{name}[/highlight]'")
                    await self.graphiti_client.build_communities()

                    logger.info(f"Episode '[highlight]{name}[/highlight]' [success]processed successfully[/success]")
                except Exception as e:
                    error_msg = str(e)
                    logger.error(
                        f"[danger]Error[/danger] processing episode '[highlight]{name}[/highlight]' for graph_id "
                        f"[highlight]{graph_id_str}[/highlight]: [danger]{error_msg}[/danger]"
                    )
                    raise QueueProcessingError(f"Error processing episode: {error_msg}") from e

            # Enqueue the task for processing
            try:
                await self.queue_manager.enqueue_task(graph_id_str, process_episode)
            except QueueError as e:
                return {"error": f"Error queuing episode task: {str(e)}"}

            # Return immediately with a success message
            # Use the queue size from the queue manager's queue for this group_id
            queue_size = 0
            if graph_id_str in self.queue_manager.episode_queues:
                queue_size = self.queue_manager.episode_queues[graph_id_str].qsize()

            return {
                "message": f"Episode '[highlight]{name}[/highlight]' queued for processing (position: "
                f"[highlight]{queue_size}[/highlight])"
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[danger]Error queuing episode task[/danger]: {error_msg}")
            if isinstance(e, QueueError):
                raise
            raise EpisodeError(f"Error queuing episode task: {error_msg}") from e

    async def search_nodes(
        self,
        query: str,
        graph_ids: list[str] | None = None,
        max_nodes: int = 10,
        center_node_uuid: str | None = None,
        entity: str = "",
    ) -> NodeSearchResponse | ErrorResponse:
        """Search the Graphiti knowledge graph for relevant node summaries.

        Args:
            query: The search query
            graph_ids: Optional list of group IDs to filter results
            max_nodes: Maximum number of nodes to return
            center_node_uuid: Optional UUID of a node to center search around
            entity: Optional entity type to filter results

        Returns:
            NodeSearchResponse or ErrorResponse
        """
        try:
            # Use the provided graph_ids or fall back to the default from config if none provided
            effective_graph_ids = (
                graph_ids if graph_ids is not None else [self.config.graph.id] if self.config.graph.id else []
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

            # the _search method is defined by the Graphiti library and is not private despite the leading underscore
            search_results = await self.graphiti_client._search(  # type: ignore
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
            max_facts: Maximum number of facts to return
            center_node_uuid: Optional UUID of a node to center search around

        Returns:
            FactSearchResponse or ErrorResponse
        """
        try:
            # Use the provided graph_ids or fall back to the default from config if none provided
            effective_graph_ids = (
                graph_ids if graph_ids is not None else [self.config.graph.id] if self.config.graph.id else []
            )

            relevant_edges = await self.graphiti_client.search(
                group_ids=effective_graph_ids,
                query=query,
                num_results=max_facts,
                center_node_uuid=center_node_uuid,
            )

            if not relevant_edges:
                return {"message": "No relevant facts found", "facts": []}

            facts = [_format_fact_result(edge) for edge in relevant_edges]
            return {"message": "Facts retrieved successfully", "facts": facts}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error searching facts: {error_msg}")
            return {"error": f"Error searching facts: {error_msg}"}

    async def delete_entity_edge(self, uuid: str) -> SuccessResponse | ErrorResponse:
        """Delete an entity edge from the Graphiti knowledge graph.

        Args:
            uuid: UUID of the entity edge to delete

        Returns:
            SuccessResponse or ErrorResponse
        """
        try:
            # Get the entity edge by UUID
            entity_edge = await EntityEdge.get_by_uuid(self.graphiti_client.driver, uuid)
            # Delete the edge using its delete method
            await entity_edge.delete(self.graphiti_client.driver)
            return {"message": f"Entity edge with UUID {uuid} deleted successfully"}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error deleting entity edge: {error_msg}")
            return {"error": f"Error deleting entity edge: {error_msg}"}

    async def delete_episode(self, uuid: str) -> SuccessResponse | ErrorResponse:
        """Delete an episode from the Graphiti knowledge graph.

        Args:
            uuid: UUID of the episode to delete

        Returns:
            SuccessResponse or ErrorResponse
        """
        try:
            # Get the episodic node by UUID
            episodic_node = await EpisodicNode.get_by_uuid(self.graphiti_client.driver, uuid)
            if not episodic_node:
                raise EpisodeError(f"Episode with UUID {uuid} not found")

            # Delete the node using its delete method
            await episodic_node.delete(self.graphiti_client.driver)
            return {"message": f"Episode with UUID {uuid} deleted successfully"}
        except EpisodeError as e:
            logger.error(f"Error deleting episode: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error deleting episode: {error_msg}")
            raise EpisodeError(f"Error deleting episode: {error_msg}") from e

    async def get_entity_edge(self, uuid: str) -> dict[str, Any] | ErrorResponse:
        """Get an entity edge from the Graphiti knowledge graph by its UUID.

        Args:
            uuid: UUID of the entity edge to retrieve

        Returns:
            Entity edge data or ErrorResponse
        """
        try:
            # Get the entity edge directly using the EntityEdge class method
            entity_edge = await EntityEdge.get_by_uuid(self.graphiti_client.driver, uuid)

            # Use the _format_fact_result function to serialize the edge
            return _format_fact_result(entity_edge)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting entity edge: {error_msg}")
            return {"error": f"Error getting entity edge: {error_msg}"}

    async def get_episodes(
        self, graph_id: str | None = None, last_n: int = 10
    ) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
        """Get the most recent episodes for a specific group.

        Args:
            graph_id: ID of the group to retrieve episodes from
            last_n: Number of most recent episodes to retrieve

        Returns:
            List of episodes or ErrorResponse
        """
        try:
            # Use the provided graph_id or fall back to the default from config
            effective_graph_id = graph_id if graph_id is not None else self.config.graph.id

            if not isinstance(effective_graph_id, str):
                return {"error": "Group ID must be a string"}

            episodes = await self.graphiti_client.retrieve_episodes(
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
            formatted_episodes = [episode.model_dump(mode="json") for episode in episodes]

            return formatted_episodes
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting episodes: {error_msg}")
            return {"error": f"Error getting episodes: {error_msg}"}

    async def clear_graph(self) -> SuccessResponse | ErrorResponse:
        """Clear all data from the Graphiti knowledge graph and rebuild indices.

        Returns:
            SuccessResponse or ErrorResponse
        """
        try:
            await clear_data(self.graphiti_client.driver)
            await self.graphiti_client.build_indices_and_constraints()
            return {"message": "Graph cleared successfully and indices rebuilt"}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error clearing graph: {error_msg}")
            return {"error": f"Error clearing graph: {error_msg}"}

    async def get_status(self) -> StatusResponse:
        """Get the status of the Graphiti MCP server and Neo4j connection.

        Returns:
            StatusResponse with status information
        """
        try:
            # Test Neo4j connection
            await self.graphiti_client.driver.verify_connectivity()
            return {
                "status": "ok",
                "message": "Graphiti MCP server is running and connected to Neo4j",
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error checking Neo4j connection: {error_msg}")
            return {
                "status": "error",
                "message": f"Graphiti MCP server is running but Neo4j connection failed: {error_msg}",
            }


def register_tools(mcp_server: FastMCP, tools_registry: MCPToolsRegistry) -> None:
    """Register all MCP tools with the MCP server.

    Args:
        mcp_server: MCP server instance
        tools_registry: Tools registry with implementation methods
    """
    # Register all tool methods
    mcp_server.tool()(tools_registry.add_episode)
    mcp_server.tool()(tools_registry.search_nodes)
    mcp_server.tool()(tools_registry.search_facts)
    mcp_server.tool()(tools_registry.delete_entity_edge)
    mcp_server.tool()(tools_registry.delete_episode)
    mcp_server.tool()(tools_registry.get_entity_edge)
    mcp_server.tool()(tools_registry.get_episodes)
    mcp_server.tool()(tools_registry.clear_graph)

    # Register the status resource
    mcp_server.resource("http://graphiti/status")(tools_registry.get_status)
