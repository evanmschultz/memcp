class QueueManager:
    """Manages episode processing queues and workers.

    This class is responsible for:
    1. Managing queue creation and access
    2. Starting and tracking queue worker tasks
    3. Tracking queue statistics
    """

    def __init__(self) -> None:
        """Initialize the queue manager."""
        self.episode_queues: dict[str, asyncio.Queue] = {}
        self.queue_workers: dict[str, bool] = {}
        self.queue_stats_tracker = QueueStatsTracker()
        self.queue_progress_display = None  # Will be set from outside

    async def enqueue_task(self, group_id: str, process_func: Callable) -> None:
        """Enqueue a task for processing.

        Args:
            group_id: The group ID for the queue
            process_func: The async function to process
        """
        # Initialize queue for this group_id if it doesn't exist
        if group_id not in self.episode_queues:
            self.episode_queues[group_id] = asyncio.Queue()

        # Track the new task in our stats tracker
        self.queue_stats_tracker.add_task(group_id)

        # Update the progress display immediately
        if self.queue_progress_display:
            self.queue_progress_display.update()

        # Add the processing function to the queue
        await self.episode_queues[group_id].put(process_func)

        # Start a worker for this queue if one isn't already running
        if not self.queue_workers.get(group_id, False):
            asyncio.create_task(self.process_episode_queue(group_id))

    async def process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes
        from the queue one at a time.
        """
        # Name the task to be able to identify it during shutdown
        current_task = asyncio.current_task()
        if current_task:
            current_task.set_name(f"queue_worker_{group_id}")

        logger.info(
            f"Starting episode queue worker for group_id: [highlight]{group_id}[/highlight]"
        )
        self.queue_workers[group_id] = True

        try:
            while True:
                # Get the next episode processing function from the queue
                # This will wait if the queue is empty
                process_func = await self.episode_queues[group_id].get()

                # Generate a unique task ID for tracking
                task_id = str(uuid.uuid4())

                try:
                    # Record that processing has started
                    self.queue_stats_tracker.start_processing(group_id, task_id)

                    # Update the progress display
                    if self.queue_progress_display:
                        self.queue_progress_display.update()

                    # Process the episode
                    await process_func()

                    # Record successful completion
                    self.queue_stats_tracker.complete_task(group_id, task_id, success=True)
                except Exception as e:
                    # Record failed completion
                    self.queue_stats_tracker.complete_task(group_id, task_id, success=False)

                    logger.error(
                        f"Error processing queued episode for group_id [highlight]{group_id}[/highlight]: [danger]{str(e)}[/danger]"
                    )
                finally:
                    # Mark the task as done regardless of success/failure
                    self.episode_queues[group_id].task_done()

                    # Update the progress display
                    if self.queue_progress_display:
                        self.queue_progress_display.update()
        except asyncio.CancelledError:
            logger.info(
                f"Episode queue worker for group_id [highlight]{group_id}[/highlight] was [success]cancelled[/success]"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in queue worker for group_id [highlight]{group_id}[/highlight]: [danger]{str(e)}[/danger]"
            )
        finally:
            self.queue_workers[group_id] = False
            logger.info(
                f"[success]Stopped[/success] episode queue worker for group_id: [highlight]{group_id}[/highlight]"
            )


class EpisodeService:
    """Service for managing episodes in the Graphiti knowledge graph.

    This service is responsible for adding, retrieving, and deleting episodes.
    """

    def __init__(
        self, graphiti_client: Graphiti | None, config: Any, queue_manager: QueueManager
    ) -> None:
        """Initialize the episode service.

        Args:
            graphiti_client: The Graphiti client instance
            config: Configuration for the service
            queue_manager: The queue manager for processing episodes
        """
        self.graphiti_client = graphiti_client
        self.config = config
        self.queue_manager = queue_manager

    async def add_episode(
        self,
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
        """
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
                        f"Processing queued episode '[highlight]{name}[/highlight]' for graph_id: [highlight]{graph_id_str}[/highlight]"
                    )
                    # Use all entity types if use_custom_entities is enabled, otherwise use empty dict
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
                        f"Episode '[highlight]{name}[/highlight]' [success]added successfully[/success]"
                    )

                    logger.info(
                        f"Building communities after episode '[highlight]{name}[/highlight]'"
                    )
                    await client.build_communities()

                    logger.info(
                        f"Episode '[highlight]{name}[/highlight]' [success]processed successfully[/success]"
                    )
                except Exception as e:
                    error_msg = str(e)
                    logger.error(
                        f"[danger]Error[/danger] processing episode '[highlight]{name}[/highlight]' for graph_id [highlight]{graph_id_str}[/highlight]: [danger]{error_msg}[/danger]"
                    )

            # Enqueue the task using the queue manager
            await self.queue_manager.enqueue_task(graph_id_str, process_episode)

            # Return immediately with a success message
            return {
                "message": f"Episode '[highlight]{name}[/highlight]' queued for processing (position: [highlight]{self.queue_manager.episode_queues[graph_id_str].qsize()}[/highlight])"
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[danger]Error queuing episode task[/danger]: {error_msg}")
            return {"error": f"Error queuing episode task: {error_msg}"}

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


class SearchService:
    """Service for searching the Graphiti knowledge graph.

    This service is responsible for searching nodes and facts.
    """

    def __init__(self, graphiti_client: Graphiti | None, config: Any) -> None:
        """Initialize the search service.

        Args:
            graphiti_client: The Graphiti client instance
            config: Configuration for the service
        """
        self.graphiti_client = graphiti_client
        self.config = config

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

        Note: entity is a single entity type to filter results (permitted: "Preference", "Procedure").

        Args:
            query: The search query
            graph_ids: Optional list of group IDs to filter results
            max_nodes: Maximum number of nodes to return (default: 10)
            center_node_uuid: Optional UUID of a node to center the search around
            entity: Optional single entity type to filter results (permitted: "Preference", "Procedure")
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


class EntityService:
    """Service for managing entity edges in the Graphiti knowledge graph.

    This service is responsible for retrieving and deleting entity edges.
    """

    def __init__(self, graphiti_client: Graphiti | None) -> None:
        """Initialize the entity service.

        Args:
            graphiti_client: The Graphiti client instance
        """
        self.graphiti_client = graphiti_client

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


class GraphManagementService:
    """Service for managing the Graphiti knowledge graph.

    This service is responsible for operations like clearing the graph.
    """

    def __init__(self, graphiti_client: Graphiti | None) -> None:
        """Initialize the graph management service.

        Args:
            graphiti_client: The Graphiti client instance
        """
        self.graphiti_client = graphiti_client

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
                "message": f"Graphiti MCP server is running but Neo4j connection failed: {error_msg}",
            }


class GraphitiToolRegistry:
    """Registry for all Graphiti MCP tools.

    This class acts as a coordinator between the MCP instance and the various
    service classes that implement the actual functionality.
    """

    def __init__(self, mcp_instance: FastMCP) -> None:
        """Initialize the tool registry with an MCP instance.

        Args:
            mcp_instance: The MCP instance to register tools with
        """
        self.mcp = mcp_instance
        self.graphiti_client = None  # Will be initialized later
        self.config = None  # Will be set from outside

        # Initialize services
        self.queue_manager = QueueManager()
        self.episode_service = None  # Will be initialized after config is set
        self.search_service = None  # Will be initialized after config is set
        self.entity_service = None  # Will be initialized after config is set
        self.graph_service = None  # Will be initialized after config is set

        # The progress display will be set from outside
        self.queue_progress_display = None

    def initialize_services(self) -> None:
        """Initialize all services with the current configuration.

        This method should be called after self.config and self.graphiti_client are set.
        """
        if self.config is None:
            raise ValueError("Config must be set before initializing services")

        # Initialize services with dependencies
        self.episode_service = EpisodeService(self.graphiti_client, self.config, self.queue_manager)
        self.search_service = SearchService(self.graphiti_client, self.config)
        self.entity_service = EntityService(self.graphiti_client)
        self.graph_service = GraphManagementService(self.graphiti_client)

        # Register all tools with the MCP instance
        self._register_tools()

    def set_progress_display(self, progress_display: QueueProgressDisplay) -> None:
        """Set the progress display for the queue manager.

        Args:
            progress_display: The progress display instance
        """
        self.queue_progress_display = progress_display
        self.queue_manager.queue_progress_display = progress_display

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

    # Tool method delegates

    # Episode service delegates
    async def add_episode(
        self,
        name: str,
        episode_body: str,
        graph_id: str | None = None,
        source: str = "text",
        source_description: str = "",
        uuid: str | None = None,
    ) -> SuccessResponse | ErrorResponse:
        """Delegate to EpisodeService.add_episode."""
        return await self.episode_service.add_episode(
            name, episode_body, graph_id, source, source_description, uuid
        )

    async def delete_episode(self, uuid: str) -> SuccessResponse | ErrorResponse:
        """Delegate to EpisodeService.delete_episode."""
        return await self.episode_service.delete_episode(uuid)

    async def get_episodes(
        self, graph_id: str | None = None, last_n: int = 10
    ) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
        """Delegate to EpisodeService.get_episodes."""
        return await self.episode_service.get_episodes(graph_id, last_n)

    # Search service delegates
    async def search_nodes(
        self,
        query: str,
        graph_ids: list[str] | None = None,
        max_nodes: int = 10,
        center_node_uuid: str | None = None,
        entity: str = "",
    ) -> NodeSearchResponse | ErrorResponse:
        """Delegate to SearchService.search_nodes."""
        return await self.search_service.search_nodes(
            query, graph_ids, max_nodes, center_node_uuid, entity
        )

    async def search_facts(
        self,
        query: str,
        graph_ids: list[str] | None = None,
        max_facts: int = 10,
        center_node_uuid: str | None = None,
    ) -> FactSearchResponse | ErrorResponse:
        """Delegate to SearchService.search_facts."""
        return await self.search_service.search_facts(query, graph_ids, max_facts, center_node_uuid)

    # Entity service delegates
    async def delete_entity_edge(self, uuid: str) -> SuccessResponse | ErrorResponse:
        """Delegate to EntityService.delete_entity_edge."""
        return await self.entity_service.delete_entity_edge(uuid)

    async def get_entity_edge(self, uuid: str) -> dict[str, Any] | ErrorResponse:
        """Delegate to EntityService.get_entity_edge."""
        return await self.entity_service.get_entity_edge(uuid)

    # Graph management service delegates
    async def clear_graph(self) -> SuccessResponse | ErrorResponse:
        """Delegate to GraphManagementService.clear_graph."""
        return await self.graph_service.clear_graph()

    async def get_status(self) -> StatusResponse:
        """Delegate to GraphManagementService.get_status."""
        return await self.graph_service.get_status()


# Updates needed to initialize_server and run_mcp_server functions:
'''
async def initialize_server() -> tuple[MCPConfig, FastMCP, GraphitiToolRegistry]:
    """Parse args and create both config and MCP instance."""
    # ... existing argument parsing code ...
    
    # Create the MCP instance
    mcp_instance = FastMCP(
        "graphiti",
        instructions=GRAPHITI_MCP_INSTRUCTIONS,
        settings={"host": mcp_config.host, "port": mcp_config.port},
    )
    
    # Create tool registry
    tool_registry = GraphitiToolRegistry(mcp_instance)
    tool_registry.config = config  # Set the config
    
    # Initialize Graphiti with the specified LLM client
    await initialize_graphiti(llm_client, destroy_graph=args.destroy_graph)
    
    # Set the graphiti_client on the tool registry and initialize services
    tool_registry.graphiti_client = graphiti_client
    tool_registry.initialize_services()

    return mcp_config, mcp_instance, tool_registry

async def run_mcp_server() -> None:
    """Run the MCP server in the current event loop."""
    # Get config, server instance, and tool registry
    mcp_config, mcp, tool_registry = await initialize_server()
    
    # Set up signal handlers for shutdown
    setup_signal_handlers()

    # Initialize the queue progress display
    queue_progress_display = QueueProgressDisplay(console, tool_registry.queue_manager.queue_stats_tracker)
    queue_progress_display.create_progress_tasks()
    
    # Set the progress display on the tool registry
    tool_registry.set_progress_display(queue_progress_display)

    # ... rest of the existing code ...
'''


# next create and update file to use MCPFactory
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
