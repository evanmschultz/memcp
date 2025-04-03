# Graphiti MCP Server Refactoring Plan

## Current Code Analysis

The `old_graphiti_mcp_server.py` script implements a server that exposes Graphiti knowledge graph functionality through the Model Context Protocol (MCP). It's currently a monolithic script with mixed responsibilities:

1. **Entity Models**: Pydantic models for Requirement, Preference, and Procedure entities
2. **Response Types**: TypedDict definitions for API responses
3. **Configuration**: Settings for Graphiti client and MCP server
4. **Server Tools**: Functions exposing graph operations through MCP endpoints
5. **Queue Management**: Async queues for processing episodes
6. **Initialization**: Bootstrap functions and main entry point

The code needs modularization to improve maintainability, testability, and adherence to best practices.

## Proposed Project Structure

```
memcp/
├── __init__.py
├── cli.py                     # Entry point for command-line interface
├── api/
│   ├── __init__.py
│   ├── mcp_server.py          # MCP server configuration and initialization
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── episode_tools.py   # Episode-related MCP tools
│   │   ├── entity_tools.py    # Entity-related MCP tools
│   │   ├── search_tools.py    # Search-related MCP tools
│   │   └── graph_tools.py     # Graph management MCP tools
├── config/
│   ├── __init__.py
│   ├── settings.py            # Configuration models and loading
├── core/
│   ├── __init__.py
│   ├── client.py              # Graphiti client initialization and management
│   ├── queue.py               # Async queue management for episodes
│   ├── search.py              # Search configurations and utilities
├── models/
│   ├── __init__.py
│   ├── entities.py            # Entity models (Requirement, Preference, Procedure)
│   ├── responses.py           # Response type definitions
│   ├── requests.py            # Request type definitions
├── utils/
│   ├── __init__.py
│   ├── logging.py             # Logging utilities
│   ├── format.py              # Formatting utilities for responses
│   ├── errors.py              # Error handling utilities
└── tests/                     # Test directory mirroring the package structure
    ├── __init__.py
    ├── conftest.py            # Pytest fixtures
    ├── test_cli.py
    ├── api/
    │   ├── __init__.py
    │   ├── test_mcp_server.py
    │   └── tools/
    │       ├── __init__.py
    │       ├── test_episode_tools.py
    │       └── ...
    ├── core/
    │   ├── __init__.py
    │   ├── test_client.py
    │   └── ...
    └── ...
```

## Class and Module Design

### 1. `memcp/models/entities.py`

Convert the entity models into a proper class hierarchy:

```python
from pydantic import BaseModel, Field

class MemCPEntity(BaseModel):
    """Base class for all MemCP entities."""

    description: str = Field(
        ...,
        description="Description of the entity. Only use information mentioned "
        "in the context to write this description.",
    )

class Requirement(MemCPEntity):
    """A Requirement represents a specific need, feature, or functionality..."""

    project_name: str = Field(
        ...,
        description="The name of the project to which the requirement belongs.",
    )

class Preference(MemCPEntity):
    """A Preference represents a user's expressed like, dislike..."""

    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')",
    )

class Procedure(MemCPEntity):
    """A Procedure informing the agent what actions to take..."""
    pass
```

### 2. `memcp/config/settings.py`

Configuration classes with proper typing and validation:

```python
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client."""

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)
    model_name: str | None = Field(default=None)
    group_id: str | None = Field(default=None)
    use_custom_entities: bool = Field(default=False)

    @classmethod
    def from_env(cls) -> "GraphitiConfig":
        """Create a configuration instance from environment variables."""
        # Implementation
```

### 3. `memcp/core/client.py`

Graphiti client initialization and management:

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMClient
from ..config.settings import GraphitiConfig

class GraphitiClientManager:
    """Manager for Graphiti client initialization and access."""

    def __init__(self, config: GraphitiConfig, llm_client: LLMClient | None = None):
        """Initialize the Graphiti client manager."""
        self.config = config
        self.llm_client = llm_client
        self.client: Graphiti | None = None

    async def initialize(self, destroy_graph: bool = False) -> None:
        """Initialize the Graphiti client."""
        # Implementation

    async def get_client(self) -> Graphiti:
        """Get the initialized Graphiti client."""
        if self.client is None:
            await self.initialize()
        return self.client
```

### 4. `memcp/core/queue.py`

Queue management for asynchronous episode processing:

```python
import asyncio
import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)

class EpisodeQueueManager:
    """Manager for asynchronous episode processing queues."""

    def __init__(self):
        """Initialize the episode queue manager."""
        self.queues: Dict[str, asyncio.Queue] = {}
        self.workers: Dict[str, bool] = {}

    async def enqueue(self, group_id: str, process_func: Callable) -> int:
        """Enqueue an episode processing function for a specific group_id."""
        # Implementation

    async def start_worker(self, group_id: str) -> None:
        """Start a worker for processing episodes in a specific group_id."""
        # Implementation
```

### 5. `memcp/api/tools/episode_tools.py`

MCP tools for episode management:

```python
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from ...core.client import GraphitiClientManager
from ...core.queue import EpisodeQueueManager
from ...config.settings import GraphitiConfig

# Global instances managed by the CLI entry point
client_manager: Optional[GraphitiClientManager] = None
queue_manager: Optional[EpisodeQueueManager] = None
config: Optional[GraphitiConfig] = None

def register_tools(mcp: FastMCP) -> None:
    """Register episode-related tools with the MCP server."""

    @mcp.tool()
    async def add_episode(
        name: str,
        episode_body: str,
        group_id: str | None = None,
        source: str = "text",
        source_description: str = "",
        uuid: str | None = None,
    ) -> Dict[str, Any]:
        """Add an episode to the Graphiti knowledge graph."""
        # Implementation
```

### 6. `memcp/cli.py`

Command-line interface and entry point:

```python
#!/usr/bin/env python3
"""Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)."""

import argparse
import asyncio
import logging
import sys
import uuid

from .api.mcp_server import create_mcp_server, run_mcp_server
from .config.settings import GraphitiConfig
from .core.client import GraphitiClientManager
from .core.queue import EpisodeQueueManager
from .utils.logging import setup_logging

def main() -> None:
    """Main function to run the Graphiti MCP server."""

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description="Run the Graphiti MCP server with optional LLM client"
        )
        # Add arguments

        args = parser.parse_args()

        # Create configuration
        config = GraphitiConfig.from_env()
        if args.group_id:
            config.group_id = args.group_id
        else:
            config.group_id = f"graph_{uuid.uuid4().hex[:8]}"

        config.use_custom_entities = args.use_custom_entities

        # Run the server
        asyncio.run(run_server(config, args))
    except Exception as e:
        logger.error(f"Error initializing Graphiti MCP server: {str(e)}")
        sys.exit(1)

async def run_server(config: GraphitiConfig, args) -> None:
    """Run the MCP server with the provided configuration."""

    # Create managers
    client_manager = GraphitiClientManager(config)
    queue_manager = EpisodeQueueManager()

    # Initialize the Graphiti client
    await client_manager.initialize(destroy_graph=args.destroy_graph)

    # Create and run the MCP server
    mcp = create_mcp_server(client_manager, queue_manager, config)

    if args.transport == "stdio":
        await mcp.run_stdio_async()
    else:
        await mcp.run_sse_async()

if __name__ == "__main__":
    main()
```

## Design Improvements

1. **Dependency Injection**: Use dependency injection for better testability, separating the creation and use of objects.

2. **Error Handling**: Implement robust error handling with specific exception types and detailed error messages.

3. **Async Management**: Use AsyncContextManager (`async with`) for managing async resources properly.

4. **Client Singleton**: Implement a singleton pattern for the Graphiti client to ensure only one instance exists.

5. **Task Management**: Improve task management with proper cancellation and cleanup.

6. **Type Hints**: Use more specific type hints throughout the codebase instead of Any.

7. **Configuration Validation**: Add proper validation for configuration with helpful error messages.

## Test Strategy

1. **Unit Tests**: Write unit tests for each module, mocking external dependencies.

2. **Integration Tests**: Test the integration between modules and with external systems (Neo4j, OpenAI).

3. **End-to-End Tests**: Test the full functionality of the MCP server with simulated client interactions.

4. **Fixtures**: Use pytest fixtures for shared test setup.

5. **Coverage**: Aim for high test coverage (>80% as specified in pyproject.toml).

## Running the Refactored Code

After refactoring, the code can be run using the following methods:

### 1. As a Module

```bash
python -m memcp --transport sse --group-id my_graph --use-custom-entities
```

### 2. As a Package

```bash
# Install the package
pip install -e .

# Run the server
memcp --transport sse --group-id my_graph --use-custom-entities
```

### 3. With Docker

```bash
# Build the Docker image
docker build -t memcp .

# Run the container
docker run -p 8000:8000 --env-file .env memcp
```

## Migration Strategy

1. Implement the new modular structure without changing functionality
2. Ensure tests pass for the refactored code
3. Update documentation to reflect the new structure
4. Deprecate the old script with a warning encouraging users to use the new package

## Future Enhancements

1. Add a proper REST API interface in addition to MCP
2. Implement rate limiting and backoff strategies for API calls
3. Add metrics collection for performance monitoring
4. Implement caching for frequently accessed data
5. Add support for additional LLM providers
