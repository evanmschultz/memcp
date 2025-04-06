# Implementing Pydantic Settings CLI for MemCP

This document outlines all changes required to refactor MemCP to use Pydantic's built-in CLI support through the `pydantic_settings` package.

## Overview of Changes

1. **Install dependencies**
2. **Update config models** in `settings.py`
3. **Refactor providers** in `providers.py`
4. **Update config manager** in `config_manager.py`
5. **Modify application builder** in `app_builder.py`
6. **Simplify CLI entry point** in `cli.py`

## 1. Install Dependencies

Add `pydantic-settings` to your dependencies:

```bash
pip install pydantic-settings
# or with poetry
poetry add pydantic-settings
```

## 2. Update Config Models

### Current Implementation

```python
# memcp/config/settings.py
from pydantic import BaseModel, Field

class MemCPConfig(BaseModel):
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j user")
    neo4j_password: str
    # other fields...

class MCPConfig(BaseModel):
    mcp_name: str = Field(default="memcp", description="Name of the MCP server")
    # other fields...
```

### New Implementation

```python
# memcp/config/settings.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ServerConfig(BaseSettings):
    """Server configuration settings."""
    transport: str = Field("sse", description="Transport type (sse or stdio)")
    host: str = Field("127.0.0.1", description="Host address for the server")
    port: int = Field(8000, description="Port number for the server")

class Neo4jConfig(BaseSettings):
    """Neo4j database connection settings."""
    uri: str = Field("bolt://localhost:7687", description="Neo4j URI")
    user: str = Field("neo4j", description="Neo4j user")
    password: str = Field(..., description="Neo4j password")

class OpenAIConfig(BaseSettings):
    """OpenAI API settings."""
    api_key: str = Field(..., description="OpenAI API key")
    base_url: str | None = Field(None, description="OpenAI API base URL")

class GraphConfig(BaseSettings):
    """Graph configuration settings."""
    id: str | None = Field(None, description="Graph ID")
    use_custom_entities: bool = Field(False, description="Enable entity extraction using predefined entities")

class ModelConfig(BaseSettings):
    """Model configuration settings."""
    name: str = Field("gpt-4o-mini", description="Model name to use with the LLM client")

class MCPConfig(BaseSettings):
    """MCP server configuration."""
    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8"
    )

    mcp_name: str = Field("memcp", description="Name of the MCP server")
    mcp_instructions: str = Field(
        default=None,  # We'll need to import GraphitiInstructions
        description="Instructions for the MCP server"
    )
    transport: str = Field("sse", description="Transport mechanism")
    host: str = Field("127.0.0.1", description="Host address")
    port: int = Field(8000, description="Port number")

class MemCPConfig(BaseSettings):
    """Configuration for MemCP.

    Centralizes all configuration parameters for the MemCP server,
    including database connection details and LLM settings.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        cli_parse_args=True,  # Enable CLI parsing
        case_sensitive=False
    )

    neo4j: Neo4jConfig
    openai: OpenAIConfig
    model: ModelConfig
    graph: GraphConfig
    server: ServerConfig
    destroy_graph: bool = Field(False, description="Destroy all graphs")

    def model_post_init(self, __context):
        """Post-initialization processing."""
        # Anything we need to do after model initialization

    # The static methods can remain
    @staticmethod
    def get_openai_api_key() -> str:
        """Get the OpenAI API key from configuration."""
        # Now we'll use the instance property instead of env var directly
        return MemCPConfig().openai.api_key
```

## 3. Refactor Providers

With pydantic-settings, we don't need separate providers as it handles environment variables, CLI arguments, and `.env` files automatically. However, we still need TOML configuration.

```python
# memcp/config/providers.py
from typing import Any, Protocol, runtime_checkable
import tomli

@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol defining a configuration provider interface."""

    def get_config(self) -> dict[str, Any]:
        """Get configuration from this provider."""
        ...

class TomlConfigProvider:
    """Provider for TOML file configuration."""

    def __init__(self, config_path: str = "config.toml") -> None:
        """Initialize with path to TOML config file."""
        self.config_path = config_path

    def get_config(self) -> dict[str, Any]:
        """Extract configuration from TOML file."""
        try:
            with open(self.config_path, "rb") as f:
                return tomli.load(f)
        except (FileNotFoundError, tomli.TOMLDecodeError):
            # Return empty config if file doesn't exist or is invalid
            return {}
```

## 4. Update Config Manager

Simplified version since most config management is handled by pydantic-settings:

```python
# memcp/config/config_manager.py
from typing import Any, Optional
from pydantic_settings import BaseSettings, CliApp
from memcp.config.settings import MCPConfig, MemCPConfig
from memcp.config.providers import TomlConfigProvider
from memcp.utils import get_logger

class ConfigManager:
    """Simplified manager for loading configuration."""

    def __init__(self, toml_path: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            toml_path: Optional path to TOML config file
        """
        self.logger = get_logger(__name__)
        self.toml_path = toml_path
        self._memcp_config = None
        self._mcp_config = None
        self._toml_config = None

    @property
    def toml_config(self) -> dict[str, Any]:
        """Get configuration from TOML file."""
        if self._toml_config is None:
            if self.toml_path:
                provider = TomlConfigProvider(self.toml_path)
                self._toml_config = provider.get_config()
            else:
                self._toml_config = {}
        return self._toml_config

    def create_memcp_config(self) -> MemCPConfig:
        """Create a MemCPConfig instance.

        Returns:
            Configured MemCPConfig instance
        """
        if self._memcp_config is None:
            # Merge TOML config with other sources
            # Pydantic-settings handles CLI args and env vars automatically
            toml_dict = self.toml_config

            # Initialize with TOML config first, then CLI args and env vars will apply
            self._memcp_config = MemCPConfig.model_validate(toml_dict)

            # Log details
            if self._memcp_config.graph.id:
                self.logger.info(f"Using graph_id: {self._memcp_config.graph.id}")
            if self._memcp_config.graph.use_custom_entities:
                self.logger.info("Entity extraction enabled using predefined MEMCP_ENTITIES")
            else:
                self.logger.info("Entity extraction disabled (no custom entities will be used)")

        return self._memcp_config

    def create_mcp_config(self) -> MCPConfig:
        """Create an MCPConfig from the merged configuration.

        Returns:
            Configured MCPConfig instance
        """
        if self._mcp_config is None:
            # Extract server config from MemCPConfig
            memcp_config = self.create_memcp_config()

            # Create MCP config
            self._mcp_config = MCPConfig(
                transport=memcp_config.server.transport,
                host=memcp_config.server.host,
                port=memcp_config.server.port,
            )

        return self._mcp_config

    def should_destroy_graph(self) -> bool:
        """Check if the graph should be destroyed.

        Returns:
            True if the graph should be destroyed, False otherwise
        """
        return self.create_memcp_config().destroy_graph

    @classmethod
    def create_default(cls, config_path: Optional[str] = None) -> "ConfigManager":
        """Create a ConfigManager with default settings.

        Args:
            config_path: Optional path to TOML config file

        Returns:
            Configured ConfigManager instance
        """
        return cls(toml_path=config_path)
```

## 5. Modify Application Builder

Update to use the new configuration approach:

```python
# memcp/app_builder.py
from memcp.api.memcp_server import MemCPServer
from memcp.config import ConfigManager, MCPConfig, MemCPConfig
from memcp.console import DisplayManager, QueueProgressDisplay
from memcp.core.queue import QueueManager, QueueStatsTracker
from memcp.llm.llm_factory import LLMClientFactory
from memcp.utils import get_logger
from memcp.utils.shutdown import ShutdownManager

from graphiti_core.llm_client import LLMClient

class ApplicationBuilder:
    """Builder for creating and wiring application components."""

    __slots__ = (
        "logger",
        "_config_manager",
        "_memcp_config",
        "_mcp_config",
        "_llm_factory",
        "_llm_client",
        "_display_manager",
        "_shutdown_manager",
        "_queue_stats_tracker",
        "_queue_manager",
        "_queue_progress_display",
    )

    def __init__(self) -> None:
        """Initialize the application builder."""
        self.logger = get_logger(__name__)
        # Other attributes will be created on demand via properties

    def configure(self, config_path: str | None = None) -> None:
        """Configure the application.

        Args:
            config_path: Optional path to TOML config file
        """
        # Create and configure the config manager
        self._config_manager = ConfigManager.create_default(config_path)

        # Create the configuration objects
        self._memcp_config = self._config_manager.create_memcp_config()
        self._mcp_config = self._config_manager.create_mcp_config()

    # The rest of the class can remain largely unchanged
    # Just update the llm_client property to use the new config structure:

    @property
    def llm_client(self) -> LLMClient:
        """Get the LLM client, creating it if necessary."""
        if not hasattr(self, "_llm_client"):
            self._llm_client = self.llm_factory.create_openai_client(
                api_key=self.memcp_config.openai.api_key,
                model=self.memcp_config.model.name
            )
        return self._llm_client
```

## 6. Simplify CLI Entry Point

```python
# memcp/cli.py
#!/usr/bin/env python3
"""Command-line interface for MemCP."""

import asyncio
import sys
from dotenv import load_dotenv
from pydantic_settings import CliApp

from memcp.app_builder import ApplicationBuilder
from memcp.config.settings import MemCPConfig
from memcp.utils import configure_logging, get_logger
from memcp.utils.errors import ServerInitializationError, ServerRuntimeError

# Load environment variables
load_dotenv()

# Configure logging
configure_logging()

# Get a logger for this module
logger = get_logger(__name__)

async def run_server(config_path: str | None = None) -> None:
    """Create and run the MemCP server asynchronously.

    Args:
        config_path: Optional path to TOML config file
    """
    try:
        builder = ApplicationBuilder()
        builder.configure(config_path)
        server = await builder.build()
        await server.run()
    except Exception as e:
        logger.error(f"Error running MemCP server: {str(e)}")
        raise

def main() -> None:
    """Main entry point for the MemCP CLI.

    Creates and runs the MemCP server within an asyncio event loop.
    """
    try:
        # Use CliApp to get configuration
        config = CliApp.run(MemCPConfig)

        # Get TOML config path if specified
        toml_path = getattr(config, "config_file", None)

        # Run the server with the configuration
        asyncio.run(run_server(toml_path))
    finally:
        # Restore stderr to its original state (for Rich console)
        sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()
```

## Summary of Changes

1. **Pydantic Settings Migration**:

    - Updated models to use `BaseSettings` instead of `BaseModel`
    - Added nested configuration structures for better organization
    - Configured environment variable and CLI parsing

2. **Provider Simplification**:

    - Removed `ArgsConfigProvider` and `EnvConfigProvider`
    - Kept `TomlConfigProvider` for custom file-based config

3. **Config Manager Updates**:

    - Simplified to focus on TOML integration
    - Leverages pydantic-settings for env vars and CLI args

4. **Application Builder Adjustments**:

    - Modified how configurations are loaded
    - Updated property access for nested config structure

5. **CLI Entry Point Improvements**:
    - Uses `CliApp.run()` for automatic CLI parsing
    - Maintains error handling and async patterns

This implementation preserves all existing functionality while simplifying the configuration management process significantly.

## Additional Considerations

1. **Migration Path**: Consider creating backwards compatibility wrappers during transition
2. **Testing**: Update tests to use the new configuration structure
3. **Documentation**: Update user documentation to reflect CLI argument changes
4. **Defaults**: Ensure all default values are properly set and documented
