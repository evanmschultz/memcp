# MemCP Configuration System Improvements

## Current Issues

After reviewing the MemCP codebase, I've identified several issues with the current configuration system:

1. **Configuration Duplication**: The project has two different `GraphitiAdapterConfig` classes:

    - In `config/settings.py` with basic settings
    - In `clients/graphiti_adapter.py` with validators and more domain-specific fields

2. **Auto-Instantiation Problem**: In `MemCPConfig`, configuration classes are instantiated directly:

    ```python
    graphiti: GraphitiAdapterConfig = GraphitiAdapterConfig()
    ```

    This doesn't allow for dependency injection or setting non-default values easily.

3. **Hardcoded Defaults**: Sensitive values like passwords have default values (`"CHANGE_ME"`), which is a security risk.

4. **Outdated Pydantic Patterns**: The code isn't using the latest Pydantic v2 patterns and best practices.

5. **Manual Environment Variable Handling**: Environment variables are manually parsed instead of using Pydantic's built-in functionality.

## Key Improvements Based on Pydantic v2 Best Practices

### 1. Using BaseSettings from pydantic-settings

In Pydantic v2, `BaseSettings` was moved to a separate package called `pydantic-settings`. This is the correct approach for configuration classes that need to load values from environment variables.

### 2. Modern Configuration Approach

Instead of using inner `Config` classes, Pydantic v2 recommends using `model_config = ConfigDict(...)` or `model_config = SettingsConfigDict(...)` for configuration settings.

### 3. Consolidating Configuration Classes

We'll consolidate the two different `GraphitiAdapterConfig` classes, taking the best aspects from both implementations:

-   Field validators from the client version
-   All useful fields from both versions
-   Proper integration with `EpisodeType` enum from graphiti_core

### 4. Proper Dependency Injection

Instead of auto-instantiating config classes, we'll implement factory methods that create fully initialized instances, making the code more flexible and testable.

## Improved Configuration Implementation

### Main Configuration Module

```python
"""Configuration settings for the MemCP server.

This module provides Pydantic models for configuration management,
supporting both TOML file and environment variable based configuration.
"""

from pathlib import Path
from typing import Any, Optional, Dict, ClassVar

import tomli
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from graphiti_core.nodes import EpisodeType


class GraphitiAdapterConfig(BaseSettings):
    """Configuration for the Graphiti adapter.

    This class defines the configuration parameters needed to initialize and connect
    to a Graphiti instance. It includes Neo4j connection details and optional settings
    for parallel runtime and other features.

    Attributes:
        neo4j_uri (str): URI for the Neo4j database (e.g., "bolt://localhost:7687").
        neo4j_user (str): Username for Neo4j authentication.
        neo4j_password (str): Password for Neo4j authentication.
        use_parallel_runtime (bool): Whether to enable Neo4j's parallel runtime feature.
            Note that this is not supported for Neo4j Community edition or smaller
            AuraDB instances. Defaults to False.
        default_episode_type (str): Default type for episodes when not specified.
            Must be one of the valid EpisodeType values. Defaults to "text".
        default_source_description (str): Default description for episode sources
            when not specified. Defaults to "memcp".
        openai_api_key (Optional[str]): Optional OpenAI API key for LLM operations.
        openai_base_url (Optional[str]): Optional OpenAI API base URL.
        model_name (Optional[str]): Optional model name for LLM operations.
        group_id (Optional[str]): Optional group ID for organizing graph data.
        use_custom_entities (bool): Whether to use custom entity types.
    """

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    use_parallel_runtime: bool = False
    default_episode_type: str = "text"
    default_source_description: str = "memcp"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    model_name: Optional[str] = None
    group_id: Optional[str] = None
    use_custom_entities: bool = False

    model_config = SettingsConfigDict(
        env_prefix="MEM_G_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @field_validator("default_episode_type")
    def validate_episode_type(cls, v: str) -> str:
        """Validate that the episode type is one of the allowed values.

        Args:
            v (str): The episode type value to validate.

        Returns:
            str: The validated episode type.

        Raises:
            ValueError: If the episode type is not valid.
        """
        try:
            EpisodeType[v]
            return v
        except KeyError:
            valid_types = ", ".join(t.name for t in EpisodeType)
            raise ValueError(
                f"Invalid episode type: {v}. Must be one of: {valid_types}"
            )

    @field_validator("neo4j_uri")
    def validate_neo4j_uri(cls, v: str) -> str:
        """Validate that the Neo4j URI is properly formatted.

        Args:
            v (str): The URI value to validate.

        Returns:
            str: The validated URI.

        Raises:
            ValueError: If the URI is not valid.
        """
        if not v.startswith(("bolt://", "neo4j://", "neo4j+s://")):
            raise ValueError(
                "Invalid Neo4j URI. Must start with bolt://, neo4j://, or neo4j+s://"
            )
        return v


class MCPConfig(BaseSettings):
    """Configuration for the MCP server.

    Attributes:
        transport: Transport type for MCP communication ("sse" or "stdio")
    """

    transport: str = "sse"

    model_config = SettingsConfigDict(
        env_prefix="MEM_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class LoggingConfig(BaseSettings):
    """Configuration for logging.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format
        file_path: Optional path for log file output
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup log files to keep
        use_rich_console: Whether to use Rich for console output
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        rich_tracebacks: Whether to use Rich for traceback formatting
    """

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[Path] = None
    max_file_size: int = Field(default=10 * 1024 * 1024, gt=0)  # 10MB
    backup_count: int = Field(default=5, gt=0)
    use_rich_console: bool = True
    log_to_console: bool = True
    log_to_file: bool = False
    rich_tracebacks: bool = True

    model_config = SettingsConfigDict(
        env_prefix="MEM_LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class MemCPConfig(BaseModel):
    """Main configuration for the MemCP server.

    This class provides a hybrid approach to configuration management,
    supporting both TOML files and environment variables.

    Attributes:
        graphiti: Graphiti adapter configuration
        mcp: MCP server configuration
        logging: Logging configuration
    """

    graphiti: GraphitiAdapterConfig
    mcp: MCPConfig
    logging: LoggingConfig

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def load(cls, env_file: Optional[str] = None) -> "MemCPConfig":
        """Load configuration from environment variables with optional .env file.

        Args:
            env_file: Optional path to the environment file

        Returns:
            A new MemCPConfig instance with settings from environment variables
        """
        env_kwargs = {"env_file": env_file} if env_file else {}

        return cls(
            graphiti=GraphitiAdapterConfig(**env_kwargs),
            mcp=MCPConfig(**env_kwargs),
            logging=LoggingConfig(**env_kwargs)
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "MemCPConfig":
        """Load configuration from a TOML file.

        Args:
            path: Path to the TOML configuration file

        Returns:
            A new MemCPConfig instance

        Raises:
            FileNotFoundError: If the TOML file doesn't exist
            ValueError: If the TOML file contains invalid configuration
            tomli.TOMLDecodeError: If the TOML file is malformed
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("rb") as f:
            try:
                data = tomli.load(f)
            except tomli.TOMLDecodeError as e:
                raise tomli.TOMLDecodeError(msg=str(e), doc=str(e), pos=0)

        try:
            # Extract section data with fallback to empty dict
            graphiti_data = data.get("graphiti", {})
            mcp_data = data.get("mcp", {})
            logging_data = data.get("logging", {})

            # Create config with data from TOML
            return cls(
                graphiti=GraphitiAdapterConfig(**graphiti_data),
                mcp=MCPConfig(**mcp_data),
                logging=LoggingConfig(**logging_data)
            )
        except Exception as e:
            raise ValueError(f"Invalid configuration values: {e}")
```

### Environment Variables Configuration (.env.example)

```ini
# Graphiti adapter configuration
MEM_G_NEO4J_URI=bolt://localhost:7687
MEM_G_NEO4J_USER=neo4j
MEM_G_NEO4J_PASSWORD=your_secure_password
MEM_G_USE_PARALLEL_RUNTIME=false
MEM_G_DEFAULT_EPISODE_TYPE=text
MEM_G_DEFAULT_SOURCE_DESCRIPTION=memcp
# MEM_G_OPENAI_API_KEY=your_openai_key
# MEM_G_OPENAI_BASE_URL=https://api.openai.com/v1
# MEM_G_MODEL_NAME=gpt-4
# MEM_G_GROUP_ID=your_group_id
# MEM_G_USE_CUSTOM_ENTITIES=false

# MCP server configuration
MEM_MCP_TRANSPORT=sse

# Logging configuration
MEM_LOG_LEVEL=INFO
MEM_LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
# MEM_LOG_FILE_PATH=/path/to/logfile.log
MEM_LOG_MAX_FILE_SIZE=10485760  # 10MB
MEM_LOG_BACKUP_COUNT=5
MEM_LOG_USE_RICH_CONSOLE=true
MEM_LOG_LOG_TO_CONSOLE=true
MEM_LOG_LOG_TO_FILE=false
MEM_LOG_RICH_TRACEBACKS=true
```

### Usage Example

```python
"""Example usage of the MemCP configuration system."""

from memcp.config.config import MemCPConfig
from memcp.clients.graphiti_adapter import GraphitiAdapter
import logging
import asyncio


async def main():
    # Option 1: Load from environment variables (and .env file if present)
    config = MemCPConfig.load()

    # Option 2: Load from a specific environment file
    # config = MemCPConfig.load(env_file=".env.production")

    # Option 3: Load from a TOML file
    # config = MemCPConfig.from_toml("config.toml")

    # Set up logging using the config
    logging_level = getattr(logging, config.logging.level.upper())
    logging.basicConfig(
        level=logging_level,
        format=config.logging.format,
    )
    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded successfully")

    # Create GraphitiAdapter using the config
    adapter = GraphitiAdapter(config.graphiti)

    # Connect to the database
    try:
        await adapter.connect()
        logger.info("Connected to Neo4j database")

        # Use the adapter to perform operations
        # For example, add an episode
        result = await adapter.add_episode(
            name="Test Episode",
            episode_body="This is a test episode content.",
            group_id=config.graphiti.group_id,
        )
        logger.info(f"Added episode: {result}")

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Close connection
        await adapter.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    asyncio.run(main())
```

## Implementation Details

### 1. Using BaseSettings from pydantic-settings

In Pydantic v2, `BaseSettings` has moved to a separate package called `pydantic-settings`. This is the correct approach for configuration classes that should load values from environment variables. All configuration classes that need to be populated from environment variables now inherit from `BaseSettings` instead of `BaseModel`.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class GraphitiAdapterConfig(BaseSettings):
    # Fields and config...
```

### 2. Modern Configuration Approach

Instead of using inner `Config` classes, we now use the recommended Pydantic v2 pattern:

```python
model_config = SettingsConfigDict(
    env_prefix="MEM_G_",
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore"
)
```

This maintains compatibility with future versions of Pydantic.

### 3. Consolidating Configuration Classes

We've combined the best aspects of both `GraphitiAdapterConfig` classes:

-   Field validators from the client version
-   All useful fields from both versions
-   Proper integration with `EpisodeType` enum from graphiti_core
-   Removed default value for password (now required)

### 4. Proper Dependency Injection

Instead of auto-instantiating config classes, we've implemented factory methods:

-   `MemCPConfig.load()` - Loads from environment variables
-   `MemCPConfig.from_toml()` - Loads from a TOML file

This makes the code more flexible, testable, and dependency-injection friendly.

### 5. Environment Variable Handling

Each configuration class now has:

-   Appropriate environment variable prefixes (`MEM_G_`, `MEM_MCP_`, etc.)
-   Built-in support for loading from `.env` files
-   Proper settings for handling extra fields

## Benefits of the New Approach

1. **No Hardcoded Secrets**: Passwords and other sensitive information must be provided through environment variables or configuration files rather than being hardcoded with defaults.

2. **Configuration Consistency**: Single source of truth for all configuration settings, eliminating duplication and potential inconsistencies.

3. **Better Testability**: The factory methods make it easy to create configuration objects with test values, improving testability.

4. **Future-Proof**: Using the latest Pydantic v2 patterns ensures compatibility with future versions.

5. **Type Safety**: Proper typing and validation throughout the configuration system.

6. **Simplified Environment Variables**: Automatic loading of environment variables with appropriate prefixes, making it easy to configure the application in different environments.

7. **Improved Documentation**: Comprehensive docstrings and type hints make the configuration system more maintainable.

## Implementation Steps

1. Install required packages:

    ```bash
    uv pip install pydantic-settings
    ```

2. Create or update the configuration module (`config/config.py`) with the new implementation.

3. Create a `.env.example` file to document the available environment variables.

4. Update existing code to use the new configuration system:

    - Remove the old `update_from_env()` method usage
    - Use the factory methods to create configuration instances
    - Pass the configuration objects explicitly where needed

5. Update any import statements for the `GraphitiAdapterConfig` class to use the consolidated version.

6. Sync development dependencies:

    ```bash
    uv sync --extra dev
    ```

    This command will install all dependencies including those specified in the `dev` extra section of your pyproject.toml file.

## Conclusion

This improved configuration system addresses all the issues identified in the current implementation, making it more robust, secure, and maintainable. By following Pydantic v2 best practices, we ensure that the code is compatible with future versions and takes advantage of the latest features and improvements.
