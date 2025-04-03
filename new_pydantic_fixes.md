# MemCP Configuration System Implementation Guide

This guide provides a comprehensive implementation plan for the improved MemCP configuration system. Based on our iterative development process, this final version addresses all identified issues and follows best practices for Pydantic v2 and Python 3.10+.

## Key Improvements

1. **Consolidated Configuration Classes**: Merged duplicate `GraphitiAdapterConfig` classes
2. **Dependency Injection Support**: Implemented factory methods for proper instantiation
3. **Simplified Environment Variables**: Removed prefixes for cleaner configuration
4. **Type-Safe Validations**: Added proper enum and literal type validation
5. **Python 3.10 Compatibility**: Ensured support for Python 3.10+ with appropriate dependencies
6. **Pydantic v2 Best Practices**: Updated to match latest patterns and recommendations
7. **Proper Exception Handling**: Added exception chaining for better error diagnosis
8. **Field Validator Syntax**: Fixed decorator syntax for Pydantic v2 compatibility

## Implementation Details

### 1. Direct Enum Usage for Episode Type

Instead of using string values, we now use the `EpisodeType` enum directly:

```python
default_episode_type: EpisodeType = EpisodeType.text
```

This provides better type safety and integration with the `graphiti_core` library.

### 2. Simplified Environment Variables

Environment variables now use simpler names without the lengthy prefixes:

```python
# Old approach with prefixes
MEMCP_GRAPHITI_NEO4J_URI=bolt://localhost:7687

# New approach - cleaner and more concise
NEO4J_URI=bolt://localhost:7687
```

This is implemented by setting the `env_prefix` to an empty string:

```python
model_config = SettingsConfigDict(
    env_prefix="",
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore"
)
```

### 3. Field Validator Syntax for Pydantic v2

Pydantic v2 requires field validators to include the `@classmethod` decorator:

```python
@field_validator("neo4j_uri")
@classmethod
def validate_neo4j_uri(cls, v: str) -> str:
    """Validate the Neo4j URI format."""
    if not v.startswith(("bolt://", "neo4j://", "neo4j+s://")):
        raise ValueError(
            "Invalid Neo4j URI. Must start with bolt://, neo4j://, or neo4j+s://"
        )
    return v
```

### 4. Literal Type for Transport Validation

Added a Literal type to ensure the transport value is either "sse" or "stdio":

```python
transport: Literal["sse", "stdio"] = "sse"
```

This provides better validation and autocompletion support compared to using strings.

### 5. Proper Environment Variable Loading

Fixed the approach to loading environment variables:

```python
@classmethod
def load(cls, env_file: Optional[str] = None) -> "MemCPConfig":
    """Load configuration from environment variables with optional .env file."""
    # Define kwargs for settings initialization
    kwargs = {}
    if env_file is not None:
        kwargs["env_file"] = env_file

    # Let each setting class load its own environment variables
    graphiti = GraphitiAdapterConfig(**kwargs)
    mcp = MCPConfig(**kwargs)
    logging = LoggingConfig(**kwargs)

    return cls(
        graphiti=graphiti,
        mcp=mcp,
        logging=logging
    )
```

### 6. Python 3.10 Compatibility

For Python 3.10 compatibility, we use `tomli` directly instead of conditionally importing:

```python
# For Python 3.10 compatibility, always use tomli
import tomli
```

The `tomli` package is included in the project dependencies for Python 3.10 support.

### 7. Exception Chaining with `raise ... from err`

Improved error handling with proper exception chaining:

```python
try:
    # Code that might raise exceptions
    pass
except tomli.TOMLDecodeError as err:
    raise tomli.TOMLDecodeError(
        msg=f"Invalid TOML format: {err}",
        doc=str(err),
        pos=0
    ) from err
except Exception as err:
    raise ValueError(f"Invalid configuration values: {err}") from err
```

This approach preserves the original exception context for better debugging.

### 8. Helper Method for EpisodeType Conversion

Added a helper method to convert strings to enum values for TOML parsing:

```python
@classmethod
def parse_episode_type(cls, value: Union[str, EpisodeType]) -> EpisodeType:
    """Convert a string to an EpisodeType enum value."""
    if isinstance(value, EpisodeType):
        return value

    try:
        return EpisodeType[value]
    except (KeyError, TypeError):
        # Default to text if invalid
        return EpisodeType.text
```

## Configuration File Structure

### Main Configuration Module (`config/settings.py`)

```python
"""Configuration settings for the MemCP server.

This module provides Pydantic models for configuration management,
supporting both TOML file and environment variable based configuration.
"""

from pathlib import Path
from typing import Any, Optional, Dict, Literal, Union

# For Python 3.10 compatibility, always use tomli
import tomli

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from graphiti_core.nodes import EpisodeType

# Configuration classes implementation
# ... (full implementation in the code artifact)
```

### Environment Variables Example (`.env.example`)

```ini
# Neo4j Database Configuration
# These settings are used to connect to your Neo4j database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=demodemo

# Episode Configuration
DEFAULT_EPISODE_TYPE=text
DEFAULT_SOURCE_DESCRIPTION=memcp
USE_CUSTOM_ENTITIES=false
USE_PARALLEL_RUNTIME=false

# OpenAI API Configuration (Optional)
# Required for LLM operations
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=o3-mini

# MCP Configuration
TRANSPORT=sse  # Can be 'sse' or 'stdio'

# Logging Configuration
LEVEL=INFO
FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
# Other logging settings...
```

## Implementation Steps

1. **Install Dependencies**:

    ```bash
    uv sync --extra dev
    ```

2. **Replace `config/settings.py`** with the new implementation.

3. **Create `.env.example`** file for documentation.

4. **Simplify `config/env.py`** (optional) to only include the `load_environment` function:

    ```python
    """Environment variable loading utility."""

    from pathlib import Path
    from dotenv import load_dotenv

    def load_environment(env_file: str | Path | None = None) -> None:
        """Load environment variables from a .env file."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
    ```

5. **Update Client Code** to use the new factory methods:

    ```python
    # Old approach
    config = MemCPConfig()
    config.update_from_env()

    # New approach
    config = MemCPConfig.load()  # Auto-loads from .env
    # or
    config = MemCPConfig.load(env_file=".env.production")
    # or
    config = MemCPConfig.from_toml("config.toml")
    ```

## Usage Examples

### Loading Configuration from Environment Variables

```python
from memcp.config.settings import MemCPConfig

# Load configuration from environment variables (.env by default)
config = MemCPConfig.load()

# Access configuration values
neo4j_uri = config.graphiti.neo4j_uri
transport = config.mcp.transport
logging_level = config.logging.level
```

### Loading Configuration from TOML

```python
from memcp.config.settings import MemCPConfig

# Load configuration from a TOML file
config = MemCPConfig.from_toml("config.toml")
```

### Configuring MCP Transport

The MCP transport configuration determines how your server communicates with clients:

```python
# In your .env file
TRANSPORT=sse  # or stdio

# In your code
if config.mcp.transport == "sse":
    # Use SSE transport
    transport = SseServerTransport()
else:
    # Use stdio transport
    transport = StdioServerTransport()
```

## Benefits of the New Implementation

1. **Type Safety**: Better type validation with enums and literals
2. **Security**: No hardcoded defaults for sensitive values
3. **Flexibility**: Easy to configure through different methods
4. **Simplicity**: More intuitive environment variable names
5. **Dependency Injection**: Easier to test and mock configurations
6. **Compatibility**: Works with Python 3.10+ while maintaining Pydantic v2 compatibility
7. **Better Error Handling**: Proper exception chaining for easier debugging
8. **Code Quality**: Follows best practices for field validators and type hints

This implementation provides a solid foundation for configuration management in your MemCP project, following best practices in Python and Pydantic v2.
