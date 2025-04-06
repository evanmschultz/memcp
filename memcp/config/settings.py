"""Configuration settings for MemCP."""

from memcp.templates.instructions.mcp_instructions import GraphitiInstructions

import logging
import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


class SecurityError(ConfigError):
    """Exception raised for security-related configuration errors."""

    pass


class MissingCredentialsError(ConfigError):
    """Exception raised when required credentials are missing."""

    pass


class ServerConfig(BaseSettings):
    """Server configuration settings."""

    model_config = SettingsConfigDict(extra="ignore")

    transport: str = Field("sse", description="Transport type (sse or stdio)")
    host: str = Field("127.0.0.1", description="Host address for the server")
    port: int = Field(8000, description="Port number for the server")


class Neo4jConfig(BaseSettings):
    """Neo4j database connection settings."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=[".env", "../.env", "../../.env"], extra="ignore")

    uri: str = Field("bolt://localhost:7687", description="Neo4j URI")
    user: str = Field("neo4j", description="Neo4j user")
    password: str = Field(..., description="Neo4j password")


class OpenAIConfig(BaseSettings):
    """OpenAI API settings."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=[".env", "../.env", "../../.env"], extra="ignore")

    api_key: str = Field(..., description="OpenAI API key")
    base_url: str | None = Field(None, description="OpenAI API base URL")


class GraphConfig(BaseSettings):
    """Graph configuration settings."""

    model_config = SettingsConfigDict(extra="ignore")

    id: str | None = Field(None, description="Graph ID")
    use_custom_entities: bool = Field(False, description="Enable entity extraction using predefined entities")


class ModelConfig(BaseSettings):
    """Model configuration settings."""

    model_config = SettingsConfigDict(extra="ignore")

    name: str = Field("gpt-4o-mini", description="Model name to use with the LLM client")


class MCPConfig(BaseSettings):
    """Configuration for MCP server."""

    model_config = SettingsConfigDict(env_prefix="MCP_", extra="ignore", env_file=".env", env_file_encoding="utf-8")

    mcp_name: str = Field("memcp", description="Name of the MCP server")
    mcp_instructions: str = Field(
        default=GraphitiInstructions.DEFAULT_MCP_INSTRUCTIONS,
        description="Instructions for the MCP server",
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
        env_file=[".env", "../.env", "../../.env"],  # Try multiple locations
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        cli_parse_args=True,  # Enable CLI parsing
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields for better compatibility
    )

    # Add config_file field for TOML config
    config_file: str | None = Field(None, description="Path to TOML configuration file")

    neo4j: Neo4jConfig = Field(default=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password=""))
    openai: OpenAIConfig = Field(default=OpenAIConfig(api_key="", base_url=None))
    model: ModelConfig = Field(default=ModelConfig(name="gpt-4o-mini"))
    graph: GraphConfig = Field(default=GraphConfig(id=None, use_custom_entities=False))
    server: ServerConfig = Field(default=ServerConfig(transport="sse", host="127.0.0.1", port=8000))
    destroy_graph: bool = Field(False, description="Destroy all graphs")

    def __init__(self, **data: Any) -> None:
        """Initialize the configuration."""
        # Filter sensitive values for logging
        filtered_data = {}
        for k, v in data.items():
            if isinstance(v, dict):
                if k == "openai" and "api_key" in v:
                    v2 = v.copy()
                    v2["api_key"] = "***"
                    filtered_data[k] = v2
                elif k == "neo4j" and "password" in v:
                    v2 = v.copy()
                    v2["password"] = "***"
                    filtered_data[k] = v2
                else:
                    filtered_data[k] = v
            else:
                filtered_data[k] = v

        logger.debug(f"Initializing MemCPConfig with: {filtered_data}")
        super().__init__(**data)

    def model_post_init(self, __context):
        """Post-initialization processing."""
        # Check if we have required values from environment

        logger.debug("Running post-initialization of MemCPConfig")

        # Get OpenAI API key from environment if not provided
        env_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai.api_key and env_api_key:
            logger.debug("Using OpenAI API key from environment variable")
            self.openai.api_key = env_api_key

        # Get Neo4j password from environment if not provided
        env_password = os.environ.get("NEO4J_PASSWORD")
        if not self.neo4j.password and env_password:
            logger.debug("Using Neo4j password from environment variable")
            self.neo4j.password = env_password

        # Validate API key exists
        if not self.openai.api_key:
            raise MissingCredentialsError(
                "OPENAI_API_KEY environment variable is not set. Please set this environment "
                "variable with your OpenAI API key."
            )

        # Validate Neo4j password exists
        if not self.neo4j.password:
            raise MissingCredentialsError(
                "NEO4J_PASSWORD environment variable is not set. Please set this environment "
                "variable with your Neo4j password."
            )

        # If config_file is specified, validate it doesn't contain secrets
        if self.config_file:
            logger.debug(f"Validating configuration file: {self.config_file}")
            self._validate_toml_file_security(self.config_file)

    def _validate_toml_file_security(self, file_path: str):
        """Validate that the TOML file doesn't contain sensitive information.

        Args:
            file_path: Path to the TOML file

        Raises:
            SecurityError: If the TOML file contains sensitive information
        """
        import tomli

        logger = logging.getLogger(__name__)

        try:
            with open(file_path, "rb") as f:
                config = tomli.load(f)

            logger.debug("Validating TOML configuration for security issues")

            # Check for OpenAI API key
            if "openai" in config and isinstance(config["openai"], dict) and "api_key" in config["openai"]:
                if config["openai"]["api_key"] and len(config["openai"]["api_key"]) > 0:
                    raise SecurityError(
                        "SECURITY ERROR: TOML file contains OpenAI API key. "
                        "Please use the OPENAI_API_KEY environment variable instead. "
                        "NEVER store API keys in configuration files."
                    )

            # Check for Neo4j password
            if "neo4j" in config and isinstance(config["neo4j"], dict) and "password" in config["neo4j"]:
                if config["neo4j"]["password"] and len(config["neo4j"]["password"]) > 0:
                    raise SecurityError(
                        "SECURITY ERROR: TOML file contains Neo4j password. "
                        "Please use the NEO4J_PASSWORD environment variable instead. "
                        "NEVER store passwords in configuration files."
                    )

        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {file_path}")
            return
        except tomli.TOMLDecodeError as e:
            logger.error(f"Invalid TOML configuration file: {e}")
            raise ConfigError(f"The configuration file {file_path} is not valid TOML: {e}")
        except ValueError:
            # Re-raise security errors
            raise

    @staticmethod
    def get_openai_api_key() -> str:
        """Get the OpenAI API key from configuration."""
        # We're keeping this method for backward compatibility
        from os import environ

        api_key = environ.get("OPENAI_API_KEY")
        if not api_key:
            raise MissingCredentialsError(
                "OPENAI_API_KEY environment variable is not set. Please set this environment "
                "variable with your OpenAI API key."
            )
        return api_key
