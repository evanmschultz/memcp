"""Configuration settings for MemCP."""

from memcp.templates.instructions.mcp_instructions import GraphitiInstructions

import logging
from pathlib import Path
from typing import Any, Literal

import tomli
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, CliSuppress, SettingsConfigDict
from typing_extensions import Self

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Compute the default config path relative to this module's location
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


class SecurityError(ConfigError):
    """Exception raised for security-related configuration errors."""

    pass


class MissingCredentialsError(ConfigError):
    """Exception raised when required credentials are missing."""

    pass


class Neo4jConfig(BaseSettings):
    """Neo4j database connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        case_sensitive=False,
    )

    uri: str = Field("bolt://localhost:7687", description="Neo4j URI")
    user: str = Field("neo4j", description="Neo4j user")
    password: CliSuppress[SecretStr] = Field(..., description="Neo4j password", exclude=True)


class OpenAIConfig(BaseSettings):
    """OpenAI API settings."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        case_sensitive=False,
    )

    api_key: CliSuppress[SecretStr] = Field(..., description="OpenAI API key", exclude=True)
    model_name: str = Field("gpt-4o-mini", description="OpenAI model name")
    base_url: str | None = Field(None, description="OpenAI API base URL")


class GraphConfig(BaseSettings):
    """Graph configuration settings."""

    id: str | None = Field(None, description="Graph ID")
    use_custom_entities: bool = Field(False, description="Enable entity extraction using predefined entities")


class ServerConfig(BaseSettings):
    """Server configuration settings."""

    transport: Literal["sse", "stdio"] = Field("sse", description="Transport type (sse or stdio)")
    host: str = Field("127.0.0.1", description="Host address for the server")
    port: int = Field(8000, description="Port number for the server")


class MCPConfig(BaseSettings):
    """MCP server configuration settings."""

    name: str = Field("memcp", description="Name of the MCP server to be used by the client")
    instructions: CliSuppress[str] = Field(
        default=GraphitiInstructions.DEFAULT_MCP_INSTRUCTIONS,
        description="Instructions for the MCP server",
    )

    model_config = SettingsConfigDict(
        extra="allow",
    )


class MemCPConfig(BaseSettings):
    """Configuration for MemCP.

    Centralizes all configuration parameters for the MemCP server,
    including database connection details and LLM settings.
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # make the cofig path a Path object that validates the input string is a valid path object and the file exists
    config_path: Path = Field(
        DEFAULT_CONFIG_PATH,
        description="Path to TOML configuration file. Does not need the .toml extension.",
    )

    # TODO: fix default factory based type errors and remove the `type: ignore`
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)  # type: ignore
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)  # type: ignore
    graph: GraphConfig = Field(default_factory=GraphConfig)  # type: ignore
    server: ServerConfig = Field(default_factory=ServerConfig)  # type: ignore
    mcp: MCPConfig = Field(default_factory=MCPConfig)  # type: ignore
    destroy_graph: bool = Field(False, description="Destroy all graphs")

    @field_validator("config_path")
    @classmethod
    def validate_config_path(cls, v: Path | str) -> Path:
        """Validate that the config file exists and is a valid path."""
        # Convert to Path object
        path = Path(v) if isinstance(v, str) else v

        # Try exact path first
        if path.exists() and path.is_file():
            return path.resolve()

        # If path doesn't exist and doesn't end with .toml, try with .toml
        if not str(path).endswith(".toml"):
            path_with_toml = Path(f"{path}.toml")
            if path_with_toml.exists() and path_with_toml.is_file():
                return path_with_toml.resolve()

        # Neither path worked, provide helpful error
        if not str(path).endswith(".toml"):
            raise ValueError(f"Configuration file not found at either {path} or {path}.toml")
        else:
            raise ValueError(f"Configuration file not found at {path}")

    @model_validator(mode="after")
    def validate_credentials(self) -> Self:
        """Validate that required credentials are provided."""
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

        # If config_path is specified, validate it doesn't contain secrets
        if self.config_path:
            logger.debug(f"Validating configuration file: {self.config_path}")
            self._validate_toml_file_security(self.config_path)

        return self

    def _validate_toml_file_security(self, file_path: Path | str) -> None:
        """Validate that the TOML file doesn't contain sensitive information.

        Args:
            file_path: Path to the TOML file (can be string or Path object)

        Raises:
            SecurityError: If the TOML file contains sensitive information
        """
        logger = logging.getLogger(__name__)

        try:
            with open(file_path, "rb") as f:
                config = tomli.load(f)

            logger.debug("Validating TOML configuration for security issues")

            # Check for OpenAI API key
            if "openai" in config:
                openai_config: dict[str, Any] = config["openai"]
                if "api_key" in openai_config:
                    api_key = str(openai_config["api_key"])
                    if api_key and len(api_key) > 0:
                        raise SecurityError(
                            "SECURITY ERROR: TOML file contains OpenAI API key. "
                            "Please use the OPENAI_API_KEY environment variable instead. "
                            "NEVER store API keys in configuration files."
                        )

            # Check for Neo4j password
            if "neo4j" in config:
                neo4j_config: dict[str, Any] = config["neo4j"]
                if "password" in neo4j_config:
                    password = str(neo4j_config["password"])
                    if password and len(password) > 0:
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
            raise ConfigError(f"The configuration file {file_path} is not valid TOML: {e}") from e
        except ValueError:
            # Re-raise security errors
            raise
