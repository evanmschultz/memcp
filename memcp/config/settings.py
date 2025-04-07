"""Configuration settings for MemCP."""

# TODO: add customizable config path logic
# TODO: find a less hacky way to handle default values from config files and not need to pass each arg in the cli

from memcp.config.sources import DEFAULT_CONFIG_PATH, TomlConfigSettingsSource
from memcp.templates.instructions.mcp_instructions import GraphitiInstructions

import logging
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, CliSuppress, PydanticBaseSettingsSource, SettingsConfigDict
from typing_extensions import Self

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


class Neo4jConfig(BaseSettings):
    """Neo4j database connection configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        case_sensitive=False,
    )

    uri: str = Field(None, description="Neo4j URI. Config defaults to 'bolt://localhost:7687'.")  # type: ignore
    user: str = Field(None, description="Neo4j user. Config defaults to 'neo4j'.")  # type: ignore
    password: CliSuppress[SecretStr] = Field(..., description="Neo4j password", exclude=True)


class OpenAIConfig(BaseSettings):
    """OpenAI API settings."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        case_sensitive=False,
    )

    api_key: CliSuppress[SecretStr] = Field(..., description="OpenAI API key", exclude=True)
    model_name: str = Field(None, description="OpenAI model name. Config defaults to 'gpt-4o-mini'.")  # type: ignore
    base_url: str | None = Field(None, description="OpenAI API base URL")


class GraphConfig(BaseSettings):
    """Graph configuration settings."""

    model_config = SettingsConfigDict(
        extra="allow",
        cli_implicit_flags=True,
    )

    id: str | None = Field(
        None, description="Name of the graph to be used by the DB. If None, a random ID will be generated."
    )
    use_memcp_entities: bool = Field(
        None,
        description="Enable entity extraction using memcp-defined, not graphiti-default entities. "
        "Config defaults to 'False'.",
    )  # type: ignore


class ServerConfig(BaseSettings):
    """Server configuration settings."""

    transport: Literal["sse", "stdio"] = Field(
        "sse", description="Transport type (sse or stdio). Config defaults to 'sse'."
    )
    host: str = Field(None, description="Host address for the server. Config defaults to '127.0.0.1'.")  # type: ignore
    port: int = Field(None, description="Port number for the server. Config defaults to '8000'.")  # type: ignore


class MCPConfig(BaseSettings):
    """MCP server configuration settings."""

    name: str | None = Field(
        None, description="Name of the MCP server to be used by the client. Config defaults to 'memcp'."
    )
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

    Note:
        Update the configuration file in the project directory to ease repetted settings.
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
        cli_implicit_flags=True,
        cli_use_class_docs_for_groups=True,
        toml_file=DEFAULT_CONFIG_PATH,
    )

    # make the cofig path a Path object that validates the input string is a valid path object and the file exists
    # config_path: Path = Field(
    #     DEFAULT_CONFIG_PATH,
    #     description="Path to TOML configuration file. Does not need the .toml extension.",
    # )

    # TODO: fix default factory based type errors and remove the `type: ignore`,
    # code works but pyright complains
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)  # type: ignore
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)  # type: ignore
    graph: GraphConfig = Field(default_factory=GraphConfig)  # type: ignore
    server: ServerConfig = Field(default_factory=ServerConfig)  # type: ignore
    mcp: MCPConfig = Field(default_factory=MCPConfig)  # type: ignore
    destroy_graph: bool = Field(False, description="Destroy all graphs")

    # @field_validator("config_path")
    # @classmethod
    # def validate_config_path(cls, v: Path | str) -> Path:
    #     """Validate that the config file exists and is a valid path."""
    #     # Convert to Path object
    #     path = Path(v) if isinstance(v, str) else v

    #     # Try exact path first
    #     if path.exists() and path.is_file():
    #         return path.resolve()

    #     # If path doesn't exist and doesn't end with .toml, try with .toml
    #     if not str(path).endswith(".toml"):
    #         path_with_toml = Path(f"{path}.toml")
    #         if path_with_toml.exists() and path_with_toml.is_file():
    #             return path_with_toml.resolve()

    #     # Neither path worked, provide helpful error
    #     if not str(path).endswith(".toml"):
    #         raise ValueError(f"Configuration file not found at either {path} or {path}.toml")
    #     else:
    #         raise ValueError(f"Configuration file not found at {path}")

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

        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customise the sources for the settings."""
        # Define the sources and their priority (earlier sources take precedence)
        # The order is important here: CLI args (init_settings) overrides TOML config
        print("Customizing settings sources with TOML file")
        return (
            init_settings,  # First priority - CLI arguments
            TomlConfigSettingsSource(settings_cls),  # Second priority - TOML config file
            env_settings,  # Third priority - Environment variables
            dotenv_settings,  # Fourth priority - .env file
            file_secret_settings,  # Fifth priority - Secrets
        )
