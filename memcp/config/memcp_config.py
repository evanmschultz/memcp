"""Configuration settings for MemCP."""

# TODO: add customizable config path logic
# TODO: find a less hacky way to handle default values from config files and not need to pass each arg in the cli

from memcp.config.models import (
    EmbeddingsConfig,
    EmbeddingsConfigBuilder,
    GraphConfig,
    GraphConfigBuilder,
    LLMProviderConfig,
    LLMProviderConfigBuilder,
    MCPConfig,
    MCPConfigBuilder,
    Neo4jConfig,
    Neo4jConfigBuilder,
    ServerConfig,
    ServerConfigBuilder,
)
from memcp.config.sources import DEFAULT_CONFIG_PATH, TomlConfigSettingsSource

import logging

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliSuppress, PydanticBaseSettingsSource, SettingsConfigDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MemCPConfig(BaseModel):
    """MemCP configuration."""

    neo4j: Neo4jConfig
    llm: LLMProviderConfig
    graph: GraphConfig
    server: ServerConfig
    mcp: MCPConfig
    destroy_graph: bool
    embeddings: EmbeddingsConfig


class MemCPConfigBuilder(BaseSettings):
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
        env_file=".env",
        env_file_encoding="utf-8",
    )

    neo4j: Neo4jConfigBuilder = Field(default_factory=Neo4jConfigBuilder)  # type: ignore
    llm: LLMProviderConfigBuilder = Field(default_factory=LLMProviderConfigBuilder)  # type: ignore
    graph: GraphConfigBuilder = Field(default_factory=GraphConfigBuilder)  # type: ignore
    server: ServerConfigBuilder = Field(default_factory=ServerConfigBuilder)  # type: ignore
    mcp: MCPConfigBuilder = Field(default_factory=MCPConfigBuilder)  # type: ignore
    destroy_graph: bool = Field(False, description="Destroy all graphs")
    embeddings: CliSuppress[EmbeddingsConfigBuilder] = Field(default_factory=EmbeddingsConfigBuilder)  # type: ignore

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
        return (
            init_settings,  # First priority - CLI arguments
            TomlConfigSettingsSource(settings_cls),  # Second priority - TOML config file
            env_settings,  # Third priority - Environment variables
            dotenv_settings,  # Fourth priority - .env file
            file_secret_settings,  # Fifth priority - Secrets
        )

    def to_memcp_config(self) -> MemCPConfig:
        """Convert to graphiti-core MemCPConfig."""
        return MemCPConfig(
            neo4j=self.neo4j.to_neo4j_config(),
            llm=self.llm.to_llm_provider_config(),
            graph=self.graph.to_graph_config(),
            server=self.server.to_server_config(),
            mcp=self.mcp.to_mcp_config(),
            destroy_graph=self.destroy_graph,
            embeddings=self.embeddings.to_embeddings_config(),
        )
