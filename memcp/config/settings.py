"""Configuration settings for MemCP."""

# TODO: add customizable config path logic
# TODO: find a less hacky way to handle default values from config files and not need to pass each arg in the cli

from memcp.config.config_errors import MissingCredentialsError
from memcp.config.sources import DEFAULT_CONFIG_PATH, TomlConfigSettingsSource
from memcp.templates import GraphitiInstructions

import logging
from typing import Literal, cast

import graphiti_core.llm_client.config
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.embedder.openai import DEFAULT_EMBEDDING_MODEL as DEFAULT_OPENAI_EMBEDDING_MODEL
from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from graphiti_core.llm_client.openai_client import DEFAULT_MODEL as DEFAULT_OPENAI_MODEL
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, CliSuppress, PydanticBaseSettingsSource, SettingsConfigDict
from typing_extensions import Self

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def is_anthropic_available() -> bool:
    """Check if anthropic is available.

    Used to confirm that the Anthropic extension to MemCP is installed before allowing Anthropic usage.
    """
    try:
        import anthropic  # type: ignore # noqa: F401

        return True
    except ImportError:
        return False


def get_model_name(provider: Literal["openai", "anthropic"]) -> str:
    """Get the model name for the current provider.

    Returns:
        str: The model name.
    """
    if is_anthropic_available() and provider == "anthropic":
        from graphiti_core.llm_client.anthropic_client import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL

        return ANTHROPIC_DEFAULT_MODEL
    elif provider == "openai":
        return DEFAULT_OPENAI_MODEL
    else:
        raise ValueError(f"Invalid provider: {provider}")


def get_model_name_description() -> str:
    """Get the model name description for the current provider.

    Conditionally defines the model name description based on the availability of the anthropic provider.

    Returns:
        str: The model name description.
    """
    description = f"Model name to use for completions. Defaults to {DEFAULT_OPENAI_MODEL} for OpenAI."
    if is_anthropic_available():
        from graphiti_core.llm_client.anthropic_client import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL

        description += f" Defaults to {ANTHROPIC_DEFAULT_MODEL} for Anthropic."
    return description


class Neo4jConfig(BaseModel):
    """Neo4j database connection configuration."""

    uri: str
    user: str
    password: SecretStr


class Neo4jConfigBuilder(BaseSettings):
    """Neo4j database connection configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        case_sensitive=False,
    )

    uri: str = Field(None, description="Neo4j URI. Config defaults to 'bolt://localhost:7687'.")  # type: ignore
    user: str = Field(None, description="Neo4j user. Config defaults to 'neo4j'.")  # type: ignore
    password: CliSuppress[SecretStr] = Field(..., description="Neo4j password", exclude=True)

    @model_validator(mode="after")
    def validate_credentials(self) -> Self:
        """Validate that required credentials are provided."""
        # Validate Neo4j password exists
        if not self.password:
            raise MissingCredentialsError("NEO4J_PASSWORD environment variable is not set.")

        return self

    def to_neo4j_config(self) -> Neo4jConfig:
        """Convert to graphiti-core Neo4jConfig."""
        password = SecretStr(self.password.get_secret_value())
        return Neo4jConfig(
            uri=self.uri,
            user=self.user,
            password=password,
        )


class EmbeddingsConfig(BaseModel):
    """Embeddings provider configuration."""

    embeddings_provider: Literal["openai"]
    api_key: SecretStr
    embeddings_model_name: str
    embeddings_dim: int
    embeddings_base_url: str | None

    def to_graphiti_embeddings_config(self) -> OpenAIEmbedderConfig:
        """Convert to graphiti-core LLMConfig for embeddings."""
        return OpenAIEmbedderConfig(
            api_key=self.api_key.get_secret_value(),
            embedding_model=self.embeddings_model_name,
            embedding_dim=self.embeddings_dim,
            base_url=self.embeddings_base_url,
        )


class EmbeddingsConfigBuilder(BaseSettings):
    """Embeddings provider configuration.

    Notes:
        - Currently only OpenAI is supported due to graphiti-core limitations.
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="OPENAI_",
        extra="allow",
    )

    embeddings_provider: CliSuppress[Literal["openai"]] = Field(
        "openai", description="LLM provider to use for embeddings (currently only OpenAI supported)"
    )
    api_key: CliSuppress[SecretStr] = Field(..., description="API key for the embeddings provider")
    embeddings_model_name: CliSuppress[str] = Field(
        DEFAULT_OPENAI_EMBEDDING_MODEL,
        description="Model name to use for embeddings. Defaults to provider-specific default.",
    )
    embeddings_dim: CliSuppress[int] = Field(
        EMBEDDING_DIM, description="Dimension of the embeddings. Defaults to provider-specific default."
    )
    embeddings_base_url: CliSuppress[str | None] = Field(None, description="Base URL for the embeddings API")

    @model_validator(mode="after")
    def validate_credentials(self) -> Self:
        """Validate that required credentials are provided."""
        if not self.api_key:
            raise MissingCredentialsError(
                f"{self.embeddings_provider.upper()}_API_KEY environment variable is not set for embeddings."
            )
        return self

    def to_embeddings_config(self) -> EmbeddingsConfig:
        """Convert to graphiti-core EmbeddingsConfig."""
        return EmbeddingsConfig(
            embeddings_provider=self.embeddings_provider,
            api_key=self.api_key,
            embeddings_model_name=self.embeddings_model_name,
            embeddings_dim=self.embeddings_dim,
            embeddings_base_url=self.embeddings_base_url,
        )


class AnthropicConfigBuilder(BaseSettings):
    """Anthropic configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ANTHROPIC_",
        case_sensitive=False,
    )

    api_key: CliSuppress[SecretStr | None] = Field(None, description="API key for the anthropic provider")


class OpenAIConfigBuilder(BaseSettings):
    """OpenAI configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        case_sensitive=False,
    )

    api_key: CliSuppress[SecretStr] = Field(..., description="API key for the openai provider")


class LLMProviderConfig(BaseModel):
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic"]
    api_key: SecretStr
    model_name: str
    base_url: str | None
    temperature: float
    max_tokens: int

    def to_graphiti_llm_config(self) -> graphiti_core.llm_client.config.LLMConfig:
        """Convert to graphiti-core LLMConfig."""
        return graphiti_core.llm_client.LLMConfig(
            api_key=self.api_key.get_secret_value(),
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class LLMProviderConfigBuilder(BaseSettings):
    """Base configuration for LLM providers."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
    )

    provider: Literal["openai", "anthropic"] = Field(
        "openai",  # Default to OpenAI
        description="LLM provider to use for completions (openai or anthropic)",
    )

    openai_config: CliSuppress[OpenAIConfigBuilder] = Field(default_factory=OpenAIConfigBuilder)  # type: ignore
    anthropic_config: CliSuppress[AnthropicConfigBuilder] = Field(default_factory=AnthropicConfigBuilder)  # type: ignore
    api_key: CliSuppress[SecretStr | None] = Field(None, description="API key for the completion provider")

    model_name: str | None = Field(
        None,
        description=get_model_name_description(),
    )

    base_url: CliSuppress[str | None] = Field(None, description="Base URL for the completion API")

    temperature: CliSuppress[float] = Field(DEFAULT_TEMPERATURE, description="Temperature for generation")
    max_tokens: CliSuppress[int] = Field(DEFAULT_MAX_TOKENS, description="Maximum tokens to generate")

    @model_validator(mode="before")
    @classmethod
    def set_env_prefix(cls, data: dict[str, str]) -> dict[str, str]:
        """Set the environment prefix based on the provider."""
        provider = data.get("provider", "openai")
        cls.model_config["env_prefix"] = f"{provider.upper()}_"
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_provider(cls, data: dict[str, str]) -> dict[str, str]:
        """Validate that the provider is available."""
        provider = data.get("provider", "openai")
        if provider == "anthropic" and not is_anthropic_available():
            raise ImportError(
                'The anthropic extra is required for Anthropic support. Install it with: uv add "memcp\\[anthropic]"'
            )
        return data

    @model_validator(mode="after")
    def validate_credentials(self) -> Self:
        """Validate that required credentials are provided."""
        if self.provider == "openai":
            if not self.openai_config.api_key:
                raise MissingCredentialsError("OPENAI_API_KEY environment variable is not set for llm provider.")

            self.api_key = self.openai_config.api_key
        elif self.provider == "anthropic":
            if not self.anthropic_config.api_key:
                raise MissingCredentialsError("ANTHROPIC_API_KEY environment variable is not set for llm provider.")

            self.api_key = self.anthropic_config.api_key
        return self

    def to_llm_provider_config(self) -> LLMProviderConfig:
        """Convert to graphiti-core LLMProviderConfig."""
        api_key = cast(SecretStr, self.api_key)
        if not self.model_name:
            self.model_name = get_model_name(self.provider)

        return LLMProviderConfig(
            provider=self.provider,
            api_key=api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class GraphConfig(BaseModel):
    """Graph configuration."""

    id: str | None
    use_memcp_entities: bool


class GraphConfigBuilder(BaseSettings):
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

    def to_graph_config(self) -> GraphConfig:
        """Convert to graphiti-core GraphConfig."""
        return GraphConfig(
            id=self.id,
            use_memcp_entities=self.use_memcp_entities,
        )


class ServerConfig(BaseModel):
    """Server configuration."""

    transport: Literal["sse", "stdio"]
    host: str
    port: int


class ServerConfigBuilder(BaseSettings):
    """Server configuration settings."""

    transport: Literal["sse", "stdio"] = Field(
        "sse", description="Transport type (sse or stdio). Config defaults to 'sse'."
    )
    host: str = Field(None, description="Host address for the server. Config defaults to '127.0.0.1'.")  # type: ignore
    port: int = Field(None, description="Port number for the server. Config defaults to '8000'.")  # type: ignore

    def to_server_config(self) -> ServerConfig:
        """Convert to graphiti-core ServerConfig."""
        return ServerConfig(
            transport=self.transport,
            host=self.host,
            port=self.port,
        )


class MCPConfig(BaseModel):
    """MCP server configuration."""

    name: str | None
    instructions: str


class MCPConfigBuilder(BaseSettings):
    """MCP server configuration settings."""

    model_config = SettingsConfigDict(
        extra="allow",
    )

    name: str | None = Field(
        None, description="Name of the MCP server to be used by the client. Config defaults to 'memcp'."
    )
    instructions: CliSuppress[str] = Field(
        default=GraphitiInstructions.DEFAULT_MCP_INSTRUCTIONS,
        description="Instructions for the MCP server",
    )

    def to_mcp_config(self) -> MCPConfig:
        """Convert to graphiti-core MCPConfig."""
        return MCPConfig(
            name=self.name,
            instructions=self.instructions,
        )


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
