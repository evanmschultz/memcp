"""Embeddings provider configuration."""

from memcp.config.errors import MissingCredentialsError

import logging
from typing import Literal

from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.embedder.openai import DEFAULT_EMBEDDING_MODEL as DEFAULT_OPENAI_EMBEDDING_MODEL
from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, CliSuppress, SettingsConfigDict
from typing_extensions import Self

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
