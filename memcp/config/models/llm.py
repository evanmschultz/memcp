"""Configuration settings for MemCP."""

from memcp.config.errors import MissingCredentialsError
from memcp.config.utils import get_model_name, get_model_name_description, is_anthropic_available

import logging
from typing import Literal, cast

import graphiti_core.llm_client.config
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, CliSuppress, SettingsConfigDict
from typing_extensions import Self

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
