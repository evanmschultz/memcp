"""LLM client factory for creating LLM clients."""

from memcp.utils import get_logger

from typing import Any

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient


# TODO: Question, should we use kwargs for the create_client methods?
class LLMClientFactory:
    """Factory for creating LLM clients."""

    _client_classes: dict[str, type[LLMClient]] = {
        "openai": OpenAIClient,
        # Add more providers as they become available
        # "anthropic": AnthropicClient,
        # "groq": GroqClient,
    }

    def __init__(self) -> None:
        """Initialize the LLM client factory."""
        self.logger = get_logger(__name__)

    def create_client(
        self,
        provider: str = "openai",
        api_key: str = "",
        model: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> LLMClient:
        """Create an LLM client for the specified provider.

        Args:
            provider: The LLM provider (default: "openai")
            api_key: API key for the LLM provider
            model: Model name to use
            **kwargs: Additional provider-specific configuration

        Returns:
            An instance of the LLM client

        Raises:
            ValueError: If the provider is not supported or required config is missing
        """
        if provider not in self._client_classes:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        if not api_key:
            raise ValueError(f"API key is required for {provider}")

        # Create base config
        llm_config = LLMConfig(api_key=api_key)

        # Set model if provided
        if model:
            llm_config.model = model

        # Update with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(llm_config, key):
                setattr(llm_config, key, value)

        # Create and return the client
        client_class = self._client_classes[provider]
        self.logger.info(f"Creating {provider} LLM client with model: {llm_config.model}")
        return client_class(config=llm_config)

    @classmethod
    def create_openai_client(
        cls,
        api_key: str,
        model: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> LLMClient:
        """Create an OpenAI LLM client.

        Args:
            api_key: API key for the OpenAI service
            model: Model name to use
            **kwargs: Additional OpenAI-specific configuration

        Returns:
            An instance of the OpenAI LLM client
        """
        factory = cls()
        return factory.create_client(provider="openai", api_key=api_key, model=model, **kwargs)
