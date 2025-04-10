"""LLM client factory for creating LLM clients."""

from memcp.config import MemCPConfig
from memcp.utils import get_logger

# from typing import Any
import graphiti_core.llm_client
import graphiti_core.llm_client.anthropic_client
import graphiti_core.llm_client.openai_client


# TODO: Question, should we use kwargs for the create_client methods?
class LLMClientFactory:
    """Factory for creating LLM clients."""

    _client_classes: dict[str, type[graphiti_core.llm_client.LLMClient]] = {
        "openai": graphiti_core.llm_client.openai_client.OpenAIClient,
        "anthropic": graphiti_core.llm_client.anthropic_client.AnthropicClient,
    }

    def __init__(self, config: MemCPConfig) -> None:
        """Initialize the LLM client factory."""
        self.logger = get_logger(__name__)
        self.config = config

    def create_client(self) -> graphiti_core.llm_client.LLMClient:
        """Create an LLM client for the specified provider.

        Args:
            provider: The LLM provider (default: "openai")
            config: The LLM configuration


        Returns:
            (graphiti_core.llm_client.LLMClient): An instance of the LLM client
        """
        llm_config = self.config.llm.to_graphiti_llm_config()

        # Create and return the client
        client_class = self._client_classes[self.config.llm.provider]
        self.logger.info(f"Creating {self.config.llm.provider.capitalize()} LLM client with model: {llm_config.model}")
        return client_class(config=llm_config)

    # @classmethod
    # def create_openai_client(
    #     cls,
    #     api_key: str,
    #     model: str | None = None,
    #     **kwargs: Any,  # noqa: ANN401
    # ) -> LLMClient:
    #     """Create an OpenAI LLM client.

    #     Args:
    #         api_key: API key for the OpenAI service
    #         model: Model name to use
    #         **kwargs: Additional OpenAI-specific configuration

    #     Returns:
    #         An instance of the OpenAI LLM client
    #     """
    #     factory = cls()
    #     return factory.create_client(provider="openai", api_key=api_key, model=model, **kwargs)
