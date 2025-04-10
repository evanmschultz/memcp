"""Utility functions for the MemCP config package."""

import logging
from typing import Literal

from graphiti_core.llm_client.openai_client import DEFAULT_MODEL as DEFAULT_OPENAI_MODEL

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
