import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from graphiti_core.llm_client import OpenAIClient, OpenAIGenericClient, LLMConfig
from graphiti_core.prompts.models import Message


@pytest.mark.asyncio
async def test_openai_client_non_o_series_model():
    """Test that OpenAIClient uses max_tokens for non-o-series models."""
    # Mock the AsyncOpenAI client
    mock_client = MagicMock()
    mock_client.beta = MagicMock()
    mock_client.beta.chat = MagicMock()
    mock_client.beta.chat.completions = MagicMock()

    # Create a mock response
    mock_message = MagicMock()
    mock_message.parsed = {"result": "success"}
    mock_message.refusal = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Set up the parse method to return our mock response
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

    # Create an OpenAIClient with a non-o-series model
    config = LLMConfig(model="gpt-4", temperature=0.7, max_tokens=100)
    client = OpenAIClient(config=config, client=mock_client)

    # Generate a response
    messages = [Message(role="user", content="test")]
    await client.generate_response(messages)

    # Check that parse was called with max_tokens
    call_args = mock_client.beta.chat.completions.parse.call_args[1]
    assert "max_tokens" in call_args
    assert "max_completion_tokens" not in call_args
    assert call_args["temperature"] == 0.7


@pytest.mark.asyncio
async def test_openai_client_o_series_model():
    """Test that OpenAIClient uses max_completion_tokens for o-series models."""
    # Mock the AsyncOpenAI client
    mock_client = MagicMock()
    mock_client.beta = MagicMock()
    mock_client.beta.chat = MagicMock()
    mock_client.beta.chat.completions = MagicMock()

    # Create a mock response
    mock_message = MagicMock()
    mock_message.parsed = {"result": "success"}
    mock_message.refusal = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Set up the parse method to return our mock response
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

    # Create an OpenAIClient with an o-series model
    config = LLMConfig(model="o3-mini", temperature=0.7, max_tokens=100)
    client = OpenAIClient(config=config, client=mock_client)

    # Generate a response
    messages = [Message(role="user", content="test")]
    await client.generate_response(messages)

    # Check that parse was called with max_completion_tokens
    call_args = mock_client.beta.chat.completions.parse.call_args[1]
    assert "max_completion_tokens" in call_args
    assert "max_tokens" not in call_args
    assert "temperature" not in call_args  # Temperature should be excluded for o-series models


@pytest.mark.asyncio
async def test_openai_generic_client_non_o_series_model():
    """Test that OpenAIGenericClient uses max_tokens for non-o-series models."""
    # Mock the AsyncOpenAI client
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()

    # Create a mock response
    mock_message = MagicMock()
    mock_message.content = json.dumps({"result": "success"})

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Set up the create method to return our mock response
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Create an OpenAIGenericClient with a non-o-series model
    config = LLMConfig(model="gpt-4", temperature=0.7, max_tokens=100)
    client = OpenAIGenericClient(config=config, client=mock_client)

    # Generate a response
    messages = [Message(role="user", content="test")]
    await client.generate_response(messages)

    # Check that create was called with max_tokens
    call_args = mock_client.chat.completions.create.call_args[1]
    assert "max_tokens" in call_args
    assert "max_completion_tokens" not in call_args
    assert call_args["temperature"] == 0.7


@pytest.mark.asyncio
async def test_openai_generic_client_o_series_model():
    """Test that OpenAIGenericClient uses max_completion_tokens for o-series models."""
    # Mock the AsyncOpenAI client
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()

    # Create a mock response
    mock_message = MagicMock()
    mock_message.content = json.dumps({"result": "success"})

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Set up the create method to return our mock response
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Create an OpenAIGenericClient with an o-series model
    config = LLMConfig(model="o3-mini", temperature=0.7, max_tokens=100)
    client = OpenAIGenericClient(config=config, client=mock_client)

    # Generate a response
    messages = [Message(role="user", content="test")]
    await client.generate_response(messages)

    # Check that create was called with max_completion_tokens
    call_args = mock_client.chat.completions.create.call_args[1]
    assert "max_completion_tokens" in call_args
    assert "max_tokens" not in call_args
    assert "temperature" not in call_args  # Temperature should be excluded for o-series models


def test_is_o_series_model():
    """Test the _is_o_series_model method."""
    client = OpenAIClient()

    # Test cases that should return True
    assert client._is_o_series_model("o1") is True
    assert client._is_o_series_model("o1-mini") is True
    assert client._is_o_series_model("o3-mini") is True

    # Test cases that should return False
    assert client._is_o_series_model("gpt-4") is False
    assert client._is_o_series_model("gpt-3.5-turbo") is False
    assert client._is_o_series_model("") is False
    assert client._is_o_series_model(None) is False
    assert client._is_o_series_model("other-o3-mini") is False  # Must start with 'o'


def test_llm_config_o_series_detection():
    """Test that LLMConfig correctly detects o-series models."""
    # Test o-series models
    config = LLMConfig(model="o1")
    assert config.is_o_series_model is True

    config = LLMConfig(model="o1-mini")
    assert config.is_o_series_model is True

    config = LLMConfig(model="o3-mini")
    assert config.is_o_series_model is True

    # Test non-o-series models
    config = LLMConfig(model="gpt-4")
    assert config.is_o_series_model is False

    config = LLMConfig(model="gpt-3.5-turbo")
    assert config.is_o_series_model is False

    config = LLMConfig(model=None)
    assert config.is_o_series_model is False
