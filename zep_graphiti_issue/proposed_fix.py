# This implementation adds model-aware parameter selection to OpenAI client classes

# 1. First, let's modify the OpenAIClient class in llm_client/openai_client.py:

import logging
import typing
from typing import ClassVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, client: typing.Any = None):
            Initializes the OpenAIClient with the provided configuration, cache setting, and client.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError("Caching is not implemented for OpenAI")

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    def _is_o_series_model(self, model_name: str) -> bool:
        """
        Check if the model is from the OpenAI o-series (o1, o1-mini, o3-mini).

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model is an o-series model, False otherwise.
        """
        if not model_name:
            return False

        # Check if model name starts with 'o' followed by a digit
        return model_name.startswith("o") and len(model_name) > 1 and model_name[1].isdigit()

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == "user":
                openai_messages.append({"role": "user", "content": m.content})
            elif m.role == "system":
                openai_messages.append({"role": "system", "content": m.content})
        try:
            # Get the current model name, defaulting to DEFAULT_MODEL if not set
            model_name = self.model or DEFAULT_MODEL

            # Check if this is an o-series model that requires different parameters
            is_o_series = self._is_o_series_model(model_name)

            # Build request parameters based on model type
            request_params = {
                "model": model_name,
                "messages": openai_messages,
                "response_format": response_model,  # type: ignore
            }

            # Add temperature only for non-o-series models
            if not is_o_series and self.temperature is not None:
                request_params["temperature"] = self.temperature

            # Use the appropriate token parameter based on model type
            if is_o_series:
                request_params["max_completion_tokens"] = max_tokens or self.max_tokens
            else:
                request_params["max_tokens"] = max_tokens or self.max_tokens

            response = await self.client.beta.chat.completions.parse(**request_params)

            response_object = response.choices[0].message

            if response_object.parsed:
                return response_object.parsed.model_dump()
            elif response_object.refusal:
                raise RefusalError(response_object.refusal)
            else:
                raise Exception(f"Invalid response from LLM: {response_object.model_dump()}")
        except openai.LengthFinishReasonError as e:
            raise Exception(f"Output length exceeded max tokens {self.max_tokens}: {e}") from e
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f"Error in generating LLM response: {e}")
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, typing.Any]:
        retry_count = 0
        last_error = None

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(messages, response_model, max_tokens)
                return response
            except (RateLimitError, RefusalError):
                # These errors should not trigger retries
                raise
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                # Let OpenAI's client handle these retries
                raise
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f"Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}")
                    raise

                retry_count += 1

                # Construct a detailed error message for the LLM
                error_context = (
                    f"The previous response attempt was invalid. "
                    f"Error type: {e.__class__.__name__}. "
                    f"Error details: {str(e)}. "
                    f"Please try again with a valid response, ensuring the output matches "
                    f"the expected format and constraints."
                )

                error_message = Message(role="user", content=error_context)
                messages.append(error_message)
                logger.warning(
                    f"Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}"
                )

        # If we somehow get here, raise the last error
        raise last_error or Exception("Max retries exceeded with no specific error")


# 2. Next, let's also update the OpenAIGenericClient class in llm_client/openai_generic_client.py
# with similar changes:

import json
import logging
import typing
from typing import ClassVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIGenericClient(LLMClient):
    """
    OpenAIGenericClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, client: typing.Any = None):
            Initializes the OpenAIClient with the provided configuration, cache setting, and client.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self, config: LLMConfig | None = None, cache: bool = False, client: typing.Any = None
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError("Caching is not implemented for OpenAI")

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    def _is_o_series_model(self, model_name: str) -> bool:
        """
        Check if the model is from the OpenAI o-series (o1, o1-mini, o3-mini).

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model is an o-series model, False otherwise.
        """
        if not model_name:
            return False

        # Check if model name starts with 'o' followed by a digit
        return model_name.startswith("o") and len(model_name) > 1 and model_name[1].isdigit()

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == "user":
                openai_messages.append({"role": "user", "content": m.content})
            elif m.role == "system":
                openai_messages.append({"role": "system", "content": m.content})
        try:
            # Get the current model name, defaulting to DEFAULT_MODEL if not set
            model_name = self.model or DEFAULT_MODEL

            # Build request parameters based on model type
            request_params = {
                "model": model_name,
                "messages": openai_messages,
                "response_format": {"type": "json_object"},
            }

            # Check if this is an o-series model that requires different parameters
            is_o_series = self._is_o_series_model(model_name)

            # Add temperature only for non-o-series models
            if not is_o_series and self.temperature is not None:
                request_params["temperature"] = self.temperature

            # Use the appropriate token parameter based on model type
            if is_o_series:
                request_params["max_completion_tokens"] = max_tokens or self.max_tokens
            else:
                request_params["max_tokens"] = max_tokens or self.max_tokens

            response = await self.client.chat.completions.create(**request_params)

            result = response.choices[0].message.content or ""
            return json.loads(result)
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f"Error in generating LLM response: {e}")
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, typing.Any]:
        retry_count = 0
        last_error = None

        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f"\n\nRespond with a JSON object in the following format:\n\n{serialized_model}"
            )

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens=max_tokens
                )
                return response
            except (RateLimitError, RefusalError):
                # These errors should not trigger retries
                raise
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                # Let OpenAI's client handle these retries
                raise
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f"Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}")
                    raise

                retry_count += 1

                # Construct a detailed error message for the LLM
                error_context = (
                    f"The previous response attempt was invalid. "
                    f"Error type: {e.__class__.__name__}. "
                    f"Error details: {str(e)}. "
                    f"Please try again with a valid response, ensuring the output matches "
                    f"the expected format and constraints."
                )

                error_message = Message(role="user", content=error_context)
                messages.append(error_message)
                logger.warning(
                    f"Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}"
                )

        # If we somehow get here, raise the last error
        raise last_error or Exception("Max retries exceeded with no specific error")


# 3. Finally, we should update the LLMConfig class in llm_client/config.py
# to make it model-aware:

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0


class LLMConfig:
    """
    Configuration class for the Language Learning Model (LLM).

    This class encapsulates the necessary parameters to interact with an LLM API,
    such as OpenAI's GPT models. It stores the API key, model name, and base URL
    for making requests to the LLM service.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the LLMConfig with the provided parameters.

        Args:
                api_key (str): The authentication key for accessing the LLM API.
                                                This is required for making authorized requests.

                model (str, optional): The specific LLM model to use for generating responses.
                                                                Defaults to "gpt-4o-mini", which appears to be a custom model name.
                                                                Common values might include "gpt-3.5-turbo" or "gpt-4".

                base_url (str, optional): The base URL of the LLM API service.
                                                                        Defaults to "https://api.openai.com", which is OpenAI's standard API endpoint.
                                                                        This can be changed if using a different provider or a custom endpoint.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Determine if this is an o-series model for parameter selection
        self.is_o_series_model = False
        if model and model.startswith("o") and len(model) > 1 and model[1].isdigit():
            self.is_o_series_model = True
