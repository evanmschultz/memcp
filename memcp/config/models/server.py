"""Server configuration models."""

import logging
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
