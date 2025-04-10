"""MCP configuration models."""

from memcp.templates import GraphitiInstructions

import logging

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliSuppress, SettingsConfigDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
