"""Configuration settings for MemCP."""

from memcp.templates.instructions.mcp_instructions import GraphitiInstructions

import os

from pydantic import BaseModel, Field


class MemCPConfig(BaseModel):
    """Configuration for MemCP.

    Centralizes all configuration parameters for the MemCP server,
    including database connection details and LLM settings.
    """

    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j user")
    neo4j_password: str
    openai_base_url: str | None = None
    model_name: str | None = None
    graph_id: str | None = None
    use_custom_entities: bool = False
    # template_config: TemplateConfig | None = None

    @classmethod
    def from_env(cls) -> "MemCPConfig":
        """Create a configuration instance from environment variables."""
        neo4j_password = os.environ.get("NEO4J_PASSWORD")
        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD must be set")

        return cls(
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=neo4j_password,
            openai_base_url=os.environ.get("OPENAI_BASE_URL"),
            model_name=os.environ.get("MODEL_NAME"),
        )

    @staticmethod
    def get_openai_api_key() -> str:
        """Get the OpenAI API key from environment variables.

        This method doesn't store the API key in the config object,
        instead retrieving it directly from environment variables
        when needed.

        Returns:
            The OpenAI API key if found, None otherwise
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        return api_key


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    mcp_name: str = Field(default="memcp", description="Name of the MCP server")
    mcp_instructions: str = Field(
        default=GraphitiInstructions.DEFAULT_MCP_INSTRUCTIONS,
        description="Instructions for the MCP server",
    )
    transport: str
    host: str = Field(default="127.0.0.1", description="Host address for the server when transport is 'sse'")
    port: int = Field(default=8000, description="Port number for the server when transport is 'sse'")
