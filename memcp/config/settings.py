"""Configuration settings for MemCP."""

import os

from pydantic import BaseModel, Field


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client.

    Centralizes all configuration parameters for the Graphiti client,
    including database connection details and LLM settings.
    """

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    model_name: str | None = None
    graph_id: str | None = None
    use_custom_entities: bool = False

    @classmethod
    def from_env(cls) -> "GraphitiConfig":
        """Create a configuration instance from environment variables."""
        neo4j_password = os.environ.get("NEO4J_PASSWORD")
        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD must be set")

        return cls(
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=neo4j_password,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_base_url=os.environ.get("OPENAI_BASE_URL"),
            model_name=os.environ.get("MODEL_NAME"),
        )


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str
    host: str = Field(
        default="127.0.0.1", description="Host address for the server when transport is 'sse'"
    )
    port: int = Field(
        default=8000, description="Port number for the server when transport is 'sse'"
    )
