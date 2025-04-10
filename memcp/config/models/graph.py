"""Graph configuration models."""

import logging

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GraphConfig(BaseModel):
    """Graph configuration."""

    id: str | None
    use_memcp_entities: bool


class GraphConfigBuilder(BaseSettings):
    """Graph configuration settings."""

    model_config = SettingsConfigDict(
        extra="allow",
        cli_implicit_flags=True,
    )

    id: str | None = Field(
        None, description="Name of the graph to be used by the DB. If None, a random ID will be generated."
    )
    use_memcp_entities: bool = Field(
        None,
        description="Enable entity extraction using memcp-defined, not graphiti-default entities. "
        "Config defaults to 'False'.",
    )  # type: ignore

    def to_graph_config(self) -> GraphConfig:
        """Convert to graphiti-core GraphConfig."""
        return GraphConfig(
            id=self.id,
            use_memcp_entities=self.use_memcp_entities,
        )
