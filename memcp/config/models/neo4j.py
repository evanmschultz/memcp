"""Neo4j configuration models."""

from memcp.config.errors import MissingCredentialsError

import logging

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, CliSuppress, SettingsConfigDict
from typing_extensions import Self

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Neo4jConfig(BaseModel):
    """Neo4j database connection configuration."""

    uri: str
    user: str
    password: SecretStr


class Neo4jConfigBuilder(BaseSettings):
    """Neo4j database connection configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        case_sensitive=False,
    )

    uri: str = Field(None, description="Neo4j URI. Config defaults to 'bolt://localhost:7687'.")  # type: ignore
    user: str = Field(None, description="Neo4j user. Config defaults to 'neo4j'.")  # type: ignore
    password: CliSuppress[SecretStr] = Field(..., description="Neo4j password", exclude=True)

    @model_validator(mode="after")
    def validate_credentials(self) -> Self:
        """Validate that required credentials are provided."""
        # Validate Neo4j password exists
        if not self.password:
            raise MissingCredentialsError("NEO4J_PASSWORD environment variable is not set.")

        return self

    def to_neo4j_config(self) -> Neo4jConfig:
        """Convert to graphiti-core Neo4jConfig."""
        password = SecretStr(self.password.get_secret_value())
        return Neo4jConfig(
            uri=self.uri,
            user=self.user,
            password=password,
        )
