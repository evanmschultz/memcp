"""Configuration manager for MemCP."""

from memcp.config.providers import (
    ArgsConfigProvider,
    ConfigProvider,
    EnvConfigProvider,
    TomlConfigProvider,
)
from memcp.config.settings import MCPConfig, MemCPConfig
from memcp.utils import get_logger

import uuid
from typing import Any


class ConfigManager:
    """Manager for loading and merging configuration from multiple sources."""

    def __init__(self) -> None:
        """Initialize the configuration manager."""
        self.logger = get_logger(__name__)
        self.providers: list[ConfigProvider] = []
        self.merged_config: dict[str, Any] = {}

    def add_provider(self, provider: ConfigProvider) -> None:
        """Add a configuration provider.

        Providers are used in the order they are added, with later providers
        overriding values from earlier ones.

        Args:
            provider: The configuration provider to add
        """
        self.providers.append(provider)

    def load_config(self) -> dict[str, Any]:
        """Load and merge configuration from all providers.

        Returns:
            The merged configuration
        """
        self.merged_config = {}

        # Process providers in order (later ones override earlier ones)
        for provider in self.providers:
            provider_config = provider.get_config()
            self._deep_merge(self.merged_config, provider_config)

        # Set defaults for any missing values
        self._set_defaults()

        return self.merged_config

    def _deep_merge(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        """Recursively merge source dictionary into target.

        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recursively merge dictionaries
                self._deep_merge(target[key], value)
            elif value is not None:
                # Override or add value
                target[key] = value

    def _set_defaults(self) -> None:
        """Set default values for missing configuration."""
        # Server defaults
        server_config = self.merged_config.setdefault("server", {})
        server_config.setdefault("transport", "sse")
        server_config.setdefault("host", "127.0.0.1")
        server_config.setdefault("port", 8000)

        # Neo4j defaults
        neo4j_config = self.merged_config.setdefault("neo4j", {})
        neo4j_config.setdefault("uri", "bolt://localhost:7687")
        neo4j_config.setdefault("user", "neo4j")

        # Model defaults
        model_config = self.merged_config.setdefault("model", {})
        model_config.setdefault("name", "gpt-4o-mini")

        # Graph defaults
        graph_config = self.merged_config.setdefault("graph", {})
        if not graph_config.get("id"):
            graph_config["id"] = f"graph_{uuid.uuid4().hex[:8]}"
        graph_config.setdefault("use_custom_entities", False)

        # Others
        self.merged_config.setdefault("destroy_graph", False)

    def create_memcp_config(self) -> MemCPConfig:
        """Create a MemCPConfig from the merged configuration.

        Returns:
            Configured MemCPConfig instance
        """
        neo4j_password = self.merged_config["neo4j"].get("password")
        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD must be set")

        graph_id = self.merged_config["graph"].get("id")
        if graph_id:
            self.logger.info(f"Using graph_id: {graph_id}")
        else:
            graph_id = f"graph_{uuid.uuid4().hex[:8]}"
            self.logger.info(f"Generated random graph_id: {graph_id}")

        use_custom_entities = self.merged_config["graph"].get("use_custom_entities", False)
        if use_custom_entities:
            self.logger.info("Entity extraction enabled using predefined MEMCP_ENTITIES")
        else:
            self.logger.info("Entity extraction disabled (no custom entities will be used)")

        return MemCPConfig(
            neo4j_uri=self.merged_config["neo4j"]["uri"],
            neo4j_user=self.merged_config["neo4j"]["user"],
            neo4j_password=neo4j_password,
            openai_base_url=self.merged_config.get("openai", {}).get("base_url"),
            model_name=self.merged_config["model"]["name"],
            graph_id=graph_id,
            use_custom_entities=use_custom_entities,
        )

    def create_mcp_config(self) -> MCPConfig:
        """Create an MCPConfig from the merged configuration.

        Returns:
            Configured MCPConfig instance
        """
        return MCPConfig(
            transport=self.merged_config["server"]["transport"],
            host=self.merged_config["server"]["host"],
            port=self.merged_config["server"]["port"],
        )

    def should_destroy_graph(self) -> bool:
        """Check if the graph should be destroyed.

        Returns:
            True if the graph should be destroyed, False otherwise
        """
        return self.merged_config.get("destroy_graph", False)

    @classmethod
    def create_default(cls, args: Any | None = None) -> "ConfigManager":
        """Create a ConfigManager with default providers.

        Args:
            args: Optional parsed command-line arguments

        Returns:
            Configured ConfigManager instance
        """
        manager = cls()

        # Add providers in order of precedence (lowest to highest)
        manager.add_provider(EnvConfigProvider())

        # Check if args has config attribute and it's set
        config_path = getattr(args, "config", None) if args else None
        if config_path:
            manager.add_provider(TomlConfigProvider(config_path))
        else:
            # Try default config location
            manager.add_provider(TomlConfigProvider())

        # Command line args have highest precedence
        manager.add_provider(ArgsConfigProvider(args))

        # Load the configuration
        manager.load_config()

        return manager
