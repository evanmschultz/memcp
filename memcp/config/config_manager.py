"""Configuration manager for MemCP."""

from memcp.config.providers import TomlConfigProvider
from memcp.config.settings import MCPConfig, MemCPConfig
from memcp.utils import get_logger

import os
from pathlib import Path
from typing import Any


class ConfigManager:
    """Simplified manager for loading configuration."""

    def __init__(self, toml_path: str | None = None):
        """Initialize the configuration manager.

        Args:
            toml_path: Optional path to TOML config file
        """
        self.logger = get_logger(__name__)
        self.toml_path = toml_path
        self._memcp_config = None
        self._mcp_config = None
        self._toml_config = None

    @property
    def toml_config(self) -> dict[str, Any]:
        """Get configuration from TOML file."""
        if self._toml_config is None:
            # Priority:
            # 1. Explicitly provided path (CLI arg)
            # 2. Current working directory
            # 3. Project directory

            toml_paths = []

            # 1. Explicitly provided path
            if self.toml_path:
                toml_paths.append(self.toml_path)

            # 2. Current working directory
            toml_paths.append(os.path.join(os.getcwd(), "config.toml"))

            # 3. Project directory
            project_path = Path(__file__).parent.parent.parent / "config.toml"
            toml_paths.append(str(project_path))

            # Try each path in order
            for path in toml_paths:
                self.logger.debug(f"Checking for TOML config at: {path}")
                provider = TomlConfigProvider(path)
                config = provider.get_config()
                if config:
                    self.logger.info(f"Loaded TOML config from: {path}")
                    self._toml_config = config
                    break

            # If no config found, use empty dict
            if self._toml_config is None:
                self._toml_config = {}

        return self._toml_config

    def create_memcp_config(self) -> MemCPConfig:
        """Create a MemCPConfig instance.

        Returns:
            Configured MemCPConfig instance
        """
        if self._memcp_config is None:
            # Initialize with TOML config first
            toml_dict = self.toml_config

            # Create config with CLI args and env vars (handled by pydantic-settings)
            self._memcp_config = MemCPConfig.model_validate(toml_dict)

            # If toml_path was provided, set it in the config
            if self.toml_path:
                self._memcp_config.config_file = self.toml_path

            # Log details
            if self._memcp_config.graph.id:
                self.logger.info(f"Using graph_id: {self._memcp_config.graph.id}")
            else:
                import uuid

                # Generate a new graph ID if not provided
                self._memcp_config.graph.id = f"graph_{uuid.uuid4().hex[:8]}"
                self.logger.info(f"Generated random graph_id: {self._memcp_config.graph.id}")

            if self._memcp_config.graph.use_custom_entities:
                self.logger.info("Entity extraction enabled using predefined MEMCP_ENTITIES")
            else:
                self.logger.info("Entity extraction disabled (no custom entities will be used)")

        return self._memcp_config

    def create_mcp_config(self) -> MCPConfig:
        """Create an MCPConfig from the merged configuration.

        Returns:
            Configured MCPConfig instance
        """
        if self._mcp_config is None:
            # Extract server config from MemCPConfig
            memcp_config = self.create_memcp_config()

            # Create MCP config
            self._mcp_config = MCPConfig(
                mcp_name="memcp",
                transport=memcp_config.server.transport,
                host=memcp_config.server.host,
                port=memcp_config.server.port,
            )

        return self._mcp_config

    def should_destroy_graph(self) -> bool:
        """Check if the graph should be destroyed.

        Returns:
            True if the graph should be destroyed, False otherwise
        """
        return self.create_memcp_config().destroy_graph

    @classmethod
    def create_default(cls, config_path: str | None = None) -> "ConfigManager":
        """Create a ConfigManager with default settings.

        Args:
            config_path: Optional path to TOML config file

        Returns:
            Configured ConfigManager instance
        """
        return cls(toml_path=config_path)
