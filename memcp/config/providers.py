"""Configuration providers for different config sources."""

from typing import Any, Protocol, runtime_checkable

import tomli


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol defining a configuration provider interface."""

    def get_config(self) -> dict[str, Any]:
        """Get configuration from this provider."""
        ...


class TomlConfigProvider:
    """Provider for TOML file configuration."""

    def __init__(self, config_path: str = "config.toml") -> None:
        """Initialize with path to TOML config file."""
        self.config_path = config_path

    def get_config(self) -> dict[str, Any]:
        """Extract configuration from TOML file."""
        try:
            with open(self.config_path, "rb") as f:
                return tomli.load(f)
        except (FileNotFoundError, tomli.TOMLDecodeError):
            # Return empty config if file doesn't exist or is invalid
            return {}
