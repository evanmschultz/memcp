"""Configuration errors."""

from pathlib import Path


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


class MissingCredentialsError(ConfigError):
    """Exception raised when required credentials are missing."""

    pass


class ConfigMissingError(ConfigError):
    """Exception raised when a configuration file is missing."""

    def __init__(self, config_path: Path) -> None:
        """Initialize the ConfigMissingError.

        Args:
            config_path: The path to the configuration file
        """
        self.config_path = config_path
        super().__init__(f"Configuration file not found at: {config_path}")
