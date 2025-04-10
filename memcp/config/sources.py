"""Settings source for loading configuration values from a TOML file."""

from memcp.config.config_errors import ConfigMissingError

from pathlib import Path
from typing import Any

import tomli
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

# TODO: Add a custom config path logic


# Compute the default config path relative to this module's location (custom config path not yet supported)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


class TomlConfigSettingsSource(PydanticBaseSettingsSource):
    """Settings source for loading values from a TOML file."""

    def __init__(self, settings_cls: type[BaseSettings], toml_file: Path | None = None) -> None:
        """Initialize the TOML settings source.

        Args:
            settings_cls: The settings class type
            toml_file: Optional path to the TOML file, defaults to config.toml
        """
        super().__init__(settings_cls)
        self.toml_file = toml_file or DEFAULT_CONFIG_PATH
        self.toml_data: dict[str, Any] = {}

        try:
            if not self.toml_file.exists():
                raise ConfigMissingError(self.toml_file)
            with open(self.toml_file, "rb") as f:
                self.toml_data = tomli.load(f)
        except FileNotFoundError as e:
            raise ValueError(f"TOML configuration file not found at {self.toml_file}") from e
        except tomli.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML configuration file: {e}") from e

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:  # noqa: ANN401
        """Get a field value from the TOML data.

        Args:
            field: The field info
            field_name: The name of the field

        Returns:
            Tuple of (field_value, field_key, value_is_complex)
        """
        field_value = None
        field_key = field_name
        value_is_complex = False

        # Handle nested structures in TOML (graph.use_memcp_entities)
        parts = field_name.split(".")
        if len(parts) > 1 and parts[0] in self.toml_data:
            # Handle nested values (e.g., graph.use_memcp_entities)
            section = self.toml_data
            for part in parts[:-1]:
                section = section.get(part, {})

            field_value = section.get(parts[-1])
        elif field_name in self.toml_data:
            # Direct top-level value
            field_value = self.toml_data.get(field_name)

        return field_value, field_key, value_is_complex

    def prepare_field_value(self, field_name: str, field: Any, value: Any, value_is_complex: bool) -> Any:  # noqa: ANN401
        """Prepare the field value for use.

        Args:
            field_name: The name of the field
            field: The field info
            value: The value to prepare
            value_is_complex: Whether the value is complex

        Returns:
            The prepared value
        """
        return value

    def __call__(self) -> dict[str, Any]:
        """Return the TOML data as a dictionary."""
        return self.toml_data
