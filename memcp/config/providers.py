"""Configuration providers for different config sources."""

import argparse
import os
from typing import Any, Protocol, runtime_checkable

import tomli


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol defining a configuration provider interface."""

    def get_config(self) -> dict[str, Any]:
        """Get configuration from this provider."""
        ...


class EnvConfigProvider:
    """Provider for environment variables configuration."""

    def get_config(self) -> dict[str, Any]:
        """Extract configuration from environment variables."""
        config = {
            "neo4j": {
                "uri": os.environ.get("NEO4J_URI"),
                "user": os.environ.get("NEO4J_USER"),
                "password": os.environ.get("NEO4J_PASSWORD"),
            },
            "openai": {
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "base_url": os.environ.get("OPENAI_BASE_URL"),
            },
            "model": {
                "name": os.environ.get("MODEL_NAME"),
            },
            "graph": {
                "id": os.environ.get("GRAPH_ID"),
                "use_custom_entities": os.environ.get("USE_CUSTOM_ENTITIES") == "true",
            },
            "server": {
                "transport": os.environ.get("TRANSPORT"),
                "host": os.environ.get("HOST"),
                "port": os.environ.get("PORT"),
            },
        }
        return config


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


class ArgsConfigProvider:
    """Provider for command line arguments configuration."""

    def __init__(self, args: argparse.Namespace | None = None) -> None:
        """Initialize with parsed arguments or parse them."""
        self.args = args if args is not None else self._parse_args()

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Run the MemCP server")
        parser.add_argument(
            "--graph-id",
            help="Graph ID for the knowledge graph. If not provided, a random graph ID will be generated.",
            default=None,
        )
        parser.add_argument(
            "--transport",
            choices=["sse", "stdio"],
            help="Transport to use for communication with the client. Default: sse",
            default="sse",
        )
        parser.add_argument(
            "--model",
            help="Model name to use with the LLM client. Default: gpt-4o-mini",
            default="gpt-4o-mini",
        )
        parser.add_argument("--destroy-graph", action="store_true", help="Destroy all graphs")
        parser.add_argument(
            "--use-custom-entities",
            action="store_true",
            help="Enable entity extraction using predefined entities",
        )
        parser.add_argument(
            "--host",
            help="Host address to bind the server to. Default: 127.0.0.1",
            default="127.0.0.1",
        )
        parser.add_argument(
            "--port",
            type=int,
            help="Port number to bind the server to. Default: 8000",
            default=8000,
        )
        parser.add_argument(
            "--config",
            help="Path to TOML configuration file. If not provided, the application will use environment variables "
            "and command line arguments.",
            default=None,
        )

        return parser.parse_args()

    def get_config(self) -> dict[str, Any]:
        """Extract configuration from command line arguments."""
        config = {
            "model": {},
            "graph": {},
            "server": {},
        }

        # Only add non-None values
        if self.args.model:
            config["model"]["name"] = self.args.model

        if self.args.graph_id:
            config["graph"]["id"] = self.args.graph_id

        if self.args.use_custom_entities:
            config["graph"]["use_custom_entities"] = True

        if self.args.transport:
            config["server"]["transport"] = self.args.transport

        if self.args.host:
            config["server"]["host"] = self.args.host

        if self.args.port:
            config["server"]["port"] = self.args.port

        # Add destroy_graph flag separately since it's not part of any config object
        config["destroy_graph"] = self.args.destroy_graph

        return config
