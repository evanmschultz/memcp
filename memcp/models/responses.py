"""Models for MemCP Server responses."""

from typing import Any, TypedDict


class ErrorResponse(TypedDict):
    """Represents an error response from the MCP server."""

    error: str


class SuccessResponse(TypedDict):
    """Represents a successful response from the MCP server."""

    message: str


class NodeResult(TypedDict):
    """Represents a node result from the MCP server."""

    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    """Represents a node search response from the MCP server."""

    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    """Represents a fact search response from the MCP server."""

    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    """Represents an episode search response from the MCP server."""

    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    """Represents a status response from the MCP server."""

    status: str
    message: str
