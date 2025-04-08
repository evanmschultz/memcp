"""Custom exceptions for the MemCP application."""


class MemCPServerError(Exception):
    """Base exception for MemCP server errors."""

    pass


class ServerInitializationError(MemCPServerError):
    """Raised when the MemCP server fails to initialize properly."""

    pass


class TransportError(MemCPServerError):
    """Base exception for transport-related errors."""

    pass


class UnsupportedTransportError(TransportError):
    """Raised when an unsupported transport type is specified."""

    pass


class GraphOperationError(Exception):
    """Base exception for graph operation errors."""

    pass


class EpisodeError(GraphOperationError):
    """Base exception for episode-related errors."""

    pass
