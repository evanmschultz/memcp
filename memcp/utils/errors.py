"""Custom exceptions for the MemCP application."""


class MemCPError(Exception):
    """Base exception for all MemCP errors."""

    pass


class MemCPServerError(MemCPError):
    """Base exception for MemCP server errors."""

    pass


class ServerInitializationError(MemCPServerError):
    """Raised when the MemCP server fails to initialize properly."""

    pass


class ServerRuntimeError(MemCPServerError):
    """Raised when the MemCP server encounters an error during operation."""

    pass
