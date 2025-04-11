"""Uvicorn configuration utilities for MemCP."""

from typing import Any

# Minimal Uvicorn log configuration to silence default logs
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "minimal": {
            "format": "%(message)s",
        },
    },
    "handlers": {
        "null": {
            "class": "logging.NullHandler",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "uvicorn.error": {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "uvicorn.access": {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "uvicorn.asgi": {"handlers": ["null"], "level": "ERROR", "propagate": False},
    },
}


def get_uvicorn_config(host: str, port: int) -> dict[str, Any]:
    """Get Uvicorn configuration with proper logging settings.

    Args:
        host: Host to bind the server to
        port: Port to bind the server to

    Returns:
        Dictionary with Uvicorn configuration
    """
    return {
        "host": host,
        "port": port,
        "log_level": "critical",  # Only show critical errors
        "log_config": UVICORN_LOG_CONFIG,
        "access_log": False,  # Disable access logs
        "server_header": False,  # Disable server header
        "date_header": False,  # Disable date header
    }
