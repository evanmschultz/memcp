#!/usr/bin/env python3
"""Command-line interface for MemCP."""

from memcp.api.memcp_server import MemCPServer
from memcp.app_builder import create_server
from memcp.utils import configure_logging, get_logger
from memcp.utils.errors import ServerInitializationError, ServerRuntimeError

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()
configure_logging()
logger = get_logger(__name__)


async def run_server() -> None:
    """Create and run the MemCP server asynchronously.

    This function is the core async implementation called by main():
    1. Creates the server using the application builder
    2. Runs the server once initialized

    Raises:
        ServerInitializationError: If the server fails to initialize properly
        ServerRuntimeError: If an error occurs during server operation

    The function uses structured error handling to convert generic exceptions
    into specific MemCP error types with contextual information.
    """
    try:
        server: MemCPServer = await create_server()
    except Exception as e:
        raise ServerInitializationError(f"Failed to initialize MemCP server: {str(e)}") from e

    try:
        await server.run()
    except Exception as e:
        raise ServerRuntimeError(f"Error while running MemCP server: {str(e)}") from e


def main() -> None:
    """Main entry point for the MemCP CLI.

    Creates and runs the MemCP server within an asyncio event loop.

    Important implementation notes:
    - Uses asyncio.run() to manage the event loop lifecycle
    - Restores sys.stderr to its original state in a finally block
      This is necessary because Rich library modifies terminal streams
      and not resetting them can affect other terminal programs

    Errors during server creation or operation will propagate up
    and should be handled by the caller or shell.
    """
    try:
        asyncio.run(run_server())
    finally:
        sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
