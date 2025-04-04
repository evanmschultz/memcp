#!/usr/bin/env python3
"""Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)."""

from memcp.app_factory import AppFactory
from memcp.utils import configure_logging, get_logger

import asyncio
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
configure_logging()

# Get a logger for this module
logger = get_logger(__name__)


async def run_mcp_server() -> None:
    """Run the MCP server in the current event loop."""
    try:
        # Create the app factory
        factory = AppFactory()

        # Use the factory to create a fully configured server
        server, _ = await factory.create_server()

        # Check if run method exists
        if not hasattr(server, "run"):
            # If not, access the attributes (sometimes this helps with property resolution)
            server_dir = dir(server)
            print(f"Available methods: {server_dir}")
            raise AttributeError(
                f"'GraphitiServer' object has no 'run' method. Available methods: {server_dir}"
            )

        # Run the server
        await server.run()
    except Exception as e:
        logger.error(f"Error running Graphiti MCP server: {str(e)}")
        raise


def main() -> None:
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except Exception as e:
        # Ensure we show an error but also our final message
        logger.error(f"Error initializing Graphiti MCP server: {str(e)}")
        # Let the exception propagate so the traceback is shown
        raise
    finally:
        # Ensure we reset stderr
        sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
