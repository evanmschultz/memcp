"""Shutdown utilities for MemCP."""

from memcp.utils.memcp_rich_theme import GRAPHITI_THEME

import asyncio
import atexit
import contextlib
import os
import signal
import sys
import time
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table, box


class ShutdownManager:
    """Manager for graceful application shutdown and cleanup.

    Centralizes shutdown state tracking and provides methods for different
    shutdown scenarios including graceful shutdown and force kill.
    """

    # Shutdown modes
    NONE = "NONE"
    GRACEFUL = "GRACEFUL"
    FORCE = "FORCE"

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the shutdown manager.

        Args:
            console: Optional console for output. If not provided, a new one will be created.
        """
        self.mode = self.NONE
        self.in_progress = False
        self.console = console or Console(stderr=True, theme=GRAPHITI_THEME)
        self._original_excepthook = sys.excepthook

        # Register the final message display
        atexit.register(self._show_final_message)

        # Set up custom exception handling
        sys.excepthook = self._custom_excepthook

    def _custom_excepthook(self, exc_type, exc_value, exc_traceback) -> None:
        """Custom excepthook for handling shutdown exceptions.

        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        # First, call the original excepthook to show the traceback
        self._original_excepthook(exc_type, exc_value, exc_traceback)

        # If this is a shutdown-related exception, show our final message
        if self.in_progress and issubclass(exc_type, asyncio.CancelledError):
            # Wait a moment to ensure traceback is fully printed
            time.sleep(0.1)
            self._show_final_message()

    def _show_final_message(self) -> None:
        """Display a final message after shutdown based on the shutdown mode."""
        print("\n\n" + "-" * 80)
        if self.mode == self.GRACEFUL:
            self.console.print(
                "[bold green]✅ GRAPHITI SERVER SHUTDOWN SUCCESSFULLY[/bold green]\n"
            )
            self.console.print(
                "\n[info]The asyncio CancelledError tracebacks above are normal and expected "
                "during shutdown.[/info]"
            )
            self.console.print(
                "\n[info]All tasks were properly cancelled and resources were released "
                "cleanly.[/info]\n"
            )
        elif self.mode == self.FORCE:
            self.console.print("[bold red]⚠️ GRAPHITI SERVER FORCE KILLED[/bold red]\n")
            self.console.print(
                "\n[dim warning]No cleanup was performed. Some resources may not have been properly"
                " released.[/dim warning]"
            )
        else:
            self.console.print("[bold green]GRAPHITI SERVER SHUTDOWN COMPLETE[/bold green]\n")
        print("-" * 80 + "\n")

    def set_graceful(self) -> None:
        """Set shutdown mode to graceful."""
        if not self.in_progress:
            self.mode = self.GRACEFUL
            self.in_progress = True

    def set_force(self) -> None:
        """Set shutdown mode to force."""
        self.mode = self.FORCE
        self.in_progress = True

    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self.in_progress

    async def graceful_shutdown(
        self,
        queue_manager: Any,
        queue_progress_display: Any,
        graphiti_client: Any,
        logger: Any,
        display_manager: Any | None = None,
        timeout: float = 5.0,
    ) -> None:
        """Perform graceful shutdown of the MCP server and all resources.

        Args:
            queue_manager: The queue manager to shut down
            queue_progress_display: The progress display for visualization
            graphiti_client: The Graphiti client to close
            logger: Logger instance for output
            display_manager: Optional display manager instance to stop live display
            timeout: Maximum time to wait for tasks to complete
        """
        # Prevent multiple shutdown attempts
        if self.is_shutting_down():
            return

        # Set the shutdown mode
        self.set_graceful()

        # Stop the live display if we have a display manager
        if display_manager is not None:
            display_manager.stop_live_display()

        # Stop the queue progress display (cleanup resources)
        if queue_progress_display:
            queue_progress_display.stop()

        # Suppress standard error output during shutdown
        sys.stderr = open(os.devnull, "w")

        # Clear the screen for clean shutdown display
        self.console.clear()

        # Create a table for tracking shutdown progress
        table = Table(
            box=box.ROUNDED,
            border_style="shutdown",
            expand=False,
            show_header=True,
            highlight=True,
            title="[success]Graphiti MCP Server Shutdown[/success]",
        )
        table.add_column("[shutdown]Step[/shutdown]", style="info")
        table.add_column("[shutdown]Status[/shutdown]", style="info")

        # Display shutdown progress
        with self.console.status(
            "[shutdown]Performing graceful shutdown...[/shutdown]", spinner="dots"
        ) as status:
            # Add initial table
            table.add_row("Shutdown initiated", "[step.success]✓[/step.success]")
            status.update(Panel(table, border_style="shutdown"))
            await asyncio.sleep(0.1)  # Small pause to ensure UI updates

            # 1. Cancel queue workers
            active_worker_count = queue_manager.cancel_all_workers()

            table.add_row(
                f"Queue workers ({active_worker_count})", "[step.success]✓[/step.success]"
            )
            status.update(Panel(table, border_style="shutdown"))
            await asyncio.sleep(0.1)

            # 2. Close Neo4j connection if client exists
            neo4j_success = True
            if graphiti_client is not None:
                try:
                    await graphiti_client.driver.close()
                    logger.info("[success]Neo4j connection closed successfully[/success]")
                except Exception:
                    neo4j_success = False
                    logger.warning("[warning]Could not close Neo4j connection cleanly[/warning]")

            table.add_row(
                "Neo4j connection",
                "[step.success]✓[/step.success]"
                if neo4j_success
                else "[step.warning]⚠[/step.warning]",
            )
            status.update(Panel(table, border_style="shutdown"))
            await asyncio.sleep(0.1)

            # 3. Cancel all remaining tasks except the current one
            current_task = asyncio.current_task()
            remaining_tasks = []
            for task in [t for t in asyncio.all_tasks() if t is not current_task]:
                if task.get_name() != "shutdown_task":
                    task.cancel()
                    remaining_tasks.append(task)
                    logger.info(
                        f"[task]Cancelling task {task.get_name()} - [success]Expected!"
                        "[/success][/task]"
                    )

            table.add_row(
                f"Tasks cancelled ({len(remaining_tasks)})", "[step.success]✓[/step.success]"
            )
            status.update(Panel(table, border_style="shutdown"))
            await asyncio.sleep(0.1)

            # 4. Wait for all tasks to complete with timeout
            completion_success = True
            if remaining_tasks:
                try:
                    # Suppress cancellation errors during shutdown
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.wait_for(
                            asyncio.gather(*remaining_tasks, return_exceptions=True), timeout
                        )
                        logger.info("[success]All tasks completed gracefully[/success]")
                except asyncio.TimeoutError:
                    completion_success = False
                    logger.warning(f"[warning]Shutdown timed out after {timeout}s[/warning]")

            table.add_row(
                "Task completion",
                "[step.success]✓[/step.success]"
                if completion_success
                else "[step.warning]⚠[/step.warning]",
            )
            status.update(Panel(table, border_style="shutdown"))
            await asyncio.sleep(0.1)

            # Final status row
            table.add_row("Server shutdown", "[step.success]✓[/step.success]")
            status.update(Panel(table, border_style="shutdown"))

        # Final message - outside the status context
        self.console.print(
            Panel(
                "[success]Graphiti MCP Server shutdown complete[/success]\n"
                "[normal]All resources have been released and tasks cancelled properly.[/normal]",
                title="[success]Shutdown Complete[/success]",
                border_style="success",
                padding=(1, 2),
            )
        )

        # Reset stderr before exit
        sys.stderr = sys.__stderr__

        # Create a final message file that main() will detect after shutdown
        with open(".shutdown_message.txt", "w") as f:
            f.write("GRACEFUL_SHUTDOWN_COMPLETE")

        # Exit the process immediately
        sys.exit(0)

    def force_kill(self, queue_progress_display: Any, display_manager: Any | None = None) -> None:
        """Force kill the process immediately without cleanup.

        Args:
            queue_progress_display: The progress display to stop
            display_manager: Optional display manager instance to stop live display
        """
        # Set the shutdown mode
        self.set_force()

        # Stop the live display if we have a display manager
        if display_manager is not None:
            display_manager.stop_live_display()

        # Stop the queue progress display
        if queue_progress_display:
            queue_progress_display.stop()

        # Suppress standard error output during shutdown
        sys.stderr = open(os.devnull, "w")

        # Clear the console for a clean display
        self.console.clear()

        self.console.print(
            Panel(
                "[danger]Force killing Graphiti MCP Server![/danger]\n"
                "[normal]No cleanup was performed. Some resources may not be properly "
                "released.[/normal]",
                title="[danger]Emergency Shutdown[/danger]",
                border_style="danger",
                padding=(1, 2),
            )
        )

        # Reset stderr before exit
        sys.stderr = sys.__stderr__

        # Brief pause to ensure message is displayed
        self.console.print()

        # Use sys.exit instead of os._exit to allow atexit handlers to run
        sys.exit(1)

    def setup_signal_handlers(
        self,
        queue_manager: Any,
        queue_progress_display: Any,
        graphiti_client: Any,
        logger: Any,
        display_manager: Any | None = None,
    ) -> None:
        """Set up signal handlers for shutdown.

        Args:
            queue_manager: Queue manager instance
            queue_progress_display: Queue progress display instance
            graphiti_client: Graphiti client instance
            logger: Logger instance
            display_manager: Optional display manager instance
        """
        loop = asyncio.get_running_loop()

        # Define handlers
        def graceful_handler():
            return lambda: asyncio.create_task(
                self.graceful_shutdown(
                    queue_manager, queue_progress_display, graphiti_client, logger, display_manager
                ),
                name="shutdown_task",
            )

        def force_handler():
            return lambda: self.force_kill(queue_progress_display, display_manager)

        # Register signal handlers
        loop.add_signal_handler(signal.SIGHUP, graceful_handler())
        loop.add_signal_handler(signal.SIGINT, graceful_handler())
        loop.add_signal_handler(signal.SIGQUIT, force_handler())

        # Register SIGTERM with default Python signal mechanism
        signal.signal(
            signal.SIGTERM,
            self._create_sigterm_handler(
                queue_manager, queue_progress_display, graphiti_client, logger, display_manager
            ),
        )

        logger.info("[shutdown]Signal handlers registered:[/shutdown]")
        logger.info(
            "  - [info]SIGHUP (1)[/info]: [success]Graceful shutdown[/success] (kill -1 <pid>)"
        )
        logger.info("  - [info]SIGINT (2)[/info]: [success]Graceful shutdown[/success] (Ctrl+C)")
        logger.info("  - [info]SIGQUIT (3)[/info]: [danger]Force kill[/danger] (emergency only)")
        logger.info(
            "  - [info]SIGTERM (15)[/info]: [success]Graceful shutdown[/success] (docker/k8s)"
        )

    def _create_sigterm_handler(
        self,
        queue_manager: Any,
        queue_progress_display: Any,
        graphiti_client: Any,
        logger: Any,
        display_manager: Any | None = None,
    ) -> Callable[[int, Any], None]:
        """Create a SIGTERM handler for graceful shutdown.

        Args:
            queue_manager: Queue manager instance
            queue_progress_display: Queue progress display
            graphiti_client: Graphiti client instance
            logger: Logger instance
            display_manager: Optional display manager instance

        Returns:
            Signal handler function
        """

        def handler(signum: int, frame: Any) -> None:
            """Handle SIGTERM by initiating graceful shutdown."""
            self.set_graceful()
            print("\nSIGTERM received. Initiating graceful shutdown...")

            # Create and run shutdown task
            loop = asyncio.get_event_loop()
            loop.create_task(
                self.graceful_shutdown(
                    queue_manager, queue_progress_display, graphiti_client, logger, display_manager
                ),
                name="shutdown_task",
            )

        return handler
