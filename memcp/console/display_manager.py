"""Display management utilities for MemCP."""

from memcp.config import ServerConfig
from memcp.console.queue_display import QueueProgressDisplay
from memcp.utils.memcp_rich_theme import GRAPHITI_THEME

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.status import Status
from rich.table import Table, box
from rich.text import Text


class DisplayManager:
    """Manager for console displays and progress visualization.

    Centralizes console creation and display rendering to provide a consistent
    interface for all console output in the application.
    """

    def __init__(self) -> None:
        """Initialize the display manager with console instances."""
        self.main_console = Console(theme=GRAPHITI_THEME)
        self.shutdown_console = Console(stderr=True, theme=GRAPHITI_THEME)
        self.progress_console = Console(theme=GRAPHITI_THEME)

        # Live display instance will be initialized when start_live_display is called
        self.live: Live | None = None

        # Store references to display components
        self.queue_progress_display: QueueProgressDisplay | None = None
        self.status_obj: Status | None = None

        # Debounce-related attributes
        self._last_update_time = 0
        self._update_interval = 0.25  # seconds
        self._update_pending = False
        self._update_lock = asyncio.Lock()
        self._pending_state = ""

        # State tracking
        self._last_visual_state = ""

    def get_main_console(self) -> Console:
        """Get the main console instance.

        Returns:
            Main console for standard output
        """
        return self.main_console

    def get_shutdown_console(self) -> Console:
        """Get the shutdown console instance.

        Returns:
            Shutdown console for stderr output
        """
        return self.shutdown_console

    def get_progress_console(self) -> Console:
        """Get the progress console instance.

        Returns:
            Progress console for detailed output
        """
        return self.progress_console

    def create_status_display(self, config: ServerConfig) -> Status:
        """Create a status object based on MCP configuration.

        Args:
            config: Server configuration

        Returns:
            Rich Status object
        """
        if config.transport == "stdio":
            return Status(
                Text(" STDIO Server is running...", style="normal"),
                spinner="arc",
                spinner_style="success",
            )
        else:
            return Status(
                Text(f" {config.transport.upper()} Server running on {config.host}:{config.port}"),
                spinner="arc",
                spinner_style="green",
            )

    def _get_visual_state_hash(self) -> str:
        """Generate a hash representing the current visual state.

        Returns:
            String hash of the current visual state
        """
        if not self.queue_progress_display:
            return ""

        # Get stats as a string representation for hashing
        stats = self.queue_progress_display.stats_tracker.get_stats()
        totals = self.queue_progress_display.stats_tracker.get_totals()

        # Create a string representation of the state
        state_repr = (
            f"total:{totals['total']}-completed:{totals['completed']}-"
            f"in_progress:{totals['in_progress']}-failed:{totals['failed']}"
        )

        # Hash the state string
        return hashlib.md5(state_repr.encode()).hexdigest()

    async def debounced_update(self) -> None:
        """Update the display with throttling to prevent too many updates."""
        async with self._update_lock:
            current_time = time.time()

            # Get current visual state
            current_state = self._get_visual_state_hash()

            # If state hasn't changed, don't update
            if current_state == self._last_visual_state and current_state != "":
                return

            # Check if we're within the debounce interval
            if current_time - self._last_update_time < self._update_interval:
                # If already pending, don't schedule another update
                if not self._update_pending:
                    self._update_pending = True
                    # Store the state that triggered this update
                    self._pending_state = current_state
                    asyncio.create_task(self._delayed_update())
                return

            # Update the last visual state
            self._last_visual_state = current_state
            self._last_update_time = current_time
            self._update_pending = False
            self.update_display()

    def _render_queue_progress(self) -> bool:
        """Render queue progress and return True if visual state changed.

        Returns:
            True if visual state changed, False otherwise
        """
        if not self.queue_progress_display:
            return False

        # Get the visual state before rendering
        previous_state = self._last_visual_state

        # Perform rendering
        self.queue_progress_display.render()

        # Get the new visual state
        current_state = self._get_visual_state_hash()

        # Update last visual state
        self._last_visual_state = current_state

        # Return whether the state changed
        return previous_state != current_state

    def update_display(self) -> None:
        """Update the display with current content.

        This method should be called whenever queue stats change.
        """
        if not self.live:
            return

        # Use stored components
        if not self.queue_progress_display or not self.status_obj:
            return

        # Only create a new Group and update if the visual state changed
        if self._render_queue_progress():
            # Create a NEW Group with the updated panel and status
            new_display_group = Group(self.queue_progress_display.get_renderable(), self.status_obj)

            # Call live.update() with the new Group (with refresh=True)
            self.live.update(new_display_group, refresh=True)

    async def _delayed_update(self) -> None:
        """Perform a delayed update after the debounce interval."""
        await asyncio.sleep(self._update_interval)
        async with self._update_lock:
            if self._update_pending:
                # Check if state has changed since we scheduled this update
                current_state = self._get_visual_state_hash()
                if current_state != self._last_visual_state:
                    self._update_pending = False
                    self._last_visual_state = current_state
                    self._last_update_time = time.time()
                    self.update_display()
                else:
                    self._update_pending = False

    def start_live_display(
        self,
        queue_progress_display: QueueProgressDisplay,
        status_obj: Status,
    ) -> Live:
        """Start a Live display context with the current layout.

        Args:
            queue_progress_display: Progress display for queues
            status_obj: Status object with server info

        Returns:
            The Live context manager instance
        """
        # Store references to components
        self.queue_progress_display = queue_progress_display
        self.status_obj = status_obj

        # Reset the visual state tracking
        self._last_visual_state = ""

        # Ensure the progress display is rendered
        queue_progress_display.render()

        # Create initial Group with our renderables
        display_group = Group(queue_progress_display.get_renderable(), status_obj)

        # Create the Live display with the group
        self.live = Live(
            display_group,
            console=self.main_console,
            refresh_per_second=4,
            transient=False,
        )

        # Start the display
        self.live.start()

        return self.live

    def stop_live_display(self) -> None:
        """Stop the Live display."""
        if self.live is not None:
            self.live.stop()
            self.live = None

        # Clear references to components
        self.queue_progress_display = None
        self.status_obj = None

    async def run_with_live_display(
        self,
        queue_progress_display: QueueProgressDisplay,
        status_obj: Status,
        run_coroutine: Callable[[], Awaitable[Any]],
        suppress_updates_for: float = 0.0,
    ) -> None:
        """Run a coroutine with a live display.

        This method handles creating and cleaning up the Live display context.

        Args:
            queue_progress_display: Progress display for queues (can be None)
            status_obj: Status object with server info
            run_coroutine: Coroutine to run while displaying the Live context
            suppress_updates_for: Time in seconds to suppress updates after startup
        """
        # Start the live display
        self.start_live_display(queue_progress_display, status_obj)

        # If we need to suppress updates, set the last update time to the future
        if suppress_updates_for > 0:
            self._last_update_time = time.time() + suppress_updates_for

        try:
            # Run the provided coroutine
            await run_coroutine()
        finally:
            # Always stop the live display
            self.stop_live_display()

    def show_server_info(
        self,
        config: ServerConfig,
        graph_id: str | None,
        pid: int,
        neo4j_name: str,
        model_name: str,
    ) -> None:
        """Display server information in a panel.

        Args:
            config: Server configuration
            graph_id: Graph ID being used
            pid: Process ID of the server
            neo4j_name: Name of the Neo4j database
            model_name: Name of the LLM model being used
        """
        server_table = Table(show_header=False, box=box.SIMPLE, expand=False)
        server_table.add_column("Property", style="info")
        server_table.add_column("Value", style="success")
        server_table.add_row("Status", "Running")
        server_table.add_row("PID", str(pid))
        server_table.add_row("Neo4j_Name", neo4j_name)
        server_table.add_row("Transport", config.transport)
        server_table.add_row("Graph_ID", graph_id or "None")

        if config.transport == "sse":
            server_table.add_row("Address", f"{config.host}:{config.port}")

        server_table.add_row("LLM_Model", model_name)

        self.main_console.print(
            Panel(
                server_table,
                title="[highlight]Graphiti MCP Server[/highlight]",
                border_style="success",
            )
        )

        # Show commands in a separate panel
        command_table = Table(show_header=False, box=box.SIMPLE, expand=False)
        command_table.add_column("Command", style="highlight")
        command_table.add_column("Description", style="normal")
        command_table.add_row("Ctrl+C", "[success]Graceful shutdown[/success]")
        command_table.add_row(f"kill -1 {pid}", "[success]Graceful shutdown[/success]")
        command_table.add_row(f"kill -3 {pid}", "[danger]Force kill[/danger] (emergency only)")

        self.main_console.print(
            Panel(
                command_table,
                title="[normal]Control Commands[/normal]",
                border_style="normal",
                padding=(1, 2),
            )
        )
