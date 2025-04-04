"""Display management utilities for MemCP."""

from memcp.config.settings import MCPConfig
from memcp.console.queue_display import QueueProgressDisplay
from memcp.utils.memcp_rich_theme import GRAPHITI_THEME

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

    def create_status_display(self, config: MCPConfig) -> Status:
        """Create a status object based on MCP configuration.

        Args:
            config: MCP configuration

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

    def update_display(
        self,
        queue_progress_display: QueueProgressDisplay | None = None,
        status_obj: Status | None = None,
    ) -> None:
        """Update the display with current content.

        This method should be called whenever queue stats change.

        Args:
            queue_progress_display: Progress display to update (defaults to stored one)
            status_obj: Status object to update (defaults to stored one)
        """
        if not self.live:
            return

        # Use provided components or fall back to stored ones
        progress_display = queue_progress_display or self.queue_progress_display
        status = status_obj or self.status_obj

        if not progress_display or not status:
            return

        # Tell the progress display to render based on current stats
        progress_display.render()

        # Create a NEW Group with the updated panel and status
        new_display_group = Group(progress_display.get_renderable(), status)

        # Call live.update() with the new Group (with refresh=True)
        self.live.update(new_display_group, refresh=True)

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
    ) -> None:
        """Run a coroutine with a live display.

        This method handles creating and cleaning up the Live display context.

        Args:
            queue_progress_display: Progress display for queues (can be None)
            status_obj: Status object with server info
            run_coroutine: Coroutine to run while displaying the Live context
        """
        # Start the live display
        self.start_live_display(queue_progress_display, status_obj)

        try:
            # Run the provided coroutine
            await run_coroutine()
        finally:
            # Always stop the live display
            self.stop_live_display()

    def show_server_info(self, config: MCPConfig, graph_id: str | None, pid: int) -> None:
        """Display server information in a panel.

        Args:
            config: MCP server configuration
            graph_id: Graph ID being used
            pid: Process ID of the server
        """
        # Create server info table
        server_table = Table(show_header=False, box=box.SIMPLE, expand=False)
        server_table.add_column("Property", style="info")
        server_table.add_column("Value", style="success")
        server_table.add_row("Status", "Running")
        server_table.add_row("PID", str(pid))
        server_table.add_row("Transport", config.transport)
        server_table.add_row("Graph ID", graph_id or "None")

        if config.transport == "sse":
            server_table.add_row("Address", f"{config.host}:{config.port}")

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
