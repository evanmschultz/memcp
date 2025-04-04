"""Queue progress display functionality for Graphiti MCP Server."""

from memcp.core.queue.stats import QueueStatsTracker

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table


class QueueProgressDisplay:
    """Displays the progress of episode processing queues.

    This class is responsible ONLY for rendering queue statistics.
    It does not handle updates directly.
    """

    def __init__(self, console: Console, stats_tracker: "QueueStatsTracker") -> None:
        """Initialize the progress display.

        Args:
            console: Rich console to use for display
            stats_tracker: Queue statistics tracker to monitor
        """
        self.console = console
        self.stats_tracker = stats_tracker

        # Create a single progress display object that will be reused
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.fields[count]}"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[time]}"),
            expand=True,
        )

        # Track task IDs by queue_id
        self.tasks: dict[str, TaskID] = {}

        # Create a single Panel instance that wraps the Progress
        self.panel = Panel(
            self.progress,
            title="[bold cyan]Queue Progress[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )

    def render(self) -> None:
        """Update the progress display with current statistics.

        This renders the current state of queue stats onto the progress object.
        """
        stats = self.stats_tracker.get_stats()
        totals = self.stats_tracker.get_totals()

        # Initialize tasks if not already done
        if not self.tasks:
            # Create a total task
            self.tasks["total"] = self.progress.add_task(
                "[bold blue]Total Progress",
                total=max(totals["total"], 1),  # Avoid division by zero
                completed=totals["completed"],
                count=f"{totals['completed']}/{totals['total']}",
                time=f"avg: {totals['avg_time']:.1f}s",
            )

            # Create tasks for each queue
            for queue_id, queue_stats in stats.items():
                self.tasks[queue_id] = self.progress.add_task(
                    f"[green]{queue_id}",
                    total=max(queue_stats["total"], 1),
                    completed=queue_stats["completed"],
                    count=f"{queue_stats['completed']}/{queue_stats['total']}",
                    time=f"avg: {queue_stats['avg_time']:.1f}s",
                )
        else:
            # Update the total task
            self.progress.update(
                self.tasks["total"],
                total=max(totals["total"], 1),
                completed=totals["completed"],
                count=f"{totals['completed']}/{totals['total']}",
                time=f"avg: {totals['avg_time']:.1f}s",
            )

            # Update existing tasks and add new ones for each queue
            for queue_id, queue_stats in stats.items():
                # Create a new task if this is a new queue
                if queue_id not in self.tasks:
                    self.tasks[queue_id] = self.progress.add_task(
                        f"[green]{queue_id}",
                        total=max(queue_stats["total"], 1),
                        completed=queue_stats["completed"],
                        count=f"{queue_stats['completed']}/{queue_stats['total']}",
                        time=f"avg: {queue_stats['avg_time']:.1f}s",
                    )
                else:
                    # Update existing task
                    self.progress.update(
                        self.tasks[queue_id],
                        total=max(queue_stats["total"], 1),
                        completed=queue_stats["completed"],
                        count=f"{queue_stats['completed']}/{queue_stats['total']}",
                        time=f"avg: {queue_stats['avg_time']:.1f}s",
                    )

    def get_renderable(self) -> Panel:
        """Get the current progress display as a renderable panel.

        Returns the same Panel object each time, which contains the
        continuously updated Progress object.
        """
        return self.panel

    def generate_status_table(self) -> Table:
        """Generate a detailed status table of all queues.

        Returns:
            Rich Table with detailed queue information
        """
        stats = self.stats_tracker.get_stats()
        totals = self.stats_tracker.get_totals()

        table = Table(title="Queue Status", show_header=True, header_style="bold cyan")

        # Add columns
        table.add_column("Queue ID", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Completed", justify="right", style="green")
        table.add_column("In Progress", justify="right", style="yellow")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Avg Time", justify="right", style="blue")

        # Add rows for each queue
        for queue_id, queue_stats in stats.items():
            table.add_row(
                queue_id,
                str(queue_stats["total"]),
                str(queue_stats["completed"]),
                str(queue_stats["in_progress"]),
                str(queue_stats["failed"]),
                f"{queue_stats['avg_time']:.2f}s",
            )

        # Add totals row
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{totals['total']}[/bold]",
            f"[bold green]{totals['completed']}[/bold green]",
            f"[bold yellow]{totals['in_progress']}[/bold yellow]",
            f"[bold red]{totals['failed']}[/bold red]",
            f"[bold blue]{totals['avg_time']:.2f}s[/bold blue]",
        )

        return table

    def stop(self) -> None:
        """Clean up any resources used by the display."""
        pass  # No resources to clean up
