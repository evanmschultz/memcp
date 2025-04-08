"""Queue statistics tracking functionality for Graphiti MCP Server."""

import time
from typing import Any


class QueueStatsTracker:
    """Tracks statistics for episode processing queues.

    This class follows the single responsibility principle by only handling
    the tracking of queue statistics, separate from display concerns.
    """

    def __init__(self) -> None:
        """Initialize the statistics tracker."""
        self.queue_stats: dict[str, dict[str, Any]] = {}
        self.completed_count = 0
        self.total_count = 0
        self._start_times: dict[str, float] = {}
        self._processing_times: dict[str, list[float]] = {}

    def register_queue(self, queue_id: str) -> None:
        """Register a new queue for tracking.

        Args:
            queue_id: Identifier for the queue (typically a graph_id)
        """
        if queue_id not in self.queue_stats:
            self.queue_stats[queue_id] = {
                "total": 0,
                "completed": 0,
                "in_progress": 0,
                "failed": 0,
                "avg_time": 0.0,
            }

    def add_task(self, queue_id: str) -> None:
        """Record a new task added to the queue.

        Args:
            queue_id: Identifier for the queue
        """
        self.register_queue(queue_id)
        self.queue_stats[queue_id]["total"] += 1
        self.total_count += 1

    def start_processing(self, queue_id: str, task_id: str) -> None:
        """Record that a task has started processing.

        Args:
            queue_id: Identifier for the queue
            task_id: Unique identifier for the task
        """
        self.register_queue(queue_id)
        self.queue_stats[queue_id]["in_progress"] += 1
        self._start_times[task_id] = time.time()

    def complete_task(self, queue_id: str, task_id: str, success: bool = True) -> None:
        """Record a task completion.

        Args:
            queue_id: Identifier for the queue
            task_id: Unique identifier for the task
            success: Whether the task completed successfully
        """
        self.register_queue(queue_id)

        # Record completion
        if success:
            self.queue_stats[queue_id]["completed"] += 1
            self.completed_count += 1
        else:
            self.queue_stats[queue_id]["failed"] += 1

        # Update in-progress count
        self.queue_stats[queue_id]["in_progress"] -= 1

        # Calculate processing time if we have a start time
        if task_id in self._start_times:
            elapsed = time.time() - self._start_times[task_id]

            # Store processing time for this queue
            if queue_id not in self._processing_times:
                self._processing_times[queue_id] = []

            self._processing_times[queue_id].append(elapsed)

            # Recalculate average (last 10 tasks only to avoid skew from old data)
            recent_times = self._processing_times[queue_id][-10:]
            self.queue_stats[queue_id]["avg_time"] = sum(recent_times) / len(recent_times)

            # Clean up start time
            del self._start_times[task_id]

    def get_stats(self, queue_id: str | None = None) -> dict[str, Any]:
        """Get the current statistics for a specific queue or all queues.

        Args:
            queue_id: Optional identifier for a specific queue

        Returns:
            Dictionary of queue statistics
        """
        if queue_id is not None:
            return self.queue_stats.get(
                queue_id,
                {
                    "total": 0,
                    "completed": 0,
                    "in_progress": 0,
                    "failed": 0,
                    "avg_time": 0.0,
                },
            )

        # Return all stats
        return self.queue_stats

    def get_totals(self) -> dict[str, Any]:
        """Get the summed statistics across all queues.

        Returns:
            Dictionary containing aggregated statistics
        """
        totals = {
            "total": self.total_count,
            "completed": self.completed_count,
            "in_progress": 0,
            "failed": 0,
            "avg_time": 0.0,
        }

        # Calculate in_progress and failed counts
        processing_times: list[float] = []
        for stats in self.queue_stats.values():
            totals["in_progress"] += stats["in_progress"]
            totals["failed"] += stats["failed"]

            if stats["avg_time"] > 0:
                processing_times.append(stats["avg_time"])

        # Calculate overall average processing time
        if processing_times:
            totals["avg_time"] = sum(processing_times) / len(processing_times)

        return totals
