"""Queue management functionality for Graphiti MCP Server."""

from memcp.queue.stats import QueueStatsTracker

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)


class QueueManager:
    """Manages episode processing queues and workers.

    This class is responsible for:
    1. Managing queue creation and access
    2. Starting and tracking queue worker tasks
    3. Tracking queue statistics
    """

    def __init__(self, queue_stats_tracker: QueueStatsTracker) -> None:
        """Initialize the queue manager.

        Args:
            queue_stats_tracker: The stats tracker instance to use
        """
        self.episode_queues: dict[str, asyncio.Queue[Callable[[], Awaitable[None]]]] = {}
        self.queue_workers: dict[str, bool] = {}
        self.queue_stats_tracker: QueueStatsTracker = queue_stats_tracker
        self._state_change_callbacks: list[Callable[[], None]] = []
        self._last_notification_time = 0
        self._notification_interval = 0.1  # seconds
        self._notification_pending = False

    def add_state_change_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback function to be called when queue state changes.

        This allows external components to be notified of state changes
        without the queue manager needing to know about them directly.

        Args:
            callback: Function to call when state changes
        """
        if callback not in self._state_change_callbacks:
            self._state_change_callbacks.append(callback)

    def remove_state_change_callback(self, callback: Callable[[], None]) -> None:
        """Remove a previously added callback function.

        Args:
            callback: Function to remove
        """
        if callback in self._state_change_callbacks:
            self._state_change_callbacks.remove(callback)

    def _notify_state_change(self) -> None:
        """Notify all registered callbacks that state has changed.

        Includes throttling to prevent too many rapid notifications.
        """
        # Only proceed if there are callbacks registered
        if not self._state_change_callbacks:
            return

        current_time = time.time()

        # If we've notified recently, mark as pending but don't notify yet
        if current_time - self._last_notification_time < self._notification_interval:
            self._notification_pending = True
            return

        # Update notification time and reset pending flag
        self._last_notification_time = current_time
        self._notification_pending = False

        # Call all registered callbacks
        for callback in self._state_change_callbacks:
            callback()

    async def _check_pending_notification(self) -> None:
        """Check if there's a pending notification after the throttle interval."""
        await asyncio.sleep(self._notification_interval)
        if self._notification_pending:
            # Reset pending flag to prevent recursive scheduling
            self._notification_pending = False
            self._last_notification_time = time.time()

            # Call all registered callbacks
            for callback in self._state_change_callbacks:
                callback()

    async def enqueue_task(self, group_id: str, process_func: Callable[[], Awaitable[None]]) -> None:
        """Enqueue a task for processing.

        Args:
            group_id: The group ID for the queue
            process_func: The async function to process
        """
        # Initialize queue for this group_id if it doesn't exist
        if group_id not in self.episode_queues:
            self.episode_queues[group_id] = asyncio.Queue()

        # Track the new task in our stats tracker
        self.queue_stats_tracker.add_task(group_id)

        # Notify about state change
        self._notify_state_change()

        # Add the processing function to the queue
        await self.episode_queues[group_id].put(process_func)

        # Start a worker for this queue if one isn't already running
        if not self.queue_workers.get(group_id, False):
            asyncio.create_task(self.process_episode_queue(group_id))

    async def process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes
        from the queue one at a time.
        """
        # Name the task to be able to identify it during shutdown
        current_task = asyncio.current_task()
        if current_task:
            current_task.set_name(f"queue_worker_{group_id}")

        logger.info(f"Starting episode queue worker for group_id: [highlight]{group_id}[/highlight]")
        self.queue_workers[group_id] = True

        try:
            while True:
                # Get the next episode processing function from the queue
                # This will wait if the queue is empty
                process_func = await self.episode_queues[group_id].get()

                # Generate a unique task ID for tracking
                task_id = str(uuid.uuid4())

                try:
                    # Record that processing has started
                    self.queue_stats_tracker.start_processing(group_id, task_id)

                    # Notify about state change
                    self._notify_state_change()

                    # Process the episode
                    await process_func()

                    # Record successful completion
                    self.queue_stats_tracker.complete_task(group_id, task_id, success=True)
                except Exception as e:
                    # Record failed completion
                    self.queue_stats_tracker.complete_task(group_id, task_id, success=False)

                    logger.error(
                        f"Error processing queued episode for group_id [highlight]{group_id}"
                        f"[/highlight]: [danger]{str(e)}[/danger]"
                    )
                finally:
                    # Mark the task as done regardless of success/failure
                    self.episode_queues[group_id].task_done()

                    # Notify about state change
                    self._notify_state_change()
        except asyncio.CancelledError:
            logger.info(
                f"Episode queue worker for group_id [highlight]{group_id}[/highlight] was [success]cancelled[/success]"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in queue worker for group_id [highlight]{group_id}"
                f"[/highlight]: [danger]{str(e)}[/danger]"
            )
        finally:
            self.queue_workers[group_id] = False
            logger.info(
                f"[success]Stopped[/success] episode queue worker for group_id: [highlight]{group_id}[/highlight]"
            )

    def cancel_all_workers(self) -> int:
        """Cancel all active queue worker tasks.

        Returns:
            int: Number of worker tasks that were canceled
        """
        canceled_count = 0
        for task in asyncio.all_tasks():
            # Worker tasks are named with the prefix 'queue_worker_'
            # followed by the group_id
            if task.get_name().startswith("queue_worker_"):
                task.cancel()
                canceled_count += 1
                # Extract group_id from task name
                group_id = task.get_name()[len("queue_worker_") :]
                if group_id in self.queue_workers:
                    self.queue_workers[group_id] = False

                # Log cancellation
                logger.info(f"[task]Cancelling queue worker task {task.get_name()} - [success]Good![/success][/task]")

        # Notify callbacks about state change
        self._notify_state_change()

        return canceled_count
