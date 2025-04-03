#!/usr/bin/env python3
"""Example of using the Queue Progress Display system independently."""

from queue_progress_display import QueueProgressDisplay
from queue_stats import QueueStatsTracker

import asyncio
import random
import uuid

from rich.console import Console


async def simulate_task(
    queue_id: str,
    task_id: str,
    tracker: QueueStatsTracker,
    display: QueueProgressDisplay,
    min_duration: float = 0.5,
    max_duration: float = 3.0,
):
    """Simulate a task processing with random duration."""
    # Start processing
    tracker.start_processing(queue_id, task_id)
    display.update()

    # Simulate work
    duration = random.uniform(min_duration, max_duration)
    await asyncio.sleep(duration)

    # 90% chance of success
    success = random.random() < 0.9

    # Complete task
    tracker.complete_task(queue_id, task_id, success)
    display.update()

    return duration, success


async def simulate_queue(
    queue_id: str,
    num_tasks: int,
    tracker: QueueStatsTracker,
    display: QueueProgressDisplay,
):
    """Simulate a queue of tasks being processed."""
    # Add tasks to the queue
    for _ in range(num_tasks):
        tracker.add_task(queue_id)

    display.update()

    # Process tasks with some concurrency (3 at a time)
    tasks = []
    for _ in range(num_tasks):
        task_id = str(uuid.uuid4())
        tasks.append(asyncio.create_task(simulate_task(queue_id, task_id, tracker, display)))

        # Limit concurrency to 3 tasks at a time
        if len(tasks) >= 3:
            # Wait for one task to complete before adding more
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            # Remove completed tasks
            tasks = [t for t in tasks if not t.done()]

    # Wait for remaining tasks
    if tasks:
        await asyncio.wait(tasks)


async def main():
    """Run the queue progress display example."""
    # Create console, tracker and display
    console = Console()
    tracker = QueueStatsTracker()
    display = QueueProgressDisplay(console, tracker)

    # Start the display
    display.start()

    try:
        # Simulate multiple queues with different numbers of tasks
        queues = [
            ("project_docs", 5),
            ("data_processing", 15),
            ("api_requests", 8),
        ]

        # Process queues concurrently
        await asyncio.gather(
            *(
                simulate_queue(queue_id, num_tasks, tracker, display)
                for queue_id, num_tasks in queues
            )
        )

        # Show all completed
        console.print("\n[bold green]All tasks completed![/bold green]")

        # Display final statistics
        console.print(display.generate_status_table())

    finally:
        # Make sure to stop the display
        display.stop()


if __name__ == "__main__":
    # Print header
    print("Queue Progress Display Example")
    print("-----------------------------")
    print("This example demonstrates the queue progress display system.")
    print("It simulates processing tasks in multiple queues concurrently.")
    print("\n")

    # Run the example
    asyncio.run(main())
