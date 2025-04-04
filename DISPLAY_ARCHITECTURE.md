# MemCP Display Architecture

This document outlines the display architecture of the MemCP application, focusing on how the console UI is rendered and updated, with particular attention to the Queue Progress display component which is experiencing a stacking issue.

## Components Overview

The display system consists of several key components that work together:

1. **DisplayManager**: Central manager for all console displays
2. **QueueProgressDisplay**: Specialized component for showing queue processing progress
3. **QueueManager**: Manages background tasks and notifies display when queue state changes
4. **QueueStatsTracker**: Tracks statistics for processing queues
5. **GraphitiServer**: Main server that coordinates all these components
6. **Rich.Live**: Underlying Rich component used for live display updates

## Initialization Flow

The display initialization happens in the following sequence:

```
AppFactory                         GraphitiServer                   DisplayManager                    QueueProgressDisplay
    |                                   |                                |                                   |
    |-- create_server() -----------→   |                                |                                   |
    |   - Creates DisplayManager        |                                |                                   |
    |   - Creates QueueStatsTracker     |                                |                                   |
    |   - Creates QueueManager          |                                |                                   |
    |   - Creates QueueProgressDisplay  |                                |                                   |
    |   - Creates GraphitiServer        |                                |                                   |
    |                                   |                                |                                   |
    |                                   |-- initialize_queue_components()|                                   |
    |                                   |   - Registers display callback |                                   |
    |                                   |                                |                                   |
    |                                   |-- run() ------------------→   |                                   |
    |                                   |                                |                                   |
    |                                   |                                |-- initialize_display() ------→   |-- initialize_tasks()
    |                                   |                                |                                   |   - Creates progress tasks
    |                                   |                                |                                   |   - Sets initialized=True
    |                                   |                                |                                   |
    |                                   |                                |-- start_live_display() -----→    |
    |                                   |                                |   - Creates Live context          |
    |                                   |                                |   - Starts refresh task           |
    |                                   |                                |                                   |
```

## Update Flow

The queue progress display is updated whenever the queue state changes:

```
QueueManager                      QueueStatsTracker              QueueProgressDisplay           DisplayManager
    |                                 |                                 |                            |
    |-- enqueue_task() ------→       |-- add_task() -------→          |                            |
    |   - Adds task to queue          |   - Adds task to stats         |                            |
    |                                 |                                 |                            |
    |-- _notify_state_change() --→   |                                 |-- update() -----→         |
    |   - Calls callback               |                                |   - Updates tasks         |
    |                                 |                                 |   - Updates panel         |
    |                                 |                                 |                            |
    |-- process_episode_queue() --→  |-- start_processing() -→        |                            |
    |   - Processes next task         |   - Updates processing stats    |                            |
    |                                 |                                 |                            |
    |-- _notify_state_change() --→   |                                 |-- update() -----→         |
    |   - Calls callback               |                                |   - Updates tasks         |
    |                                 |                                 |   - Updates panel         |
    |                                 |                                 |                            |
    |                                 |                                 |                            |-- refresh_display_periodically()
    |                                 |                                 |                            |   - Updates layout with panel
    |                                 |                                 |                            |   - Calls live.refresh()
```

## Rich.Live and Layout Interaction

The key area where issues might occur is in how the Live context manager and Layout interact:

### Current Implementation

1. **DisplayManager.initialize_display()**:

    ```python
    def initialize_display(self, queue_progress_display, status_obj):
        # Update the layout sections
        if queue_progress_display is not None:
            self.layout["progress"].update(queue_progress_display.panel)
        self.layout["status"].update(status_obj)
        return self.layout
    ```

2. **DisplayManager.refresh_display_periodically()**:

    ```python
    async def refresh_display_periodically(self, queue_progress_display, status_obj, interval):
        # ...
        if queue_progress_display is not None:
            self.layout["progress"].update(queue_progress_display.panel)
        self.layout["status"].update(status_obj)
        self.live.refresh()
        # ...
    ```

3. **QueueProgressDisplay.update()**:
    ```python
    def update(self):
        if not self.initialized:
            self.initialize_tasks()
        else:
            self.update_tasks()
        # We don't need to update self.panel because it's a reference to the
        # same panel object that contains the continuously updated Progress object
    ```

## The Issue: Panel Stacking

The core issue appears to be in how Rich's Layout.update() method handles panels. When we call:

```python
self.layout["progress"].update(queue_progress_display.panel)
```

Instead of replacing the existing content, it may be adding the panel as a child to the layout node, resulting in stacked panels.

## Root Cause Analysis

Based on the Rich documentation and the behavior shown in the screenshot:

1. The `Layout.update()` method in Rich doesn't replace content by default, it may add to it
2. Each time the periodic refresh runs, it calls `.update()` with the same panel object
3. This results in multiple copies of the same panel being rendered

## Potential Fixes

1. **Use direct renderable assignment**:

    ```python
    self.layout["progress"].renderable = queue_progress_display.panel
    ```

2. **Clear the layout before updating**:

    ```python
    self.layout["progress"].update(None)  # Clear first
    self.layout["progress"].update(queue_progress_display.panel)  # Then update
    ```

3. **Use the Live.update() method directly**:
    ```python
    # Update the entire layout every refresh
    self.live.update(self.layout)
    ```

## Resolution

The simplest fix is likely to modify the `refresh_display_periodically` method to use direct renderable assignment instead of the update method:

```python
if queue_progress_display is not None:
    self.layout["progress"].renderable = queue_progress_display.panel
self.layout["status"].renderable = status_obj
```

This ensures that we're replacing the content each time, rather than potentially adding to it.
