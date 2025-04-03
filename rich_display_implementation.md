# Rich Display Implementation Plan for Graphiti MCP Server

## Overview

This document outlines the implementation plan for enhancing the Graphiti MCP Server's terminal display using the Rich library. The implementation will provide real-time status updates, queue metrics, and operation logging with a clean, organized interface.

## Components

### 1. GraphitiDisplayManager Class

The central class managing all display components and updates.

```python
class GraphitiDisplayManager:
    def __init__(self):
        self.console = Console()
        self.live = Live(
            self.generate_display(),
            console=self.console,
            refresh_per_second=4,
            vertical_overflow="visible",
        )
        self.server_status = self.create_server_status_table()
        self._queue_table = Table(show_header=False, box=None)
        self._queue_table.add_column("Metric", style="cyan")
        self._queue_table.add_column("Value", justify="right")
        self.queue_status = Panel(
            self._queue_table,
            title="[bold cyan]Queue Status",
            border_style="cyan"
        )
        self.operation_log = self.create_operation_log_table()
```

### 2. Display Layout

Organized vertical layout with three main sections:

```python
def generate_display(self) -> Layout:
    layout = Layout()
    layout.split_column(
        self.server_status,
        self.queue_status,
        self.operation_log
    )
    return layout
```

### 3. Server Status Table

Real-time component status display:

```python
def create_server_status_table(self) -> Table:
    table = Table(
        title="[bold cyan]Graphiti MCP Server Status",
        show_header=True
    )
    table.add_column("Component", style="blue")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")
    return table

def update_server_status(self, component: str, status: bool, details: str):
    status_icon = "[green]✓[/]" if status else "[red]✗[/]"
    self.server_status.add_row(component, status_icon, details)
```

### 4. Queue Metrics Panel

Live queue statistics:

```python
def update_queue_metrics(self, metrics: dict):
    self._queue_table.rows.clear()
    self._queue_table.add_row(
        "Active Episodes",
        f"[bold blue]{metrics['active']}[/]"
    )
    self._queue_table.add_row(
        "Completed",
        f"[bold green]{metrics['completed']}[/]"
    )
    self._queue_table.add_row(
        "Processing Time",
        f"[yellow]{metrics['avg_time']:.2f}s[/]"
    )
```

### 5. Operation Log Table

Timestamped operation tracking:

```python
def create_operation_log_table(self) -> Table:
    table = Table(
        title="[bold cyan]Recent Operations",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Time", style="dim", width=10)
    table.add_column("Type", width=12)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Description")
    return table

def add_operation_log(
    self,
    op_type: str,
    status: str,
    description: str,
    error: bool = False
):
    time_str = datetime.now().strftime("%H:%M:%S")
    status_style = "red" if error else "green"
    self.operation_log.add_row(
        time_str,
        f"[blue]{op_type}[/]",
        f"[{status_style}]{status}[/]",
        description
    )
```

## Integration Points

### 1. Server Initialization

```python
# In run_mcp_server():
display_manager = GraphitiDisplayManager()
display_manager.live.start()

# Initial status
display_manager.update_server_status("Server", True, "MCP Server initialized")
display_manager.update_server_status("Neo4j", False, "Connecting...")
display_manager.update_server_status("LLM Client", False, "Initializing...")
display_manager.update_queue_metrics({
    "active": 0,
    "completed": 0,
    "avg_time": 0.0
})
```

### 2. Episode Queue Processing

```python
# In process_episode_queue():
display_manager.update_queue_metrics({
    "active": len(episode_queues[group_id]._queue),
    "completed": completed_count,
    "avg_time": avg_processing_time
})
```

### 3. Operation Logging

```python
# For successful operations:
display_manager.add_operation_log(
    "Episode Add",
    "Success",
    f"Added episode: {name}"
)

# For errors:
display_manager.add_operation_log(
    "Neo4j",
    "Error",
    str(error),
    error=True
)
```

## Required Imports

```python
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from datetime import datetime
from typing import Optional
```

## Style Constants

```python
COLORS = {
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "info": "blue",
    "processing": "cyan"
}

ICONS = {
    "success": "✓",
    "error": "✗",
    "warning": "⚠",
    "info": "ℹ",
    "processing": "⋯"
}
```

## Implementation Steps

1. **Code Modularization**

    - Move display logic to separate module
    - Create display manager factory
    - Add configuration options

2. **Integration Points**

    - Add display manager to server initialization
    - Update queue processing to use display
    - Integrate operation logging

3. **Error Handling**

    - Add error recovery for display updates
    - Implement graceful degradation
    - Add display cleanup on shutdown

4. **Testing**
    - Add unit tests for display components
    - Test display updates under load
    - Verify error handling

## Future Enhancements

1. **Customization Options**

    - Configurable refresh rates
    - Custom color schemes
    - Layout adjustments

2. **Additional Features**

    - Memory usage monitoring
    - Performance metrics
    - Interactive components

3. **Performance Optimizations**
    - Batch updates
    - Selective refreshing
    - Buffer management
