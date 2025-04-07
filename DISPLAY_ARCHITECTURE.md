# MemCP Core Systems: Clean Architecture Plan

## Overall System Design Philosophy

The new architecture should follow these core principles:

-   **Single Responsibility**: Each component should do one thing and do it well
-   **Dependency Inversion**: High-level modules should not depend on low-level modules
-   **Interface Segregation**: Clients shouldn't depend on methods they don't use
-   **Event-Driven Communication**: Use events/observers for loose coupling
-   **Testability**: All components should be easily testable in isolation

## Phase 1: Queue Management System Design

### Prompt for Implementation

```
Design a clean, event-driven queue management system for our MemCP project following these requirements:

1. Create a unified QueueManager class that:
   - Manages multiple queues identified by group_id
   - Handles async task enqueuing and processing
   - Tracks queue statistics internally (tasks added, completed, failed, in progress)
   - Calculates metrics like average processing time
   - Provides a clean public API for adding tasks and retrieving statistics
   - Implements a simple event notification system
   - Does NOT contain any display or logging logic

2. Implement an event system using:
   - A QueueEvent class with event types (TASK_ADDED, TASK_STARTED, TASK_COMPLETED, etc.)
   - A simple observer pattern where listeners can subscribe to queue events
   - Event notifications include all necessary statistics and metadata

This system should be completely decoupled from display or shutdown logic. Communication should happen through well-defined events. Include proper error handling, graceful failure modes, and ensure all operations are thread-safe.

The key is simplicity - this should be a focused component that tracks processes going in, releases them when done, and notifies watchers when things change.

Show me the code with appropriate type hints, docstrings, and test cases.
```

### Key Components

1. **QueueEvent**

    - Represents queue-related events with type, metadata, and timestamp
    - Types include: TASK_ADDED, TASK_STARTED, TASK_COMPLETED, TASK_FAILED
    - Contains relevant statistics snapshot when emitted

2. **QueueManager**
    - Manages async task processing queue per group_id
    - Tracks all queue statistics internally
    - Provides methods to subscribe/unsubscribe to events
    - Notifies listeners when queue state changes
    - Handles task creation, scheduling, and execution
    - Supports graceful cancellation of tasks

## Phase 2: Graceful Shutdown System

### Prompt for Implementation

```
Design a robust shutdown system for our MemCP project that handles graceful termination and cleanup:

1. Create a ShutdownManager class that:
   - Listens for shutdown signals (SIGINT, SIGTERM, SIGQUIT)
   - Supports multiple shutdown modes (graceful, force)
   - Performs ordered resource cleanup
   - Tracks shutdown progress with clear status updates
   - Emits shutdown events that can be consumed by display components
   - Does NOT directly depend on QueueManager implementation details
   - Ensures 1x Ctrl+C triggers graceful shutdown, 3x triggers forced shutdown

2. Implement signal handlers that:
   - Convert OS signals to appropriate shutdown actions
   - Track multiple interrupt signals (counting Ctrl+C presses)
   - Perform appropriate actions based on signal type and count
   - Handle terminal cleanup properly

The shutdown system should wait for in-progress queue tasks to complete during graceful shutdown, cancel remaining tasks after a timeout, and properly close all resources including database connections.

Ensure proper ordering of shutdown operations: stop accepting new tasks → wait for in-progress tasks → close database connections → notify display systems → exit.

Apply clean architecture principles with events for loose coupling. Handle edge cases like shutdown during startup or multiple shutdown attempts.
```

### Key Components

1. **ShutdownManager**

    - Manages the shutdown lifecycle
    - Tracks shutdown state (not started, in progress, complete)
    - Coordinates resource cleanup
    - Handles different shutdown modes (graceful vs. force)
    - Counts interrupt signals for handling multiple Ctrl+C presses
    - Emits events about shutdown progress

2. **Signal Handlers**
    - Convert OS signals to shutdown events
    - Handle multiple interrupt signals
    - Perform appropriate actions based on signal type

### Shutdown Sequence

1. Receive shutdown signal (track Ctrl+C count)
2. Enter shutdown mode (prevent new operations)
3. Stop accepting new queue tasks
4. Wait for in-progress tasks to complete (with timeout)
5. Close database connections
6. Emit events indicating shutdown progress
7. Exit with appropriate status code

## Phase 3: Rich Console Display System

### Prompt for Implementation

```
Create a terminal display system for our MemCP project using the Rich library that:

1. Implement a DisplayManager class that:
   - Uses Rich's Live context manager for dynamic updates
   - Creates and manages all console output
   - Listens for both queue events and shutdown events
   - Creates a layout with a persistent panel at the bottom of the terminal
   - Has NO direct dependencies on QueueManager or ShutdownManager implementations

2. Implement a QueueProgressDisplay component within DisplayManager that:
   - Subscribes to QueueEvent notifications
   - Creates progress bars for active queues
   - Updates in real-time as queue statistics change
   - Formats output using our existing GRAPHITI_THEME

3. Implement a StatusDisplay component within DisplayManager that:
   - Shows current server information (host, port, PID)
   - Updates with operational status (running, error states)
   - Uses Rich's Status with spinner for active operations

4. Implement a ShutdownProgressDisplay within DisplayManager that:
   - Shows shutdown steps in real-time when shutdown is triggered
   - Uses Rich's Panel and Table to display shutdown progress
   - Color-codes success/warning/error states using our theme

Make sure the display refreshes smoothly using Rich's best practices. Use Rich's Panel class for visual organization and ensure all components stay at the bottom of the terminal. The display should handle terminal resize events gracefully.

For reference, the Rich Live class expects a renderable object that it will refresh, and works best as a context manager. The panels should use our application theme colors defined in GRAPHITI_THEME.

Show complete implementation with proper error handling and clean architecture principles.
```

### Key Components

1. **DisplayManager**
    - Central coordinator for all visual elements
    - Manages Rich Console instances for different purposes
    - Handles Live context for dynamic updates
    - Contains and coordinates all display components:
        - QueueProgressDisplay
        - StatusDisplay
        - ShutdownProgressDisplay
    - Listens for events from both queue and shutdown systems

### Rich Library Best Practices

-   Use Live context manager for dynamic updates which persists for the context duration
-   Set the refresh rate with `refresh_per_second` for optimal performance
-   Progress bars work best as context managers
-   Use Rich's Status class with context managers to track resources
-   Consider using `console.capture()` for testing console output

## Phase 4: Integration and Coordination

### Prompt for Implementation

```
Create the integration layer for our queue, display, and shutdown systems following these requirements:

1. Design an ApplicationCoordinator class that:
   - Initializes and connects all components
   - Sets up event listeners between components
   - Handles startup sequence
   - Maintains no business logic itself

2. Implement the main application entry point that:
   - Creates the coordinator
   - Handles configuration loading
   - Sets up signal handlers
   - Starts the application gracefully
   - Provides clear error messages for startup failures

3. Write the necessary glue code to:
   - Connect QueueManager event notifications to DisplayManager
   - Connect ShutdownManager event notifications to DisplayManager
   - Ensure all components work together without direct dependencies

This system should maintain clean architecture with:
- Clear separation of concerns
- Dependency injection for all components
- No circular dependencies
- Event-based communication between systems

Use Rich's Console for any startup messages outside the Live display.

Important: All display logic should depend only on events and interfaces, never on queue or shutdown implementation details. Similarly, shutdown logic should work with queue abstractions, not implementations.

Show the complete implementation including application entry point and proper error handling.
```

### Key Components

1. **ApplicationCoordinator**

    - Wires together all components
    - Manages application lifecycle
    - Sets up event subscriptions
    - No business logic, just coordination

2. **Application Entry Point**
    - Initializes configuration
    - Creates and starts coordinator
    - Sets up error handling
    - Clean startup sequence

### Integration Architecture

```
┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │
│  QueueManager   │      │ ShutdownManager │
│   (Notifier)    │      │   (Notifier)    │
│                 │      │                 │
└────────┬────────┘      └────────┬────────┘
         │                        │
         │ Events                 │ Events
         │                        │
         ▼                        ▼
┌───────────────────────────────────────────┐
│                                           │
│              DisplayManager               │
│          (Listens to all events)          │
│                                           │
└───────────────────────────────────────────┘
                    ▲
                    │
                    │ Coordinates
                    │
┌───────────────────────────────────────────┐
│                                           │
│          ApplicationCoordinator           │
│                                           │
└───────────────────────────────────────────┘
```

## Phase 5: Testing and Documentation

### Prompt for Implementation

```
Create comprehensive tests and documentation for our new systems:

1. Write unit tests for:
   - QueueManager and task processing
   - Event publishing and subscription
   - DisplayManager rendering
   - ShutdownManager sequence handling

2. Write integration tests for:
   - Queue-to-display communication
   - Shutdown sequence verification
   - Edge cases like shutdown during queue processing

3. Create documentation including:
   - Architecture overview diagram
   - Component interaction diagrams
   - API documentation for all public interfaces
   - Usage examples for each subsystem

4. Include docstrings that:
   - Explain each class's responsibility
   - Document method parameters and return values
   - Provide usage examples
   - Note any threading or async considerations

The tests should use mocks to isolate components, and the documentation should emphasize the clean architecture principles used throughout the system.

Show me the test suite and documentation structure.
```

### Testing Strategy

1. **Unit Tests**

    - Test each component in isolation
    - Mock dependencies and verify interactions
    - Test edge cases and error conditions
    - Verify event emissions

2. **Integration Tests**

    - Verify components work together correctly
    - Test event propagation between systems
    - Simulate real-world scenarios
    - Test shutdown sequences

3. **Mock Objects**
    - Create mock implementations of interfaces
    - Simulate events and responses
    - Track interaction counts and parameters
    - Verify correct subscription behavior

### Documentation Structure

1. **Architecture Overview**

    - System-level diagrams
    - Component responsibility descriptions
    - Event flow diagrams
    - Design principles explanation

2. **API Documentation**

    - Interface specifications
    - Class hierarchies
    - Method signatures and descriptions
    - Usage examples

3. **Developer Guide**
    - How to extend the system
    - Common patterns and idioms
    - Testing approaches
    - Performance considerations

## Implementation Phases

1. **Phase 1: Core Queue System**

    - Implement simple event notification system
    - Build unified QueueManager with internal statistics tracking
    - Write unit tests for queue functionality

2. **Phase 2: Shutdown Functionality**

    - Implement ShutdownManager
    - Build signal handlers with Ctrl+C counting
    - Test shutdown sequences and resource cleanup

3. **Phase 3: Display System**

    - Implement DisplayManager with Rich
    - Create queue progress visualization
    - Build server status display
    - Add shutdown progress visualization
    - Connect to queue and shutdown events

4. **Phase 4: Integration**

    - Build ApplicationCoordinator
    - Wire all components together
    - Implement main entry point
    - End-to-end testing

5. **Phase 5: Documentation and Polish**
    - Complete API documentation
    - Create architecture diagrams
    - Write developer guides
    - Performance optimization

## Terminal Display Considerations

-   Rich can auto-detect terminals and will strip control codes when not writing to one
-   Some CI systems support color but not cursor movement - handle this case
-   Use `screen()` context manager for full-screen applications
-   Ensure display handles terminal resize events gracefully
-   Consider accessibility for color-blind users with alternate indicators

## Error Handling Patterns

-   Queue errors should be captured and reported, not crash the application
-   Display errors should degrade gracefully (fall back to simpler display)
-   Shutdown errors should be logged but still attempt to exit cleanly
-   Use structured error types for better error handling
-   Provide useful error messages for debugging

---

This plan creates a robust foundation following clean architecture principles, maintaining clear separation between business logic, presentation, and application lifecycle management, while leveraging the Rich library's powerful features for creating engaging terminal interfaces.
