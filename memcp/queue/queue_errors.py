"""Queue related errors."""


class QueueError(Exception):
    """Base exception for queue operation errors."""

    pass


class QueueFullError(QueueError):
    """Raised when a queue is at capacity."""

    pass


class QueueProcessingError(QueueError):
    """Raised when there's an error processing a queued task."""

    pass


# class MemCPConnectionError(MemCPError):
#     """Base exception for connection-related errors."""

#     pass
