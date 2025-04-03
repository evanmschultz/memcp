"""Adapter module for backward compatibility with old_graphiti_mcp_server.py.

This module re-exports the queue classes from their new locations to maintain
backward compatibility with the old monolithic script. This is a temporary
solution until the old script is fully migrated to the new structure.
"""

# Re-export the classes from their new locations
from memcp.console.queue_display import QueueProgressDisplay
from memcp.core.queue import QueueManager, QueueStatsTracker


# For backward compatibility
class QueueStatsTrackerCompat(QueueStatsTracker):
    """Backwards compatible QueueStatsTracker."""

    pass


class QueueManagerCompat(QueueManager):
    """Backwards compatible QueueManager."""

    pass


class QueueProgressDisplayCompat(QueueProgressDisplay):
    """Backwards compatible QueueProgressDisplay."""

    pass
