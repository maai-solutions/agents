import json
from typing import Dict, List
from agentslm.core.models.state import SharedState, StateEntry

from loguru import logger

class StateManager:
    """Advanced state manager with versioning and rollback."""

    def __init__(self, state: SharedState):
        """Initialize state manager."""
        self.current_state = state or SharedState()
        self._snapshots: List[Dict[str, StateEntry]] = []

    def create_snapshot(self) -> int:
        """Create a snapshot of current state.

        Returns:
            Snapshot ID
        """
        snapshot = self.current_state._state.copy()
        self._snapshots.append(snapshot)
        snapshot_id = len(self._snapshots) - 1
        logger.info(f"[STATE] Created snapshot #{snapshot_id}")
        return snapshot_id

    def restore_snapshot(self, snapshot_id: int) -> None:
        """Restore state from snapshot.

        Args:
            snapshot_id: Snapshot ID to restore
        """
        if 0 <= snapshot_id < len(self._snapshots):
            self.current_state._state = self._snapshots[snapshot_id].copy()
            logger.info(f"[STATE] Restored snapshot #{snapshot_id}")
        else:
            raise ValueError(f"Invalid snapshot ID: {snapshot_id}")

    def list_snapshots(self) -> List[int]:
        """List available snapshot IDs.

        Returns:
            List of snapshot IDs
        """
        return list(range(len(self._snapshots)))

    def export_json(self, filepath: str) -> None:
        """Export state to JSON file.

        Args:
            filepath: Path to JSON file
        """
        data = self.current_state.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"[STATE] Exported to {filepath}")

    def import_json(self, filepath: str) -> None:
        """Import state from JSON file.

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.current_state.clear()
        for key, entry_dict in data.items():
            self.current_state.set(
                key=entry_dict['key'],
                value=entry_dict['value'],
                source=entry_dict.get('source'),
                metadata=entry_dict.get('metadata', {})
            )
        logger.info(f"[STATE] Imported from {filepath}")