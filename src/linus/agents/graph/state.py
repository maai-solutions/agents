"""Shared state management for agent orchestration."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
from loguru import logger


@dataclass
class StateEntry:
    """Individual state entry with metadata."""
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None  # Which node created this
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata
        }


class SharedState:
    """Shared state that can be accessed by all agents in the DAG."""

    def __init__(self):
        """Initialize shared state."""
        self._state: Dict[str, StateEntry] = {}
        self._history: List[StateEntry] = []

    def set(
        self,
        key: str,
        value: Any,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set a value in the shared state.

        Args:
            key: State key
            value: Value to store
            source: Source node name
            metadata: Optional metadata
        """
        entry = StateEntry(
            key=key,
            value=value,
            source=source,
            metadata=metadata or {}
        )

        self._state[key] = entry
        self._history.append(entry)

        logger.debug(f"[STATE] Set '{key}' = {str(value)[:100]} (source: {source})")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared state.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            Value associated with key or default
        """
        entry = self._state.get(key)
        if entry is None:
            logger.debug(f"[STATE] Get '{key}' = {default} (not found)")
            return default

        logger.debug(f"[STATE] Get '{key}' = {str(entry.value)[:100]}")
        return entry.value

    def has(self, key: str) -> bool:
        """Check if key exists in state.

        Args:
            key: State key

        Returns:
            True if key exists
        """
        return key in self._state

    def delete(self, key: str) -> None:
        """Delete a key from state.

        Args:
            key: State key
        """
        if key in self._state:
            del self._state[key]
            logger.debug(f"[STATE] Deleted '{key}'")

    def get_entry(self, key: str) -> Optional[StateEntry]:
        """Get the full state entry with metadata.

        Args:
            key: State key

        Returns:
            StateEntry or None
        """
        return self._state.get(key)

    def get_all(self) -> Dict[str, Any]:
        """Get all state values as dictionary.

        Returns:
            Dictionary of all state values
        """
        return {key: entry.value for key, entry in self._state.items()}

    def get_history(self, key: Optional[str] = None) -> List[StateEntry]:
        """Get state change history.

        Args:
            key: Optional key to filter history

        Returns:
            List of state entries
        """
        if key is None:
            return self._history.copy()
        return [entry for entry in self._history if entry.key == key]

    def clear(self) -> None:
        """Clear all state."""
        self._state.clear()
        logger.info("[STATE] Cleared all state")

    def to_dict(self) -> Dict[str, Any]:
        """Export state as dictionary.

        Returns:
            Dictionary with all state entries
        """
        return {
            key: entry.to_dict()
            for key, entry in self._state.items()
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"SharedState(entries={len(self._state)})"


class StateManager:
    """Advanced state manager with versioning and rollback."""

    def __init__(self):
        """Initialize state manager."""
        self.current_state = SharedState()
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
