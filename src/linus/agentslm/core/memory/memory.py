from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, List, Optional

from loguru import logger

from agentslm.core.interfaces.logger import ILogger, NoOpLogger


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0  # 0.0 to 1.0, for prioritization
    entry_type: str = "interaction"  # interaction, observation, thought, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "importance": self.importance,
            "entry_type": self.entry_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 1.0),
            entry_type=data.get("entry_type", "interaction")
        )
    
class MemoryBackend(ABC):
    """Abstract base class for memory backends."""

    @abstractmethod
    def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry."""
        pass

    @abstractmethod
    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get recent memory entries."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Search for relevant memories."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories."""
        pass

    @abstractmethod
    def get_all(self) -> List[MemoryEntry]:
        """Get all memories."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total memories."""
        pass


class InMemoryBackend(MemoryBackend):
    """Simple in-memory storage using a deque."""

    def __init__(
            self, 
            max_size: Optional[int] = None,
            logger: ILogger = NoOpLogger()
        ):
        """Initialize in-memory backend.

        Args:
            max_size: Maximum number of entries to keep (None for unlimited)
        """
        self.max_size = max_size
        self.memories: deque = deque(maxlen=max_size)
        self.logger = logger
        

    def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry."""
        self.memories.append(entry)
        self.logger.debug(f"[MEMORY] Added entry: {entry.content[:50]}...")

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get recent memory entries."""
        return list(self.memories)[-limit:]

    def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Simple keyword-based search."""
        query_lower = query.lower()
        matches = [
            entry for entry in self.memories
            if query_lower in entry.content.lower()
        ]
        # Sort by importance and recency
        matches.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
        return matches[:limit]

    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()
        self.logger.info("[MEMORY] Cleared all memories")

    def get_all(self) -> List[MemoryEntry]:
        """Get all memories."""
        return list(self.memories)

    def count(self) -> int:
        """Count total memories."""
        return len(self.memories)
