"""Memory management for agents with multiple storage backends."""

from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from datetime import datetime
from collections import deque
import tiktoken
from loguru import logger


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

    def __init__(self, max_size: Optional[int] = None):
        """Initialize in-memory backend.

        Args:
            max_size: Maximum number of entries to keep (None for unlimited)
        """
        self.max_size = max_size
        self.memories: deque = deque(maxlen=max_size)

    def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry."""
        self.memories.append(entry)
        logger.debug(f"[MEMORY] Added entry: {entry.content[:50]}...")

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
        logger.info("[MEMORY] Cleared all memories")

    def get_all(self) -> List[MemoryEntry]:
        """Get all memories."""
        return list(self.memories)

    def count(self) -> int:
        """Count total memories."""
        return len(self.memories)


class VectorStoreBackend(MemoryBackend):
    """Vector store backend for semantic search (stub for now)."""

    def __init__(self, embedding_model: Optional[Any] = None):
        """Initialize vector store backend.

        Args:
            embedding_model: Model for generating embeddings
        """
        self.embedding_model = embedding_model
        self.memories: List[MemoryEntry] = []
        self.embeddings: List[Any] = []
        logger.warning("[MEMORY] VectorStoreBackend is a stub - using simple storage")

    def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry."""
        self.memories.append(entry)
        # TODO: Generate and store embedding
        logger.debug(f"[MEMORY] Added entry to vector store: {entry.content[:50]}...")

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get recent memory entries."""
        return sorted(self.memories, key=lambda e: e.timestamp, reverse=True)[:limit]

    def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Semantic search (currently keyword-based)."""
        # TODO: Implement actual vector similarity search
        query_lower = query.lower()
        matches = [
            entry for entry in self.memories
            if query_lower in entry.content.lower()
        ]
        matches.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
        return matches[:limit]

    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()
        self.embeddings.clear()
        logger.info("[MEMORY] Cleared vector store")

    def get_all(self) -> List[MemoryEntry]:
        """Get all memories."""
        return self.memories

    def count(self) -> int:
        """Count total memories."""
        return len(self.memories)


class MemoryManager:
    """Manages agent memory with token-aware context window management."""

    def __init__(
        self,
        backend: MemoryBackend,
        max_context_tokens: int = 4096,
        summary_threshold_tokens: int = 2048,
        llm: Optional[Any] = None,
        encoding_name: str = "cl100k_base"
    ):
        """Initialize memory manager.

        Args:
            backend: Memory storage backend
            max_context_tokens: Maximum tokens for context window
            summary_threshold_tokens: When to trigger summarization
            llm: Language model for summarization (optional)
            encoding_name: Tokenizer encoding to use
        """
        self.backend = backend
        self.max_context_tokens = max_context_tokens
        self.summary_threshold_tokens = summary_threshold_tokens
        self.llm = llm
        self.summary: Optional[str] = None

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer {encoding_name}: {e}")
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: rough estimation
            return len(text) // 4

    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        entry_type: str = "interaction"
    ) -> None:
        """Add a new memory.

        Args:
            content: Memory content
            metadata: Optional metadata
            importance: Importance score (0.0 to 1.0)
            entry_type: Type of memory entry
        """
        entry = MemoryEntry(
            content=content,
            metadata=metadata or {},
            importance=importance,
            entry_type=entry_type
        )
        self.backend.add(entry)

        # Check if we need to summarize
        if self._should_summarize():
            self._create_summary()

    def _should_summarize(self) -> bool:
        """Check if summarization is needed based on token count."""
        total_tokens = sum(
            self.count_tokens(entry.content)
            for entry in self.backend.get_all()
        )
        return total_tokens > self.summary_threshold_tokens

    def _create_summary(self) -> None:
        """Create a summary of older memories to save tokens."""
        if not self.llm:
            logger.warning("[MEMORY] No LLM provided for summarization")
            return

        all_memories = self.backend.get_all()
        if len(all_memories) < 5:
            return  # Not enough to summarize

        # Keep recent memories, summarize older ones
        recent_count = min(10, len(all_memories) // 2)
        memories_to_summarize = all_memories[:-recent_count]

        if not memories_to_summarize:
            return

        # Create summary
        memory_text = "\n".join([
            f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M')}] {entry.content}"
            for entry in memories_to_summarize
        ])

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content="Summarize the following conversation history concisely, preserving key information and context."),
                HumanMessage(content=f"Conversation history:\n{memory_text}\n\nProvide a concise summary:")
            ]

            response = self.llm.invoke(messages)
            self.summary = response.content

            logger.info(f"[MEMORY] Created summary of {len(memories_to_summarize)} memories")
            logger.debug(f"[MEMORY] Summary: {self.summary[:200]}...")

        except Exception as e:
            logger.error(f"[MEMORY] Failed to create summary: {e}")

    def get_context(
        self,
        max_tokens: Optional[int] = None,
        include_summary: bool = True,
        query: Optional[str] = None
    ) -> str:
        """Get memory context that fits within token limit.

        Args:
            max_tokens: Maximum tokens (uses max_context_tokens if None)
            include_summary: Whether to include summary if available
            query: Optional query for semantic search

        Returns:
            Formatted context string
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens

        context_parts = []
        current_tokens = 0

        # Add summary if available and requested
        if include_summary and self.summary:
            summary_text = f"=== Summary of Earlier Conversation ===\n{self.summary}\n\n"
            summary_tokens = self.count_tokens(summary_text)
            if summary_tokens < max_tokens * 0.3:  # Use max 30% for summary
                context_parts.append(summary_text)
                current_tokens += summary_tokens

        # Get relevant memories
        if query:
            # Semantic search for relevant memories
            memories = self.backend.search(query, limit=20)
        else:
            # Get recent memories
            memories = self.backend.get_recent(limit=50)

        # Add memories until we hit token limit
        context_parts.append("=== Recent Context ===\n")

        for entry in reversed(memories):  # Most recent last
            entry_text = f"[{entry.timestamp.strftime('%H:%M:%S')}] {entry.content}\n"
            entry_tokens = self.count_tokens(entry_text)

            if current_tokens + entry_tokens > max_tokens:
                break

            context_parts.append(entry_text)
            current_tokens += entry_tokens

        context = "".join(context_parts)
        logger.debug(f"[MEMORY] Generated context with {current_tokens} tokens")

        return context

    def search_memories(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Search for relevant memories.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching memory entries
        """
        return self.backend.search(query, limit)

    def clear_memory(self) -> None:
        """Clear all memories and summary."""
        self.backend.clear()
        self.summary = None
        logger.info("[MEMORY] Cleared all memories")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        all_memories = self.backend.get_all()
        total_tokens = sum(self.count_tokens(entry.content) for entry in all_memories)

        return {
            "total_memories": self.backend.count(),
            "total_tokens": total_tokens,
            "has_summary": self.summary is not None,
            "summary_tokens": self.count_tokens(self.summary) if self.summary else 0,
            "max_context_tokens": self.max_context_tokens,
            "utilization": total_tokens / self.max_context_tokens if self.max_context_tokens > 0 else 0
        }

    def export_memories(self) -> List[Dict[str, Any]]:
        """Export all memories to JSON-serializable format.

        Returns:
            List of memory dictionaries
        """
        return [entry.to_dict() for entry in self.backend.get_all()]

    def import_memories(self, memories: List[Dict[str, Any]]) -> None:
        """Import memories from JSON-serializable format.

        Args:
            memories: List of memory dictionaries
        """
        for mem_dict in memories:
            entry = MemoryEntry.from_dict(mem_dict)
            self.backend.add(entry)
        logger.info(f"[MEMORY] Imported {len(memories)} memories")


def create_memory_manager(
    backend_type: str = "in_memory",
    max_context_tokens: int = 4096,
    summary_threshold_tokens: int = 2048,
    llm: Optional[Any] = None,
    max_size: Optional[int] = None
) -> MemoryManager:
    """Factory function to create a memory manager.

    Args:
        backend_type: Type of backend ("in_memory" or "vector_store")
        max_context_tokens: Maximum tokens for context window
        summary_threshold_tokens: When to trigger summarization
        llm: Language model for summarization
        max_size: Maximum number of memories (for in_memory backend)

    Returns:
        Configured MemoryManager instance
    """
    if backend_type == "in_memory":
        backend = InMemoryBackend(max_size=max_size)
    elif backend_type == "vector_store":
        backend = VectorStoreBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    return MemoryManager(
        backend=backend,
        max_context_tokens=max_context_tokens,
        summary_threshold_tokens=summary_threshold_tokens,
        llm=llm
    )
