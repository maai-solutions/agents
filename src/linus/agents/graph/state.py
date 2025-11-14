"""Shared state management for agent orchestration with pluggable backends."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import json
from loguru import logger

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available for state context management, using fallback token counting")


class StateContextStrategy(str, Enum):
    """Strategy for managing shared state context in prompts."""
    FULL = "full"  # Include all state (no truncation)
    CLIP = "clip"  # Keep only most recent N entries (by timestamp)
    COMPACT = "compact"  # LLM-based summarization of state


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


class StateBackend(ABC):
    """Abstract base class for state storage backends.

    Similar to MemoryBackend but designed for SharedState architecture.
    Backends handle storage and retrieval logic while SharedState handles
    token counting, summarization, and context formatting.
    """

    @abstractmethod
    def add(self, entry: StateEntry) -> None:
        """Add a state entry to the backend.

        Args:
            entry: StateEntry to store
        """
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[StateEntry]:
        """Get a specific entry by key.

        Args:
            key: State key to retrieve

        Returns:
            StateEntry or None if not found
        """
        pass

    @abstractmethod
    def get_all(self) -> List[StateEntry]:
        """Get all state entries.

        Returns:
            List of all StateEntry objects
        """
        pass

    @abstractmethod
    def get_recent(self, limit: int = 10) -> List[StateEntry]:
        """Get recent state entries sorted by timestamp.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent StateEntry objects
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete an entry by key.

        Args:
            key: State key to delete
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all state entries."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total number of entries.

        Returns:
            Number of entries in backend
        """
        pass


class KeyValueBackend(StateBackend):
    """Key-value storage backend (default SharedState behavior).

    Stores state as key-value pairs with history tracking.
    """

    def __init__(self):
        """Initialize key-value backend."""
        self._state: Dict[str, StateEntry] = {}
        self._history: List[StateEntry] = []

    def add(self, entry: StateEntry) -> None:
        """Add/update a state entry."""
        self._state[entry.key] = entry
        self._history.append(entry)
        logger.debug(f"[KV-BACKEND] Set '{entry.key}' (source: {entry.source})")

    def get(self, key: str) -> Optional[StateEntry]:
        """Get entry by key."""
        return self._state.get(key)

    def get_all(self) -> List[StateEntry]:
        """Get all current state entries."""
        return list(self._state.values())

    def get_recent(self, limit: int = 10) -> List[StateEntry]:
        """Get recent entries from history."""
        return sorted(self._history, key=lambda e: e.timestamp, reverse=True)[:limit]

    def delete(self, key: str) -> None:
        """Delete a key from state."""
        if key in self._state:
            del self._state[key]
            logger.debug(f"[KV-BACKEND] Deleted '{key}'")

    def clear(self) -> None:
        """Clear all state and history."""
        self._state.clear()
        self._history.clear()
        logger.info("[KV-BACKEND] Cleared all state")

    def count(self) -> int:
        """Count entries."""
        return len(self._state)

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


class ConversationMemoryBackend(StateBackend):
    """Conversation memory backend for storing interaction history.

    Replaces the old MemoryManager with a StateBackend implementation.
    Stores conversations as sequential entries (not key-value pairs).
    """

    def __init__(self, max_size: Optional[int] = None):
        """Initialize conversation memory backend.

        Args:
            max_size: Maximum number of conversation entries (None = unlimited)
        """
        self.max_size = max_size
        self.conversations: deque = deque(maxlen=max_size)
        self._counter = 0  # For generating unique keys

    def add(self, entry: StateEntry) -> None:
        """Add a conversation entry.

        For conversations, the key is auto-generated if not provided.
        The value should be the conversation text.
        """
        # Auto-generate key if needed
        if not entry.key or entry.key.startswith("auto_"):
            entry.key = f"conversation_{self._counter}"
            self._counter += 1

        self.conversations.append(entry)
        logger.debug(f"[CONV-BACKEND] Added: {str(entry.value)[:50]}...")

    def get(self, key: str) -> Optional[StateEntry]:
        """Get a specific conversation by key."""
        for entry in self.conversations:
            if entry.key == key:
                return entry
        return None

    def get_all(self) -> List[StateEntry]:
        """Get all conversation entries."""
        return list(self.conversations)

    def get_recent(self, limit: int = 10) -> List[StateEntry]:
        """Get recent conversation entries."""
        all_entries = list(self.conversations)
        return sorted(all_entries, key=lambda e: e.timestamp, reverse=True)[:limit]

    def delete(self, key: str) -> None:
        """Delete a conversation entry by key."""
        self.conversations = deque(
            (e for e in self.conversations if e.key != key),
            maxlen=self.max_size
        )
        logger.debug(f"[CONV-BACKEND] Deleted '{key}'")

    def clear(self) -> None:
        """Clear all conversations."""
        self.conversations.clear()
        self._counter = 0
        logger.info("[CONV-BACKEND] Cleared all conversations")

    def count(self) -> int:
        """Count conversation entries."""
        return len(self.conversations)

    def search(self, query: str, limit: int = 5) -> List[StateEntry]:
        """Search conversations by keyword.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching StateEntry objects
        """
        query_lower = query.lower()
        matches = [
            entry for entry in self.conversations
            if query_lower in str(entry.value).lower()
        ]
        # Sort by timestamp (most recent first)
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        return matches[:limit]


class SharedState:
    """Shared state that can be accessed by all agents in the DAG.

    Supports token-aware context management to prevent prompt overflow.
    Uses pluggable backends for flexible storage (key-value, conversation memory, etc.).
    """

    def __init__(
        self,
        backend: Optional[StateBackend] = None,
        max_context_tokens: Optional[int] = None,
        summary_threshold_tokens: Optional[int] = None,
        llm_client: Optional[Any] = None,
        model: Optional[str] = None,
        encoding_name: str = "cl100k_base",
        context_strategy: StateContextStrategy = StateContextStrategy.FULL
    ):
        """Initialize shared state.

        Args:
            backend: Storage backend (defaults to KeyValueBackend)
            max_context_tokens: Maximum tokens for state context (None = no limit)
            summary_threshold_tokens: When to trigger summarization (for COMPACT strategy)
            llm_client: OpenAI-compatible client for COMPACT strategy
            model: Model name for summarization
            encoding_name: Tiktoken encoding name
            context_strategy: Default strategy for get_context() calls
        """
        # Use provided backend or default to KeyValueBackend
        self.backend = backend or KeyValueBackend()

        self.max_context_tokens = max_context_tokens
        self.summary_threshold_tokens = summary_threshold_tokens or (max_context_tokens // 2 if max_context_tokens else None)
        self.llm_client = llm_client
        self.model = model or "gemma3:27b"
        self.context_strategy = context_strategy
        self._summary: Optional[str] = None

        # Initialize tokenizer
        self.encoding = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
                logger.debug(f"[STATE] Initialized tiktoken encoding: {encoding_name}")
            except Exception as e:
                logger.warning(f"[STATE] Could not load tokenizer {encoding_name}: {e}")

        if not self.encoding:
            logger.debug("[STATE] Using fallback token estimation (1 token ~= 4 chars)")

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

        self.backend.add(entry)
        logger.debug(f"[STATE] Set '{key}' = {str(value)[:100]} (source: {source})")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared state.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            Value associated with key or default
        """
        entry = self.backend.get(key)
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
        return self.backend.get(key) is not None

    def delete(self, key: str) -> None:
        """Delete a key from state.

        Args:
            key: State key
        """
        self.backend.delete(key)

    def get_entry(self, key: str) -> Optional[StateEntry]:
        """Get the full state entry with metadata.

        Args:
            key: State key

        Returns:
            StateEntry or None
        """
        return self.backend.get(key)

    def get_all(self) -> Dict[str, Any]:
        """Get all state values as dictionary.

        Returns:
            Dictionary of all state values
        """
        entries = self.backend.get_all()
        return {entry.key: entry.value for entry in entries}

    def get_history(self, key: Optional[str] = None) -> List[StateEntry]:
        """Get state change history (if backend supports it).

        Args:
            key: Optional key to filter history

        Returns:
            List of state entries
        """
        # Try to get history from backend if it supports it (like KeyValueBackend)
        if hasattr(self.backend, 'get_history'):
            return self.backend.get_history(key)

        # Fallback: return all entries filtered by key
        all_entries = self.backend.get_all()
        if key is None:
            return all_entries
        return [entry for entry in all_entries if entry.key == key]

    def clear(self) -> None:
        """Clear all state."""
        self.backend.clear()
        self._summary = None  # Clear summary as well
        logger.info("[STATE] Cleared all state")

    def to_dict(self) -> Dict[str, Any]:
        """Export state as dictionary.

        Returns:
            Dictionary with all state entries
        """
        entries = self.backend.get_all()
        return {entry.key: entry.to_dict() for entry in entries}

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
            # Fallback: rough estimation (1 token ~= 4 characters)
            return len(text) // 4

    def get_context(
        self,
        strategy: Optional[StateContextStrategy] = None,
        max_tokens: Optional[int] = None,
        include_summary: bool = True
    ) -> str:
        """Get state context formatted for prompt inclusion with token management.

        Inspired by MemoryManager.get_context() but adapted for state data.

        Args:
            strategy: Context management strategy (FULL, CLIP, or COMPACT), defaults to self.context_strategy
            max_tokens: Maximum tokens (uses self.max_context_tokens if None)
            include_summary: Whether to include summary for COMPACT strategy

        Returns:
            Formatted state string ready for prompt
        """
        if self.backend.count() == 0:
            return ""

        strategy = strategy or self.context_strategy
        max_tokens = max_tokens or self.max_context_tokens

        if strategy == StateContextStrategy.FULL:
            return self._get_full_context(max_tokens)
        elif strategy == StateContextStrategy.CLIP:
            return self._get_clipped_context(max_tokens)
        elif strategy == StateContextStrategy.COMPACT:
            return self._get_compact_context(max_tokens, include_summary)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _get_full_context(self, max_tokens: Optional[int]) -> str:
        """Get full state context without truncation.

        Args:
            max_tokens: Maximum tokens (warning only)

        Returns:
            Full state as formatted string
        """
        state_dict = {k: str(v) for k, v in self.get_all().items()}
        state_json = json.dumps(state_dict, indent=2)
        tokens = self.count_tokens(state_json)

        if max_tokens and tokens > max_tokens:
            logger.warning(
                f"[STATE] Full state context ({tokens} tokens) exceeds max_tokens ({max_tokens}). "
                f"Consider using CLIP or COMPACT strategy."
            )

        logger.debug(f"[STATE] Full context: {tokens} tokens, {self.backend.count()} entries")
        return f"\n\nShared state: {state_json}"

    def _get_clipped_context(self, max_tokens: Optional[int]) -> str:
        """Get state context keeping only most recent entries.

        Similar to MemoryManager's token-aware context building.

        Args:
            max_tokens: Maximum tokens allowed

        Returns:
            Clipped state as formatted string
        """
        if not max_tokens:
            logger.debug("[STATE] No max_tokens specified for CLIP strategy, using FULL")
            return self._get_full_context(None)

        # Get all entries and sort by timestamp (most recent first)
        all_entries = self.backend.get_all()
        sorted_entries = sorted(all_entries, key=lambda e: e.timestamp, reverse=True)

        clipped_state = {}
        current_tokens = 0
        included_count = 0

        # Add entries until we hit token limit (similar to MemoryManager approach)
        for entry in sorted_entries:
            # Try adding this entry
            temp_state = {**clipped_state, entry.key: str(entry.value)}
            temp_json = json.dumps(temp_state, indent=2)
            temp_tokens = self.count_tokens(temp_json)

            # Check if we would exceed limit
            if temp_tokens > max_tokens:
                if included_count == 0:
                    # At least include one entry
                    logger.warning(
                        f"[STATE] Single entry exceeds max_tokens ({temp_tokens} > {max_tokens}). "
                        f"Including it anyway."
                    )
                    clipped_state[entry.key] = str(entry.value)
                    current_tokens = temp_tokens
                    included_count = 1
                break

            # Add entry
            clipped_state[entry.key] = str(entry.value)
            current_tokens = temp_tokens
            included_count += 1

        if not clipped_state:
            logger.warning("[STATE] CLIP strategy resulted in empty state")
            return ""

        state_json = json.dumps(clipped_state, indent=2)
        dropped_count = self.backend.count() - included_count

        logger.info(
            f"[STATE] CLIP context: kept {included_count}/{self.backend.count()} entries "
            f"({current_tokens} tokens, dropped {dropped_count} oldest)"
        )

        return f"\n\nShared state (most recent {included_count} entries): {state_json}"

    def _get_compact_context(self, max_tokens: Optional[int], include_summary: bool) -> str:
        """Get state context with LLM-based summarization.

        Inspired by MemoryManager's summarization approach.

        Args:
            max_tokens: Maximum tokens allowed
            include_summary: Whether to use/create summary

        Returns:
            Compacted state as formatted string
        """
        state_dict = {k: str(v) for k, v in self.get_all().items()}
        state_json = json.dumps(state_dict, indent=2)
        full_tokens = self.count_tokens(state_json)

        # Check if we need to compact
        if not max_tokens or full_tokens <= max_tokens:
            logger.debug(
                f"[STATE] Full state ({full_tokens} tokens) fits within limit, "
                f"no compaction needed"
            )
            return f"\n\nShared state: {state_json}"

        # Need to compact
        if not self.llm_client:
            logger.error(
                "[STATE] COMPACT strategy requires LLM client but none provided. "
                "Falling back to CLIP strategy."
            )
            return self._get_clipped_context(max_tokens)

        logger.info(
            f"[STATE] COMPACT strategy: summarizing state "
            f"({full_tokens} tokens -> target ~{max_tokens} tokens)"
        )

        try:
            # Create or use existing summary
            if not self._summary or not include_summary:
                self._create_state_summary(max_tokens)

            if self._summary:
                summary_tokens = self.count_tokens(self._summary)
                logger.info(
                    f"[STATE] Compacted state: {full_tokens} -> {summary_tokens} tokens "
                    f"({len(self._state)} entries)"
                )
                return f"\n\nShared state (summarized): {self._summary}"
            else:
                # Summary creation failed, fallback
                logger.warning("[STATE] Summary creation failed, falling back to CLIP")
                return self._get_clipped_context(max_tokens)

        except Exception as e:
            logger.error(f"[STATE] COMPACT strategy failed: {e}")
            logger.warning("[STATE] Falling back to CLIP strategy")
            return self._get_clipped_context(max_tokens)

    def _create_state_summary(self, target_tokens: Optional[int]) -> None:
        """Create LLM summary of state data.

        Similar to MemoryManager._create_summary().

        Args:
            target_tokens: Target token count for summary
        """
        if not self.llm_client:
            logger.warning("[STATE] No LLM client provided for summarization")
            return

        if self.backend.count() == 0:
            return

        # Format state for summarization
        state_dict = self.to_dict()
        state_text = json.dumps(state_dict, indent=2, default=str)

        # Create summarization prompt
        target_info = f"approximately {target_tokens} tokens or less" if target_tokens else "concisely"
        prompt = f"""You are a helpful assistant that summarizes shared state data for an AI agent.

The following is the current shared state that contains information from previous agent steps:

{state_text}

Please create a concise summary that:
1. Preserves the most important information and key-value pairs
2. Maintains essential context needed for the agent to continue its work
3. Uses {target_info}
4. Keeps critical data like task results, decisions, and intermediate outputs
5. Can drop verbose details but keep semantic meaning
6. Maintains the relationships between state entries when relevant

Return ONLY the summary in a clear, structured format. Do not include explanations or meta-commentary."""

        try:
            # Call LLM using OpenAI client interface
            messages = [{"role": "user", "content": prompt}]

            generation_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3  # Lower temperature for consistent summaries
            }

            if target_tokens:
                generation_kwargs["max_tokens"] = int(target_tokens * 1.5)  # Allow some overhead

            response = self.llm_client.chat.completions.create(**generation_kwargs)
            self._summary = response.choices[0].message.content.strip()

            logger.info(f"[STATE] Created state summary ({self.count_tokens(self._summary)} tokens)")
            logger.debug(f"[STATE] Summary: {self._summary[:200]}...")

        except Exception as e:
            logger.exception(f"[STATE] Failed to create state summary: {e}")
            self._summary = None

    def clear_summary(self) -> None:
        """Clear the cached summary."""
        self._summary = None
        logger.debug("[STATE] Cleared state summary")

    def get_state_stats(self) -> Dict[str, Any]:
        """Get state statistics including token usage.

        Similar to MemoryManager.get_memory_stats().

        Returns:
            Dictionary with state statistics
        """
        state_dict = {k: str(v) for k, v in self.get_all().items()}
        state_json = json.dumps(state_dict, indent=2)
        total_tokens = self.count_tokens(state_json)

        stats = {
            "total_entries": self.backend.count(),
            "total_tokens": total_tokens,
            "has_summary": self._summary is not None,
            "summary_tokens": self.count_tokens(self._summary) if self._summary else 0,
            "max_context_tokens": self.max_context_tokens
        }

        # Include history length if backend supports it
        if hasattr(self.backend, 'get_history'):
            history = self.backend.get_history()
            stats["history_length"] = len(history)

        if self.max_context_tokens:
            stats["utilization"] = total_tokens / self.max_context_tokens

        return stats

    def search(self, query: str, limit: int = 5) -> List[StateEntry]:
        """Search state entries (if backend supports search).

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching StateEntry objects
        """
        if hasattr(self.backend, 'search'):
            return self.backend.search(query, limit)

        # Fallback: simple keyword search on all entries
        query_lower = query.lower()
        all_entries = self.backend.get_all()
        matches = [
            entry for entry in all_entries
            if query_lower in str(entry.value).lower()
        ]
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        return matches[:limit]

    def __repr__(self) -> str:
        """String representation."""
        return f"SharedState(backend={self.backend.__class__.__name__}, entries={self.backend.count()})"


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
        # Create snapshot of all backend entries
        snapshot = {entry.key: entry for entry in self.current_state.backend.get_all()}
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
            # Clear current state and restore from snapshot
            self.current_state.clear()
            snapshot = self._snapshots[snapshot_id]
            for entry in snapshot.values():
                self.current_state.backend.add(entry)
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
