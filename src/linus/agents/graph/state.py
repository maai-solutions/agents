"""Shared state management for agent orchestration."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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


class SharedState:
    """Shared state that can be accessed by all agents in the DAG.

    Supports token-aware context management to prevent prompt overflow.
    """

    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        summary_threshold_tokens: Optional[int] = None,
        llm_client: Optional[Any] = None,
        model: Optional[str] = None,
        encoding_name: str = "cl100k_base",
        context_strategy: StateContextStrategy = StateContextStrategy.FULL
    ):
        """Initialize shared state.

        Args:
            max_context_tokens: Maximum tokens for state context (None = no limit)
            summary_threshold_tokens: When to trigger summarization (for COMPACT strategy)
            llm_client: OpenAI-compatible client for COMPACT strategy
            model: Model name for summarization
            encoding_name: Tiktoken encoding name
            context_strategy: Default strategy for get_context() calls
        """
        self._state: Dict[str, StateEntry] = {}
        self._history: List[StateEntry] = []
        self.max_context_tokens = max_context_tokens
        self.summary_threshold_tokens = summary_threshold_tokens or (max_context_tokens // 2 if max_context_tokens else None)
        self.llm_client = llm_client
        self.model = model or "gemma3:27b"
        self.context_strategy = context_strategy
        self._summary: Optional[str] = None

        # Initialize tokenizer (inspired by MemoryManager)
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
        strategy: StateContextStrategy = StateContextStrategy.FULL,
        max_tokens: Optional[int] = None,
        include_summary: bool = True
    ) -> str:
        """Get state context formatted for prompt inclusion with token management.

        Inspired by MemoryManager.get_context() but adapted for state data.

        Args:
            strategy: Context management strategy (FULL, CLIP, or COMPACT)
            max_tokens: Maximum tokens (uses self.max_context_tokens if None)
            include_summary: Whether to include summary for COMPACT strategy

        Returns:
            Formatted state string ready for prompt
        """
        if not self._state:
            return ""

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

        logger.debug(f"[STATE] Full context: {tokens} tokens, {len(self._state)} entries")
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

        # Sort entries by timestamp (most recent first)
        sorted_entries = sorted(self._state.values(), key=lambda e: e.timestamp, reverse=True)

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
        dropped_count = len(self._state) - included_count

        logger.info(
            f"[STATE] CLIP context: kept {included_count}/{len(self._state)} entries "
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

        if not self._state:
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
            "total_entries": len(self._state),
            "total_tokens": total_tokens,
            "has_summary": self._summary is not None,
            "summary_tokens": self.count_tokens(self._summary) if self._summary else 0,
            "max_context_tokens": self.max_context_tokens,
            "history_length": len(self._history)
        }

        if self.max_context_tokens:
            stats["utilization"] = total_tokens / self.max_context_tokens

        return stats

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
