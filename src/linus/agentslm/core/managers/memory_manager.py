from typing import Any, Dict, List, List, Optional

from loguru import logger
import tiktoken
from agentslm.core.interfaces.logger import ILogger, NoOpLogger
from agentslm.core.memory.memory import InMemoryBackend, MemoryBackend, MemoryEntry


class MemoryManager:
    """Manages agent memory with token-aware context window management."""

    def __init__(
        self,
        backend: MemoryBackend,
        max_context_tokens: int = 4096,
        summary_threshold_tokens: int = 2048,
        llm: Optional[Any] = None,
        model: Optional[str] = None,
        encoding_name: str = "cl100k_base",
        logger: ILogger = NoOpLogger(),
        tracer: Optional[Any] = None
    ):
        """Initialize memory manager.

        Args:
            backend: Memory storage backend
            max_context_tokens: Maximum tokens for context window
            summary_threshold_tokens: When to trigger summarization
            llm: Language model for summarization (OpenAI client recommended)
            model: Model name for OpenAI client (e.g., "gpt-3.5-turbo", "gemma3:27b")
            encoding_name: Tokenizer encoding to use
            logger: Logger instance
            tracer: Optional telemetry tracer (AgentTracer or LangfuseTracer)
        """
        self.backend = backend
        self.max_context_tokens = max_context_tokens
        self.summary_threshold_tokens = summary_threshold_tokens
        self.llm = llm
        self.model = model or "gpt-3.5-turbo"
        self.summary: Optional[str] = None
        self.logger = logger
        self.tracer = tracer

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            self.logger.warning(f"Could not load tokenizer {encoding_name}: {e}")
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
        # Trace memory addition
        if self.tracer:
            span_ctx = self._trace_memory_operation("add_memory", {"entry_type": entry_type, "importance": importance})
        else:
            from contextlib import nullcontext
            span_ctx = nullcontext()

        with span_ctx:
            entry = MemoryEntry(
                content=content,
                metadata=metadata or {},
                importance=importance,
                entry_type=entry_type
            )
            self.backend.add(entry)

            # Record memory stats
            if self.tracer:
                self._record_memory_stats()

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
            self.logger.warning("[MEMORY] No LLM provided for summarization")
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
            # Use OpenAI client directly
            messages = [
                {"role": "system", "content": "Summarize the following conversation history concisely, preserving key information and context."},
                {"role": "user", "content": f"Conversation history:\n{memory_text}\n\nProvide a concise summary:"}
            ]

            # Check if it's an OpenAI client or old LangChain LLM
            if hasattr(self.llm, 'chat'):
                # OpenAI client (sync or async)
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                )
                self.summary = response.choices[0].message.content
            else:
                # Fallback for other LLM types (backward compatibility)
                self.logger.warning("[MEMORY] Using fallback LLM interface - consider updating to OpenAI client")
                response = self.llm.invoke(messages)
                self.summary = response.content if hasattr(response, 'content') else str(response)

            self.logger.info(f"[MEMORY] Created summary of {len(memories_to_summarize)} memories")
            self.logger.debug(f"[MEMORY] Summary: {self.summary[:200]}...")

        except Exception as e:
            self.logger.exception(f"[MEMORY] Failed to create summary: {e}")


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
        # Trace context retrieval
        if self.tracer:
            span_ctx = self._trace_memory_operation("get_context", {
                "max_tokens": max_tokens or self.max_context_tokens,
                "include_summary": include_summary,
                "has_query": query is not None
            })
        else:
            from contextlib import nullcontext
            span_ctx = nullcontext()

        with span_ctx:
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

            # Record final token count
            if self.tracer:
                self.tracer.set_attribute("memory.context_tokens", current_tokens)

            return context

    def search_memories(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Search for relevant memories.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching memory entries
        """
        # Trace search operation
        if self.tracer:
            span_ctx = self._trace_memory_operation("search", {"query_length": len(query), "limit": limit})
        else:
            from contextlib import nullcontext
            span_ctx = nullcontext()

        with span_ctx:
            results = self.backend.search(query, limit)

            # Record search results count
            if self.tracer:
                self.tracer.set_attribute("memory.search_results", len(results))

            return results

    def clear_memory(self) -> None:
        """Clear all memories and summary."""
        self.backend.clear()
        self.summary = None
        self.logger.info("[MEMORY] Cleared all memories")

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
        self.logger.info(f"[MEMORY] Imported {len(memories)} memories")


def create_memory_manager(
    backend_type: str = "in_memory",
    max_context_tokens: int = 4096,
    summary_threshold_tokens: int = 2048,
    llm: Optional[Any] = None,
    model: Optional[str] = None,
    max_size: Optional[int] = None,
    logger: ILogger = NoOpLogger(),
    tracer: Optional[Any] = None
) -> MemoryManager:
    """Factory function to create a memory manager.

    Args:
        backend_type: Type of backend ("in_memory" or "vector_store")
        max_context_tokens: Maximum tokens for context window
        summary_threshold_tokens: When to trigger summarization
        llm: Language model for summarization (OpenAI client recommended)
        model: Model name for OpenAI client (e.g., "gpt-3.5-turbo", "gemma3:27b")
        max_size: Maximum number of memories (for in_memory backend)
        tracer: Optional telemetry tracer (AgentTracer or LangfuseTracer)

    Returns:
        Configured MemoryManager instance

    Example:
        from openai import OpenAI
        from linus.agents.telemetry import initialize_telemetry

        llm = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")
        tracer = initialize_telemetry(exporter_type="langfuse", enabled=True)

        memory = create_memory_manager(
            backend_type="in_memory",
            llm=llm,
            model="gemma3:27b",
            max_context_tokens=4096,
            tracer=tracer
        )
    """
    if backend_type == "in_memory":
        backend = InMemoryBackend(max_size=max_size, logger=logger)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    return MemoryManager(
        backend=backend,
        max_context_tokens=max_context_tokens,
        summary_threshold_tokens=summary_threshold_tokens,
        llm=llm,
        model=model,
        logger=logger,
        tracer=tracer
    )