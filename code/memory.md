# Memory System Overview

This document provides a concise description of the two core components defined in **`src/linus/agents/agent/memory.py`**:

* **`MemoryBackend`** – an abstract interface that defines how memories are stored and retrieved.
* **`MemoryManager`** – a higher‑level manager that uses a backend to maintain a token‑aware context window, summarise older memories, and expose useful utilities for agents.

---

## `MemoryBackend`

`MemoryBackend` is an **abstract base class** (ABC) that specifies the contract any concrete storage implementation must fulfill. It defines the following operations:

| Method | Purpose |
|--------|---------|
| `add(entry: MemoryEntry) -> None` | Store a new `MemoryEntry`. |
| `get_recent(limit: int = 10) -> List[MemoryEntry]` | Retrieve the most recent *limit* entries. |
| `search(query: str, limit: int = 5) -> List[MemoryEntry]` | Return entries that match a textual *query* (default keyword‑based). |
| `clear() -> None` | Remove all stored entries. |
| `get_all() -> List[MemoryEntry]` | Return **all** stored entries. |
| `count() -> int` | Return the total number of stored entries. |

### Implementations

| Class | Storage type | Key characteristics |
|-------|--------------|---------------------|
| `InMemoryBackend` | Python `deque` (in‑process) | Optional `max_size` limits the deque length; fast O(1) appends; simple keyword search. |
| `VectorStoreBackend` | List‑based stub (intended for a vector store) | Holds entries and optional embeddings; currently uses the same keyword search as `InMemoryBackend`; placeholder for future semantic similarity. |

Both implementations inherit the abstract methods, providing concrete behavior for adding, retrieving, searching, and clearing memories.

---

## `MemoryManager`

`MemoryManager` orchestrates **agent‑level memory handling** on top of a `MemoryBackend`. Its responsibilities include:

1. **Token‑aware context window**  
   * Uses `tiktoken` (or a rough estimate) to count tokens in stored memories.  
   * Ensures the returned context fits within `max_context_tokens` (default 4096).

2. **Summarisation**  
   * When the total token count exceeds `summary_threshold_tokens` (default 2048), it optionally generates a concise summary of older memories using a supplied LLM (`llm`).  
   * The summary is stored in `self.summary` and can be prepended to future context strings.

3. **Memory CRUD helpers**  
   * `add_memory(...)` creates a `MemoryEntry` and adds it to the backend, triggering summarisation if needed.  
   * `clear_memory()` clears both the backend and any stored summary.  
   * `export_memories()` / `import_memories()` provide JSON‑serialisable round‑tripping.

4. **Context retrieval** (`get_context`)  
   * Returns a formatted string containing an optional summary and a set of recent or query‑matched memories, respecting the token limit.  
   * Supports optional semantic search (`query`) which currently falls back to the backend’s `search`.

5. **Statistics & introspection**  
   * `get_memory_stats()` reports counts, token usage, and summarisation status.  
   * `search_memories()` is a thin wrapper around the backend’s `search`.

### Construction

```python
memory = MemoryManager(
    backend=InMemoryBackend(max_size=1000),
    max_context_tokens=4096,
    summary_threshold_tokens=2048,
    llm=my_openai_client,   # optional for summarisation
    model="gpt-3.5-turbo"
)
```

A helper factory `create_memory_manager` is provided to instantiate either an in‑memory or vector‑store backend via the `backend_type` argument.

---

## Summary

* **`MemoryBackend`** defines a minimal storage API. Concrete backends (in‑memory or vector store) implement this API.
* **`MemoryManager`** builds on a backend to provide token‑bounded context, optional summarisation via an LLM, and convenient utilities for agents to add, query, and manage their memories.

This design keeps the low‑level storage pluggable while giving agents a high‑level, token‑aware memory interface suitable for large language model interactions.
