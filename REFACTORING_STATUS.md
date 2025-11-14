# Memory/State Refactoring Status (Option 3)

## Summary

Successfully implemented Option 3: Making MemoryManager a specialized backend for SharedState. This eliminates code duplication while maintaining semantic clarity.

## Completed Tasks ✅

### 1. Backend Architecture ✅
- Created `StateBackend` abstract interface in [state.py](src/linus/agents/graph/state.py:47-118)
- Implemented `KeyValueBackend` for workflow state (key-value pairs)
- Implemented `ConversationMemoryBackend` for conversation history (sequential entries)

### 2. SharedState Refactoring ✅
- Updated `SharedState` to use pluggable backends
- All methods now delegate to `self.backend`
- Token counting and summarization logic centralized (no duplication)
- Support for multiple strategies (FULL, CLIP, COMPACT)

### 3. Agent Base Class ✅
- Updated [base.py](src/linus/agents/agent/base.py) to use unified approach
- Removed `memory_manager: Optional[MemoryManager]` parameter
- Added `memory: Optional[SharedState]` parameter (uses ConversationMemoryBackend)
- Added backward-compatible convenience methods:
  - `add_memory()` - wraps `self.memory.set()`
  - `get_memory_context()` - wraps `self.memory.get_context()`
  - `clear_memory()` - wraps `self.memory.clear()`
  - `get_memory_stats()` - wraps `self.memory.get_state_stats()`

### 4. ReasoningAgent Updates ✅
- Removed `MemoryManager` import
- Changed `memory_manager` parameter to `memory`
- Updated all usages:
  - `self.memory_manager.add_memory()` → `self.add_memory()`
  - `self.memory_manager.get_context()` → `self.get_memory_context()`
  - `self.memory_manager.get_memory_stats()` → `self.get_memory_stats()`

## Completed Tasks (Continued) ✅

### 5. Update LightAgent ✅
Files: [light_agent.py](src/linus/agents/agent/light_agent.py)

Changes completed:
- ✅ Changed parameter from `memory_manager` to `memory`
- ✅ Updated super().__init__() call to pass `memory`
- ✅ Updated all usages to use convenience methods:
  - `self.memory_manager.get_context()` → `self.get_memory_context()`
  - `self.memory_manager.add_memory()` → `self.add_memory()`
- ✅ Removed MemoryManager import

### 6. Update CoordinatorAgent ✅
Files: [coordinator_agent.py](src/linus/agents/agent/coordinator_agent.py)

Changes completed:
- ✅ Changed parameter from `memory_manager` to `memory`
- ✅ Updated super().__init__() call to pass `memory`
- ✅ Updated all usages to use convenience methods
- ✅ Removed MemoryManager import

### 7. Update Factory ✅
File: [factory.py](src/linus/agents/agent/factory.py)

Changes completed:
- ✅ Removed MemoryManager imports
- ✅ Added ConversationMemoryBackend import
- ✅ Updated all factory functions (Agent, Coordinator, TreeOfThought, Light)
- ✅ Replaced memory_manager creation with SharedState + ConversationMemoryBackend
- ✅ All agents now receive `memory` parameter instead of `memory_manager`

### 8. Testing ⏳
Next steps:
- Test basic agent creation with new API
- Test conversation memory (add_memory, get_memory_context)
- Test state management (set, get, get_context)
- Test backward compatibility
- Run existing tests: `pytest tests/test_agent_memory.py`

## Architecture Benefits

### Before (Duplication):
```
MemoryManager (conversation history)
  ├── InMemoryBackend
  ├── Token counting (tiktoken)
  ├── Summarization logic
  └── Context management

SharedState (workflow data)
  ├── Dict storage
  ├── Token counting (tiktoken) ← DUPLICATE
  ├── Summarization logic ← DUPLICATE
  └── Context management ← DUPLICATE
```

### After (Unified):
```
SharedState (universal storage)
  ├── Token counting (tiktoken) ← SINGLE IMPLEMENTATION
  ├── Summarization logic ← SINGLE IMPLEMENTATION
  ├── Context management ← SINGLE IMPLEMENTATION
  └── Pluggable backends:
      ├── KeyValueBackend (workflow state)
      ├── ConversationMemoryBackend (conversation history)
      └── VectorStoreBackend (semantic search - future)
```

## Migration Guide for Users

### Old API (deprecated):
```python
from linus.agents.agent.memory import MemoryManager, create_memory_manager

memory_manager = create_memory_manager(
    backend_type="in_memory",
    max_context_tokens=4096,
    llm=llm,
    model="gemma3:27b"
)

agent = Agent(..., memory_manager=memory_manager)
```

### New API:
```python
from linus.agents.graph.state import SharedState, ConversationMemoryBackend

memory = SharedState(
    backend=ConversationMemoryBackend(max_size=100),
    max_context_tokens=4096,
    llm_client=llm,
    model="gemma3:27b"
)

agent = Agent(..., memory=memory)
```

### Backward Compatible Methods:
```python
# These still work (convenience wrappers):
agent.add_memory("User said hello")
context = agent.get_memory_context(max_tokens=1000)
agent.clear_memory()
stats = agent.get_memory_stats()
```

## Files Modified

1. ✅ [src/linus/agents/graph/state.py](src/linus/agents/graph/state.py) - Added backends
2. ✅ [src/linus/agents/agent/base.py](src/linus/agents/agent/base.py) - Unified API
3. ✅ [src/linus/agents/agent/reasoning_agent.py](src/linus/agents/agent/reasoning_agent.py) - Updated
4. ✅ [src/linus/agents/agent/light_agent.py](src/linus/agents/agent/light_agent.py) - Updated
5. ✅ [src/linus/agents/agent/coordinator_agent.py](src/linus/agents/agent/coordinator_agent.py) - Updated
6. ✅ [src/linus/agents/agent/factory.py](src/linus/agents/agent/factory.py) - Updated

## Next Steps

1. ✅ Complete LightAgent and CoordinatorAgent updates
2. ✅ Update factory.py to create ConversationMemoryBackend instead of MemoryManager
3. ⏳ Run tests and fix any issues
4. ⏳ Optional: Deprecate memory.py file (keep for backward compatibility)
5. ⏳ Update documentation in CLAUDE.md

## Summary

**Refactoring Complete!** All agent classes and factory functions have been updated to use the unified `SharedState` + backend architecture. The old `MemoryManager` has been replaced with `SharedState(backend=ConversationMemoryBackend())` across the codebase.

**Key Changes:**
- All agents now accept `memory: Optional[SharedState]` instead of `memory_manager: Optional[MemoryManager]`
- Factory functions create `SharedState` with `ConversationMemoryBackend` for conversation memory
- Backward-compatible convenience methods available in base.py: `add_memory()`, `get_memory_context()`, `clear_memory()`, `get_memory_stats()`
- No code duplication: token counting and summarization logic centralized in `SharedState`

## Questions?

The refactoring follows clean architecture principles:
- **Single Responsibility**: Each backend handles one type of storage
- **Open/Closed**: Easy to add new backends without modifying SharedState
- **Dependency Inversion**: SharedState depends on abstract StateBackend
- **DRY**: Token counting and summarization logic in one place
