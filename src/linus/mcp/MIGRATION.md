# MCP Structure Migration

This document explains the reorganization of the MCP server structure.

## Changes Made

### Before (Initial Structure)
```
src/linus/mcp/
├── __init__.py
├── __main__.py
├── mcp_vector_store.py
├── test_mcp_server.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
├── Makefile
├── README.md
└── QUICKSTART.md
```

### After (Reorganized Structure)
```
src/linus/mcp/
├── __init__.py              # MCP module root (updated)
├── README.md                # Main MCP documentation (new)
├── MIGRATION.md             # This file (new)
│
└── vector_store/            # Vector Store MCP Server (new subdirectory)
    ├── __init__.py
    ├── __main__.py
    ├── server.py            # (renamed from mcp_vector_store.py)
    ├── test_server.py       # (renamed from test_mcp_server.py)
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── .env.example
    ├── .dockerignore
    ├── Makefile
    ├── README.md
    └── QUICKSTART.md
```

## Rationale

The reorganization allows for:

1. **Multiple MCP Servers**: Each server in its own subdirectory
2. **Better Organization**: Clear separation of concerns
3. **Scalability**: Easy to add new MCP server implementations
4. **Isolation**: Each server is self-contained with its own dependencies

## Import Path Changes

### Old Import Path
```python
from linus.mcp.mcp_vector_store import VectorStoreMCPServer
```

### New Import Path
```python
from linus.mcp.vector_store import VectorStoreMCPServer
# or
from linus.mcp.vector_store.server import VectorStoreMCPServer
```

## Command Changes

### Old Commands

**Run server:**
```bash
python -m linus.mcp.mcp_vector_store
```

**Run tests:**
```bash
python -m linus.mcp.test_mcp_server
```

**Docker build:**
```bash
docker build -f src/linus/mcp/Dockerfile -t mcp-vector-store .
```

**Working directory:**
```bash
cd src/linus/mcp
```

### New Commands

**Run server:**
```bash
python -m linus.mcp.vector_store
```

**Run tests:**
```bash
python -m linus.mcp.vector_store.test_server
```

**Docker build:**
```bash
docker build -f src/linus/mcp/vector_store/Dockerfile -t mcp-vector-store .
```

**Working directory:**
```bash
cd src/linus/mcp/vector_store
```

## MCP Client Configuration Changes

### Old Configuration
```json
{
  "mcpServers": {
    "vector-store": {
      "command": "python",
      "args": ["-m", "linus.mcp.mcp_vector_store"],
      "cwd": "/path/to/agents/src"
    }
  }
}
```

### New Configuration
```json
{
  "mcpServers": {
    "vector-store": {
      "command": "python",
      "args": ["-m", "linus.mcp.vector_store"],
      "cwd": "/path/to/agents/src"
    }
  }
}
```

## Docker Compose Changes

### Context Path
The docker-compose.yml build context has been updated:

**Old:**
```yaml
build:
  context: ../../..
  dockerfile: src/linus/mcp/Dockerfile
```

**New:**
```yaml
build:
  context: ../../../..
  dockerfile: src/linus/mcp/vector_store/Dockerfile
```

## Makefile Changes

All relative paths in the Makefile have been updated to account for the deeper nesting:

**Test command:**
- Old: `cd .. && python -m linus.mcp.test_mcp_server`
- New: `cd ../../../.. && python -m linus.mcp.vector_store.test_server`

**Run command:**
- Old: `cd ../../.. && python -m linus.mcp.mcp_vector_store`
- New: `cd ../../../.. && python -m linus.mcp.vector_store`

**Build command:**
- Old: `cd ../../.. && docker build -f src/linus/mcp/Dockerfile`
- New: `cd ../../../.. && docker build -f src/linus/mcp/vector_store/Dockerfile`

## Documentation Updates

All documentation has been updated to reflect the new structure:

1. **README.md** (vector_store): Updated all commands and paths
2. **QUICKSTART.md** (vector_store): Updated navigation and commands
3. **README.md** (main MCP): New file explaining the overall structure
4. **MIGRATION.md**: This file

## Adding New MCP Servers

The new structure makes it easy to add more MCP servers:

```bash
# Create new server directory
mkdir src/linus/mcp/my_new_server

# Copy template structure from vector_store
cp -r src/linus/mcp/vector_store/{__init__.py,__main__.py,requirements.txt,Dockerfile,docker-compose.yml,Makefile} src/linus/mcp/my_new_server/

# Implement server.py for your new server
```

Then update:
1. `src/linus/mcp/__init__.py` to include the new server
2. Documentation to reference the new server

## Testing the Migration

To verify everything works:

1. **Check imports:**
   ```bash
   PYTHONPATH=src python -c "from linus.mcp.vector_store import VectorStoreMCPServer; print('OK')"
   ```

2. **Test Docker build:**
   ```bash
   cd src/linus/mcp/vector_store
   make build
   ```

3. **Test Docker Compose:**
   ```bash
   cd src/linus/mcp/vector_store
   make up
   make ps
   make down
   ```

4. **Test local run:**
   ```bash
   cd src/linus/mcp/vector_store
   pip install -r requirements.txt
   make run
   ```

## Rollback (if needed)

If you need to rollback to the old structure:

1. Move all files from `vector_store/` back to `mcp/`
2. Rename `server.py` to `mcp_vector_store.py`
3. Rename `test_server.py` to `test_mcp_server.py`
4. Update all imports and paths back to original
5. Remove the `vector_store/` directory

## Benefits of New Structure

1. ✅ **Modularity**: Each MCP server is independent
2. ✅ **Scalability**: Easy to add new servers
3. ✅ **Organization**: Clear hierarchy and separation
4. ✅ **Isolation**: Each server has its own dependencies
5. ✅ **Documentation**: Better structured docs
6. ✅ **Future-proof**: Supports multiple MCP implementations

## Checklist

- [x] Move files to `vector_store/` subdirectory
- [x] Rename `mcp_vector_store.py` → `server.py`
- [x] Rename `test_mcp_server.py` → `test_server.py`
- [x] Update `Dockerfile` CMD path
- [x] Update `docker-compose.yml` context and dockerfile path
- [x] Update `Makefile` commands
- [x] Update `README.md` (vector_store)
- [x] Update `QUICKSTART.md` (vector_store)
- [x] Create main `README.md` (mcp/)
- [x] Update `__init__.py` files
- [x] Create `__main__.py` for module execution
- [x] Create migration documentation
- [x] Test imports
- [x] Verify structure
