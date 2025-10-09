# MCP Servers for Linus Agents

This directory contains Model Context Protocol (MCP) servers that expose various agent capabilities through the MCP protocol.

## Overview

MCP (Model Context Protocol) is a standardized protocol for communication between AI systems and external tools/data sources. Each subdirectory in this folder contains a standalone MCP server implementation.

## Available MCP Servers

### [vector_store](vector_store/)

Vector search server using Weaviate and hybrid search capabilities.

**Features:**
- Hybrid search (vector + keyword)
- Configurable search parameters
- OpenAI-compatible embedding support
- Docker deployment ready

**Quick Start:**
```bash
cd vector_store
make up
```

See [vector_store/QUICKSTART.md](vector_store/QUICKSTART.md) for detailed instructions.

## Architecture

```
src/linus/mcp/
├── __init__.py                    # MCP module root
├── README.md                      # This file
│
├── vector_store/                  # Vector Store MCP Server
│   ├── server.py                  # Main implementation
│   ├── Dockerfile                 # Container definition
│   ├── docker-compose.yml         # Multi-container setup
│   └── README.md                  # Server documentation
│
└── [future_server]/               # Future MCP server implementations
    ├── server.py
    ├── Dockerfile
    └── README.md
```

## Creating New MCP Servers

To add a new MCP server:

1. **Create a new directory** under `src/linus/mcp/`:
   ```bash
   mkdir src/linus/mcp/my_server
   ```

2. **Create the server implementation** (`my_server/server.py`):
   ```python
   from mcp.server import Server
   from mcp.server.stdio import stdio_server
   from mcp.types import Tool, TextContent

   class MyMCPServer:
       def __init__(self):
           self.server = Server("my-server")
           self._register_handlers()

       def _register_handlers(self):
           @self.server.list_tools()
           async def list_tools() -> list[Tool]:
               return [
                   Tool(
                       name="my_tool",
                       description="Description of my tool",
                       inputSchema={
                           "type": "object",
                           "properties": {
                               "param": {"type": "string"}
                           }
                       }
                   )
               ]

           @self.server.call_tool()
           async def call_tool(name: str, arguments: Any):
               if name == "my_tool":
                   return [TextContent(type="text", text="result")]

       async def run(self):
           async with stdio_server() as (read_stream, write_stream):
               await self.server.run(read_stream, write_stream,
                                   self.server.create_initialization_options())

   async def main():
       server = MyMCPServer()
       await server.run()

   if __name__ == "__main__":
       import asyncio
       asyncio.run(main())
   ```

3. **Add module files**:
   - `my_server/__init__.py` - Module initialization
   - `my_server/__main__.py` - Entry point for `python -m`
   - `my_server/requirements.txt` - Dependencies
   - `my_server/README.md` - Documentation

4. **Add Docker support**:
   - `my_server/Dockerfile` - Container definition
   - `my_server/docker-compose.yml` - Orchestration
   - `my_server/Makefile` - Common commands

5. **Update the main MCP module** (`src/linus/mcp/__init__.py`):
   ```python
   """MCP servers for Linus agents.

   Available MCP servers:
   - vector_store: Vector search using Weaviate
   - my_server: Description of my server
   """

   __all__ = ["vector_store", "my_server"]
   ```

## Running MCP Servers

### Local Development

```bash
# Navigate to the server directory
cd src/linus/mcp/[server_name]

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m linus.mcp.[server_name]
```

### Docker Deployment

```bash
# Navigate to the server directory
cd src/linus/mcp/[server_name]

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using with MCP Clients

Configure your MCP client (like Claude Desktop) to use a server:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "python",
      "args": ["-m", "linus.mcp.[server_name]"],
      "cwd": "/path/to/agents/src",
      "env": {
        "CONFIG_VAR": "value"
      }
    }
  }
}
```

## MCP Protocol

All servers in this directory implement the MCP specification:

- **stdio communication**: Servers use standard input/output for protocol messages
- **Tool discovery**: `list_tools` endpoint returns available tools
- **Tool execution**: `call_tool` endpoint executes requested tools
- **Standard responses**: All tools return `TextContent` or other MCP-compatible types

See the [MCP specification](https://spec.modelcontextprotocol.io/) for details.

## Dependencies

Core MCP dependencies:
```
mcp>=0.1.0
pydantic>=2.0.0
```

Each server may have additional dependencies listed in its `requirements.txt`.

## Best Practices

1. **Isolation**: Each server should be self-contained in its own directory
2. **Documentation**: Include README.md and QUICKSTART.md for each server
3. **Docker Support**: Provide Dockerfile and docker-compose.yml
4. **Testing**: Include test scripts for standalone testing
5. **Configuration**: Use environment variables for all configuration
6. **Error Handling**: Return clear error messages through MCP protocol
7. **Async Support**: Use async/await for non-blocking operations

## Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/anthropics/model-context-protocol-python)
- [Claude Desktop Integration](https://claude.ai/docs/mcp)

## Contributing

When adding new MCP servers:

1. Follow the directory structure above
2. Include complete documentation
3. Add Docker support
4. Test with a real MCP client
5. Update this README with the new server

## Support

For issues or questions about specific MCP servers, see their individual README files.
