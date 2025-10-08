# MCP (Model Context Protocol) Integration

## Overview

The agent now supports **MCP (Model Context Protocol)** - Anthropic's open standard for connecting AI applications to external data sources and tools.

MCP allows you to connect to pre-built servers that provide access to:
- **File systems** - Read/write files
- **GitHub** - Repository management, issues, PRs
- **Databases** - Postgres, MySQL queries
- **Web** - Puppeteer for browser automation
- **Communication** - Slack, email
- **And more** - Growing ecosystem of MCP servers

## Installation

```bash
# Install MCP Python SDK
pip install mcp

# MCP servers are typically Node.js packages (run via npx)
# No installation needed - npx downloads them on-demand
```

## Quick Start

### Basic Usage

```python
import asyncio
from linus.agents.agent import (
    create_gemma_agent,
    MCPServerConfig,
    connect_mcp_servers
)

async def main():
    # Configure MCP server
    servers = {
        "filesystem": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
    }

    # Connect and get tools
    mcp_tools = await connect_mcp_servers(servers)

    # Create agent with MCP tools
    agent = create_gemma_agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        tools=mcp_tools
    )

    # Use the agent
    result = agent.run("List all files in /tmp and read test.txt")
    print(result)

asyncio.run(main())
```

## Available MCP Servers

### Official Servers (by Anthropic)

All servers are NPM packages with the prefix `@modelcontextprotocol/server-*`

#### 1. **Filesystem Server**
Access local files and directories.

```python
MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
)
```

**Tools provided:**
- `read_file` - Read file contents
- `write_file` - Write to file
- `list_directory` - List directory contents
- `create_directory` - Create new directory
- `move_file` - Move/rename files
- `search_files` - Search for files

#### 2. **GitHub Server**
Interact with GitHub repositories.

```python
MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_TOKEN": "ghp_your_token_here"}
)
```

**Tools provided:**
- `list_repositories` - List user repos
- `create_issue` - Create GitHub issue
- `create_pull_request` - Create PR
- `fork_repository` - Fork a repo
- `push_files` - Push files to repo

#### 3. **Postgres Server**
Query PostgreSQL databases.

```python
MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-postgres"],
    env={"DATABASE_URL": "postgresql://user:pass@localhost/db"}
)
```

**Tools provided:**
- `query` - Execute SQL queries
- `list_tables` - List all tables
- `describe_table` - Get table schema

#### 4. **Puppeteer Server**
Web automation and scraping.

```python
MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-puppeteer"]
)
```

**Tools provided:**
- `navigate` - Go to URL
- `screenshot` - Take screenshot
- `click` - Click element
- `fill` - Fill form fields
- `evaluate` - Run JavaScript

#### 5. **Slack Server**
Slack workspace integration.

```python
MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-slack"],
    env={
        "SLACK_BOT_TOKEN": "xoxb-...",
        "SLACK_TEAM_ID": "T..."
    }
)
```

**Tools provided:**
- `post_message` - Send message to channel
- `list_channels` - List all channels
- `get_channel_history` - Get messages

## API Reference

### MCPServerConfig

Configuration for an MCP server.

```python
from linus.agents.agent import MCPServerConfig

config = MCPServerConfig(
    command="npx",                          # Command to run
    args=["-y", "package-name", "arg1"],   # Arguments
    env={"KEY": "value"}                    # Environment variables
)
```

### connect_mcp_servers()

Connect to multiple MCP servers and get their tools.

```python
from linus.agents.agent import connect_mcp_servers

servers = {
    "server1": MCPServerConfig(...),
    "server2": MCPServerConfig(...)
}

tools = await connect_mcp_servers(servers)
```

**Returns:** List of `BaseTool` objects that can be used with the agent.

### MCPClientManager

Lower-level API for managing MCP connections.

```python
from linus.agents.agent import MCPClientManager

manager = MCPClientManager()

# Connect to server
await manager.connect_server("my_server", config)

# Get tools
tools = manager.get_tools()

# Disconnect
await manager.disconnect_server("my_server")
await manager.disconnect_all()
```

### MCPTool

Wrapper class that adapts MCP tools to agent tools.

```python
# MCPTool is created automatically when connecting to servers
# You typically don't create these manually
```

## Complete Example

### Multi-Server Setup

```python
import asyncio
import os
from linus.agents.agent import (
    create_gemma_agent,
    MCPServerConfig,
    connect_mcp_servers,
    get_default_tools
)

async def multi_server_example():
    # Configure multiple MCP servers
    servers = {
        "filesystem": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
        ),
        "github": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")}
        ),
        "postgres": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres"],
            env={"DATABASE_URL": os.getenv("DATABASE_URL")}
        )
    }

    # Connect to all servers
    print("Connecting to MCP servers...")
    mcp_tools = await connect_mcp_servers(servers)
    print(f"Connected! Got {len(mcp_tools)} MCP tools")

    # Combine with default agent tools
    all_tools = get_default_tools() + mcp_tools

    # Create agent
    agent = create_gemma_agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        temperature=0.7,
        tools=all_tools,
        verbose=True
    )

    # Complex multi-tool task
    result = agent.run("""
        1. Query the database for all users created today
        2. Create a summary report file in /home/user/reports/
        3. Create a GitHub issue with the summary
    """, return_metrics=True)

    print(f"Result: {result.result}")
    print(f"Used {result.metrics.tool_executions} tools in {result.metrics.total_iterations} iterations")

asyncio.run(multi_server_example())
```

## Error Handling

```python
from linus.agents.agent import is_mcp_available

# Check if MCP SDK is installed
if not is_mcp_available():
    print("MCP not available. Install with: pip install mcp")
    exit(1)

try:
    tools = await connect_mcp_servers(servers)
except Exception as e:
    print(f"Failed to connect: {e}")
    # Check:
    # - Node.js is installed (for npx)
    # - Required environment variables are set
    # - Server package names are correct
```

## Best Practices

### 1. **Limit File System Access**

```python
# Good: Restrict to specific directory
MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/agent-workspace"]
)

# Bad: Full file system access
MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/"]
)
```

### 2. **Use Environment Variables for Secrets**

```python
# Good
env={"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")}

# Bad
env={"GITHUB_TOKEN": "ghp_hardcoded_token"}
```

### 3. **Tool Naming**

MCP tools are automatically prefixed with server name:
- `filesystem_read_file`
- `github_create_issue`
- `postgres_query`

This prevents naming conflicts between servers.

### 4. **Async Context**

MCP servers require async/await:

```python
# Correct
async def main():
    tools = await connect_mcp_servers(servers)
    agent = create_gemma_agent(tools=tools)
    result = agent.run("query")  # agent.run() is sync

asyncio.run(main())

# Wrong
tools = connect_mcp_servers(servers)  # Missing await
```

## Troubleshooting

### "MCP SDK not installed"

```bash
pip install mcp
```

### "npx: command not found"

```bash
# Install Node.js
# macOS
brew install node

# Ubuntu/Debian
sudo apt install nodejs npm

# Windows
# Download from nodejs.org
```

### "Failed to connect to MCP server"

1. Check server package name:
   ```bash
   npx -y @modelcontextprotocol/server-filesystem --help
   ```

2. Check environment variables:
   ```python
   print(os.getenv("GITHUB_TOKEN"))  # Should not be None
   ```

3. Check server logs (MCP servers output to stderr)

### "Tool execution failed"

- Check tool input schema matches what agent is sending
- Verify permissions (file access, API tokens, etc.)
- Look at agent verbose logs for details

## Architecture

```
┌─────────────┐
│   Agent     │
│  (OpenAI)   │
└──────┬──────┘
       │
       │ Uses tools
       │
       ├─────────────┬─────────────┬─────────────┐
       │             │             │             │
┌──────▼──────┐ ┌───▼────┐  ┌─────▼─────┐ ┌────▼─────┐
│ Default     │ │  MCP   │  │    MCP    │ │   MCP    │
│ Tools       │ │ Tool 1 │  │   Tool 2  │ │  Tool N  │
│ (calc, etc) │ └───┬────┘  └─────┬─────┘ └────┬─────┘
└─────────────┘     │             │            │
                    │             │            │
              ┌─────▼─────────────▼────────────▼─────┐
              │        MCP Client Manager            │
              └─────┬──────────────┬──────────────┬──┘
                    │              │              │
              ┌─────▼────┐   ┌─────▼────┐   ┌────▼─────┐
              │   MCP    │   │   MCP    │   │   MCP    │
              │ Server 1 │   │ Server 2 │   │ Server N │
              │(filesys) │   │ (github) │   │ (postgres)│
              └──────────┘   └──────────┘   └──────────┘
```

## Resources

- **MCP Specification**: https://modelcontextprotocol.io/
- **Official Servers**: https://github.com/modelcontextprotocol
- **Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **Example**: `examples/mcp_example.py`

## What's Next?

1. **Build Custom MCP Servers** - Create your own servers for custom integrations
2. **Use Community Servers** - Explore third-party MCP servers
3. **Multi-Agent MCP** - Use different MCP servers with different agents

---

**Added**: 2025-01-XX
**Status**: ✅ Production Ready
**Dependencies**: `pip install mcp` (optional)
