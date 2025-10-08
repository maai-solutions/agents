"""Example: Using MCP (Model Context Protocol) servers with the agent.

This example shows how to connect to MCP servers and use their tools
with the ReasoningAgent.

Prerequisites:
    pip install mcp

Common MCP Servers:
    - @modelcontextprotocol/server-filesystem - File operations
    - @modelcontextprotocol/server-github - GitHub API
    - @modelcontextprotocol/server-postgres - Database queries
    - @modelcontextprotocol/server-puppeteer - Web automation
    - @modelcontextprotocol/server-slack - Slack integration

Installation:
    # These are Node.js packages, typically run via npx
    npx -y @modelcontextprotocol/server-filesystem /path/to/directory
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent import (
    create_gemma_agent,
    MCPServerConfig,
    connect_mcp_servers,
    is_mcp_available,
    get_default_tools
)


async def main():
    """Main example function."""

    # Check if MCP is available
    if not is_mcp_available():
        print("❌ MCP SDK not installed")
        print("Install with: pip install mcp")
        return

    print("✅ MCP SDK available")

    # Configure MCP servers
    # Example 1: Filesystem server
    # This gives the agent access to file operations
    servers = {
        "filesystem": MCPServerConfig(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/tmp"  # Root directory for file operations
            ]
        )
    }

    # Example 2: GitHub server (requires GITHUB_TOKEN)
    # Uncomment to use:
    # servers["github"] = MCPServerConfig(
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-github"],
    #     env={"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", "")}
    # )

    # Example 3: Postgres server (requires DATABASE_URL)
    # servers["postgres"] = MCPServerConfig(
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-postgres"],
    #     env={"DATABASE_URL": os.getenv("DATABASE_URL", "")}
    # )

    print("\n🔌 Connecting to MCP servers...")
    try:
        # Connect to MCP servers and get their tools
        mcp_tools = await connect_mcp_servers(servers)
        print(f"✅ Connected! Available MCP tools: {len(mcp_tools)}")

        for tool in mcp_tools:
            print(f"  - {tool.name}: {tool.description}")

        # Combine with default tools
        all_tools = get_default_tools() + mcp_tools

        # Create agent with all tools
        print("\n🤖 Creating agent with MCP tools...")
        agent = create_gemma_agent(
            api_base="http://localhost:11434/v1",
            model="gemma3:27b",
            api_key="not-needed",
            temperature=0.7,
            tools=all_tools,
            verbose=True
        )

        print("✅ Agent created with MCP tools\n")

        # Example queries that use MCP tools
        queries = [
            # Filesystem operations
            "List all files in the /tmp directory",
            "Create a file called test.txt with content 'Hello from MCP!'",
            "Read the contents of test.txt",

            # GitHub operations (if configured)
            # "List my GitHub repositories",
            # "Get issues from repository owner/repo",

            # Database operations (if configured)
            # "Show me all tables in the database",
            # "Query the users table",
        ]

        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")

            try:
                # Run the agent
                result = agent.run(query, return_metrics=True)

                print(f"\n📝 Result: {result.result}")
                print(f"\n📊 Metrics:")
                print(f"  - Iterations: {result.metrics.total_iterations}")
                print(f"  - Tool calls: {result.metrics.tool_executions}")
                print(f"  - Time: {result.metrics.execution_time_seconds:.2f}s")

            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to connect to MCP servers: {e}")
        print("\nMake sure:")
        print("  1. Node.js is installed (for npx)")
        print("  2. MCP server packages are accessible")
        print("  3. Required environment variables are set")


async def simple_example():
    """Simple example with just filesystem access."""

    if not is_mcp_available():
        print("❌ MCP SDK not installed. Install with: pip install mcp")
        return

    # Single filesystem server
    servers = {
        "fs": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
    }

    # Connect and get tools
    mcp_tools = await connect_mcp_servers(servers)

    # Create agent
    agent = create_gemma_agent(
        api_base="http://localhost:11434/v1",
        model="gemma3:27b",
        api_key="not-needed",
        tools=mcp_tools
    )

    # Use it
    result = agent.run("List files in /tmp directory")
    print(result)


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())

    # Or run the simple example:
    # asyncio.run(simple_example())
