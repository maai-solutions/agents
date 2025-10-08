"""MCP (Model Context Protocol) client integration for agent tools.

This module provides integration with MCP servers, allowing the agent to use
MCP tools as regular agent tools.

Install MCP SDK:
    pip install mcp

Example MCP servers:
    - filesystem: File operations
    - github: GitHub API
    - postgres: Database queries
    - puppeteer: Web automation
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import json
from loguru import logger

from .tool_base import BaseTool

# MCP SDK imports (optional dependency)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not installed. Install with: pip install mcp")


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    command: str
    args: List[str] = None
    env: Dict[str, str] = None

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}


class MCPTool(BaseTool):
    """Wrapper for MCP server tools as agent tools."""

    def __init__(
        self,
        name: str,
        description: str,
        mcp_tool_name: str,
        session: 'ClientSession',
        input_schema: Optional[Dict[str, Any]] = None
    ):
        """Initialize MCP tool wrapper.

        Args:
            name: Tool name for the agent
            description: Tool description
            mcp_tool_name: Name of the tool in the MCP server
            session: Active MCP client session
            input_schema: JSON schema for tool inputs
        """
        self.name = name
        self.description = description
        self.mcp_tool_name = mcp_tool_name
        self.session = session
        self.input_schema_dict = input_schema or {}
        self.args_schema = None  # MCP tools use JSON schema

    def _run(self, **kwargs) -> str:
        """Execute the MCP tool (sync wrapper)."""
        # Run async in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        """Execute the MCP tool asynchronously."""
        try:
            logger.debug(f"[MCP] Calling tool '{self.mcp_tool_name}' with args: {kwargs}")

            # Call the MCP tool
            result = await self.session.call_tool(
                name=self.mcp_tool_name,
                arguments=kwargs
            )

            # Extract content from result
            if hasattr(result, 'content') and result.content:
                # MCP returns list of content items
                content_parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content_parts.append(item.text)
                    elif hasattr(item, 'data'):
                        content_parts.append(json.dumps(item.data))
                    else:
                        content_parts.append(str(item))

                output = "\n".join(content_parts)
            else:
                output = str(result)

            logger.debug(f"[MCP] Tool result: {output[:200]}...")
            return output

        except Exception as e:
            error_msg = f"Error calling MCP tool '{self.mcp_tool_name}': {str(e)}"
            logger.error(f"[MCP] {error_msg}")
            return error_msg


class MCPClientManager:
    """Manages MCP server connections and provides tools to the agent."""

    def __init__(self):
        """Initialize MCP client manager."""
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP SDK not installed. Install with: pip install mcp"
            )

        self.sessions: Dict[str, ClientSession] = {}
        self.tools: List[MCPTool] = []

    async def connect_server(
        self,
        server_name: str,
        config: MCPServerConfig
    ) -> None:
        """Connect to an MCP server.

        Args:
            server_name: Unique name for this server
            config: Server configuration

        Example:
            config = MCPServerConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            )
            await manager.connect_server("filesystem", config)
        """
        try:
            logger.info(f"[MCP] Connecting to server '{server_name}'...")

            # Create server parameters
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env
            )

            # Create session
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()

                    # Store session
                    self.sessions[server_name] = session

                    # List available tools
                    tools_result = await session.list_tools()

                    logger.info(
                        f"[MCP] Connected to '{server_name}'. "
                        f"Available tools: {len(tools_result.tools)}"
                    )

                    # Create tool wrappers
                    for tool in tools_result.tools:
                        mcp_tool = MCPTool(
                            name=f"{server_name}_{tool.name}",
                            description=tool.description or f"MCP tool: {tool.name}",
                            mcp_tool_name=tool.name,
                            session=session,
                            input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else None
                        )
                        self.tools.append(mcp_tool)
                        logger.debug(f"[MCP] Registered tool: {mcp_tool.name}")

        except Exception as e:
            logger.error(f"[MCP] Failed to connect to server '{server_name}': {e}")
            raise

    def get_tools(self) -> List[BaseTool]:
        """Get all MCP tools as agent tools.

        Returns:
            List of MCP tools wrapped as agent tools
        """
        return self.tools

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for server_name in list(self.sessions.keys()):
            await self.disconnect_server(server_name)

    async def disconnect_server(self, server_name: str) -> None:
        """Disconnect from a specific MCP server.

        Args:
            server_name: Name of the server to disconnect
        """
        if server_name in self.sessions:
            try:
                # Sessions are managed by context managers
                # Just remove from our tracking
                del self.sessions[server_name]

                # Remove associated tools
                self.tools = [
                    t for t in self.tools
                    if not t.name.startswith(f"{server_name}_")
                ]

                logger.info(f"[MCP] Disconnected from server '{server_name}'")
            except Exception as e:
                logger.error(f"[MCP] Error disconnecting from '{server_name}': {e}")


async def connect_mcp_servers(
    servers: Dict[str, MCPServerConfig]
) -> List[BaseTool]:
    """Connect to multiple MCP servers and return their tools.

    Args:
        servers: Dictionary of server_name -> MCPServerConfig

    Returns:
        List of all tools from all servers

    Example:
        servers = {
            "filesystem": MCPServerConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            ),
            "github": MCPServerConfig(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env={"GITHUB_TOKEN": "ghp_..."}
            )
        }

        tools = await connect_mcp_servers(servers)
    """
    manager = MCPClientManager()

    for server_name, config in servers.items():
        await manager.connect_server(server_name, config)

    return manager.get_tools()


def is_mcp_available() -> bool:
    """Check if MCP SDK is available.

    Returns:
        True if MCP SDK is installed, False otherwise
    """
    return MCP_AVAILABLE
