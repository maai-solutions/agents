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
    cwd: Optional[str] = None

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

        # Create a minimal schema-like object compatible with Pydantic v2
        # This allows the agent to access the parameter schema for argument generation
        class MCPToolSchema:
            @staticmethod
            def model_json_schema():
                """Return JSON schema compatible with Pydantic v2."""
                return input_schema or {}

        self.args_schema = MCPToolSchema

    async def _arun(self, **kwargs) -> str:
        """Internal async execution method.

        This is called by BaseTool.arun() after validation.
        For MCP tools, we override arun() to skip Pydantic validation.
        """
        # This shouldn't be called directly since we override arun()
        # But we need to implement it to satisfy the abstract method requirement
        return await self.arun(kwargs)

    async def arun(self, tool_input: Dict[str, Any]) -> str:
        """Execute the MCP tool asynchronously.

        MCP tools have their own JSON schema validation, so we bypass
        the BaseTool Pydantic validation to avoid instantiation errors.

        Args:
            tool_input: Dictionary of tool arguments

        Returns:
            Tool execution result as string
        """
        import asyncio

        try:
            logger.info(f"[MCP] Starting async call to tool '{self.mcp_tool_name}' with args: {tool_input}")

            # Call the MCP tool with timeout
            try:
                result = await asyncio.wait_for(
                    self.session.call_tool(
                        name=self.mcp_tool_name,
                        arguments=tool_input
                    ),
                    timeout=60.0  # 60 second timeout
                )
                logger.info(f"[MCP] Tool '{self.mcp_tool_name}' call completed")
            except asyncio.TimeoutError:
                error_msg = f"Timeout calling MCP tool '{self.mcp_tool_name}' (exceeded 60s)"
                logger.error(f"[MCP] {error_msg}")
                return error_msg

            # Extract content from result
            logger.debug(f"[MCP] Processing result, type: {type(result)}")
            if hasattr(result, 'content') and result.content:
                # MCP returns list of content items
                logger.debug(f"[MCP] Result has {len(result.content)} content item(s)")
                content_parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content_parts.append(item.text)
                        logger.debug(f"[MCP] Extracted text content (length={len(item.text)})")
                    elif hasattr(item, 'data'):
                        content_parts.append(json.dumps(item.data))
                        logger.debug(f"[MCP] Extracted data content")
                    else:
                        content_parts.append(str(item))
                        logger.debug(f"[MCP] Extracted string content")

                output = "\n".join(content_parts)
            else:
                output = str(result)
                logger.debug(f"[MCP] Result has no content attribute, using string representation")

            logger.info(f"[MCP] Tool '{self.mcp_tool_name}' result ready (length={len(output)})")
            logger.debug(f"[MCP] Tool result preview: {output[:200]}...")
            return output

        except Exception as e:
            error_msg = f"Error calling MCP tool '{self.mcp_tool_name}': {str(e)}"
            logger.exception(f"[MCP] {error_msg}")
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
        self._cleanup_tasks: Dict[str, Any] = {}  # Store context managers for cleanup

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
                env=config.env,
                cwd=config.cwd
            )

            # Create and enter the stdio_client context
            stdio_ctx = stdio_client(server_params)
            read, write = await stdio_ctx.__aenter__()

            # Create and enter the ClientSession context
            session_ctx = ClientSession(read, write)
            session = await session_ctx.__aenter__()

            # Initialize the session
            await session.initialize()

            # Store session and contexts for later cleanup
            self.sessions[server_name] = session
            self._cleanup_tasks[server_name] = {
                'session_ctx': session_ctx,
                'stdio_ctx': stdio_ctx
            }

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
                logger.info(f"[MCP] Disconnecting from server '{server_name}'...")
                
                # Clean up the context managers in reverse order with proper error handling
                cleanup = self._cleanup_tasks.get(server_name)
                if cleanup:
                    # Exit session context first
                    try:
                        await cleanup['session_ctx'].__aexit__(None, None, None)
                        logger.debug(f"[MCP] Session context exited for '{server_name}'")
                    except Exception as e:
                        logger.warning(f"[MCP] Error exiting session context for '{server_name}': {e}")
                    
                    # Exit stdio context with timeout and cancellation handling
                    try:
                        import asyncio
                        # Use asyncio.wait_for with shield to prevent cancellation issues
                        await asyncio.wait_for(
                            asyncio.shield(cleanup['stdio_ctx'].__aexit__(None, None, None)),
                            timeout=5.0
                        )
                        logger.debug(f"[MCP] Stdio context exited for '{server_name}'")
                    except asyncio.TimeoutError:
                        logger.warning(f"[MCP] Timeout exiting stdio context for '{server_name}' (5s)")
                    except asyncio.CancelledError:
                        logger.warning(f"[MCP] Stdio context exit cancelled for '{server_name}'")
                    except Exception as e:
                        logger.warning(f"[MCP] Error exiting stdio context for '{server_name}': {e}")
                    
                    del self._cleanup_tasks[server_name]

                # Remove session
                del self.sessions[server_name]

                # Remove associated tools
                self.tools = [
                    t for t in self.tools
                    if not t.name.startswith(f"{server_name}_")
                ]

                logger.info(f"[MCP] Successfully disconnected from server '{server_name}'")
            except Exception as e:
                logger.exception(f"[MCP] Error disconnecting from '{server_name}': {e}")
                # Clean up session and tasks even if there was an error
                if server_name in self.sessions:
                    del self.sessions[server_name]
                if server_name in self._cleanup_tasks:
                    del self._cleanup_tasks[server_name]


async def connect_mcp_servers(
    servers: Dict[str, MCPServerConfig]
) -> tuple[MCPClientManager, List[BaseTool]]:
    """Connect to multiple MCP servers and return manager and tools.

    Args:
        servers: Dictionary of server_name -> MCPServerConfig

    Returns:
        Tuple of (manager, tools) - manager must be kept alive and cleaned up

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

        manager, tools = await connect_mcp_servers(servers)
        # ... use tools ...
        # On shutdown:
        await manager.disconnect_all()
    """
    manager = MCPClientManager()

    for server_name, config in servers.items():
        await manager.connect_server(server_name, config)

    return manager, manager.get_tools()


def is_mcp_available() -> bool:
    """Check if MCP SDK is available.

    Returns:
        True if MCP SDK is installed, False otherwise
    """
    return MCP_AVAILABLE
