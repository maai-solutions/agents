"""Agent module with OpenAI client integration."""

# Core data models
from .models import (
    ReasoningResult,
    TaskExecution,
    AgentMetrics,
    AgentResponse
)

# Base agent class
from .base import Agent

# ReasoningAgent implementation
from .reasoning_agent import ReasoningAgent

# Factory functions
from .factory import Agent

# Tool base classes
from .tool_base import BaseTool, StructuredTool, tool

# Tools (already exists)
from .tools import get_default_tools, create_custom_tool

# Memory (already exists, optional)
try:
    from .memory import MemoryManager, create_memory_manager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# MCP client (optional)
try:
    from .mcp_client import (
        MCPClientManager,
        MCPServerConfig,
        MCPTool,
        connect_mcp_servers,
        is_mcp_available
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


__all__ = [
    # Data models
    "ReasoningResult",
    "TaskExecution",
    "AgentMetrics",
    "AgentResponse",
    # Classes
    "Agent",
    "ReasoningAgent",
    # Factory
    "Agent",
    # Tool base classes
    "BaseTool",
    "StructuredTool",
    "tool",
    # Tools
    "get_default_tools",
    "create_custom_tool",
    # Memory (if available)
    "MemoryManager",
    "create_memory_manager",
    "MEMORY_AVAILABLE",
    # MCP (if available)
    "MCPClientManager",
    "MCPServerConfig",
    "MCPTool",
    "connect_mcp_servers",
    "is_mcp_available",
    "MCP_AVAILABLE",
]
