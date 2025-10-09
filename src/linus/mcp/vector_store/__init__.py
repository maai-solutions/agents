"""MCP Vector Store Server - Provides vector search capabilities via MCP protocol."""

from .server import VectorStoreMCPServer, main

__all__ = ["VectorStoreMCPServer", "main"]
