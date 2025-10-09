"""Entry point for running the MCP Vector Store Server as a module."""

import asyncio
from .server import main

if __name__ == "__main__":
    asyncio.run(main())
