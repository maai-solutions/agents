"""Test script for the MCP Vector Store Server.

This script tests the MCP server functionality without requiring a full MCP client.
"""

import asyncio
import json
from server import VectorStoreMCPServer


async def test_vector_search():
    """Test the vector search functionality."""
    print("Initializing MCP Vector Store Server...")

    try:
        server = VectorStoreMCPServer()
        print("✓ Server initialized successfully")

        # Test search
        print("\nTesting vector search...")
        query = "machine learning algorithms"
        result = await server._vector_search(
            query=query,
            limit=3
        )

        print(f"\n{'='*80}")
        print("SEARCH RESULTS")
        print(f"{'='*80}")
        print(result)
        print(f"{'='*80}\n")

        print("✓ Vector search completed successfully")

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'server' in locals():
            del server


async def test_list_tools():
    """Test listing available tools."""
    print("\nTesting tool listing...")

    try:
        server = VectorStoreMCPServer()

        # Manually call the list_tools handler
        tools_handler = None
        for handler in server.server._list_tools_handlers:
            tools_handler = handler
            break

        if tools_handler:
            tools = await tools_handler()
            print(f"\nAvailable tools ({len(tools)}):")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
                print(f"    Input schema: {json.dumps(tool.inputSchema, indent=2)}")

        print("\n✓ Tool listing completed successfully")

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'server' in locals():
            del server


async def main():
    """Main test runner."""
    print("="*80)
    print("MCP VECTOR STORE SERVER TEST")
    print("="*80)

    print("\n1. Testing tool listing...")
    await test_list_tools()

    print("\n2. Testing vector search...")
    await test_vector_search()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
