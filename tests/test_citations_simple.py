#!/usr/bin/env python3
"""Simple test to verify citation extraction."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.settings.settings import Settings

async def test_citations():
    """Test citation extraction with vector_search tool."""
    print("="*80)
    print("Citation Test - Checking if vector_search tool is available")
    print("="*80)

    # Load settings
    settings = Settings()

    # Get default tools
    tools = get_default_tools()
    print(f"\nDefault tools: {[tool.name for tool in tools]}")

    # Check if vector_search is in the tools
    has_vector_search = any(tool.name == "vector_search" for tool in tools)
    print(f"Has vector_search tool: {has_vector_search}")

    if not has_vector_search:
        print("\n⚠️  WARNING: vector_search tool is NOT in default tools!")
        print("Citations will only work if vector_search tool is included.")
        print("\nTo fix this, you need to:")
        print("1. Import VectorStoreTool from linus.agents.tools.vector_store")
        print("2. Add VectorStoreTool() to your agent's tools list")
        print("\nExample:")
        print("  from linus.agents.tools.vector_store import VectorStoreTool")
        print("  from linus.agents.agent.tools import get_default_tools")
        print("  tools = get_default_tools() + [VectorStoreTool()]")
        print("  agent = Agent(..., tools=tools)")
        return

    # Create agent with tools
    agent = Agent(
        api_base=settings.llm_api_base,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0.7,
        tools=tools,
        verbose=True,
        use_async=True
    )

    # Test query that should use vector_search
    query = "Search for information about Russia and Ukraine"
    print(f"\nQuery: {query}")
    print("="*80)

    # Run agent
    response = await agent.run(query, return_metrics=True)

    # Check results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nResponse: {response.result[:200]}...")

    # Check citations
    print("\n" + "="*80)
    print("CITATIONS")
    print("="*80)

    if response.citations:
        print(f"✅ Found {len(response.citations)} citation(s)!")
        for idx, citation in enumerate(response.citations, 1):
            print(f"\n{idx}. Document: {citation.document_id}")
            print(f"   Chunk: {citation.chunk_number}")
            if citation.score:
                print(f"   Score: {citation.score:.4f}")
            if citation.content_preview:
                print(f"   Preview: {citation.content_preview[:100]}...")
    else:
        print("❌ No citations found!")
        print("\nPossible reasons:")
        print("1. vector_search tool was not used (check execution history)")
        print("2. vector_search returned no results")
        print("3. Citation extraction logic has a bug")

    # Check execution history
    print("\n" + "="*80)
    print("EXECUTION HISTORY")
    print("="*80)

    if response.execution_history:
        for item in response.execution_history:
            print(f"\nTask: {item['task']}")
            print(f"Tool: {item['tool']}")
            print(f"Status: {item['status']}")
            if item['tool'] == 'vector_search':
                print(f"Result preview: {str(item['result'])[:300]}...")

    return response

if __name__ == "__main__":
    asyncio.run(test_citations())
