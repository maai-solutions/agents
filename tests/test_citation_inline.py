#!/usr/bin/env python3
"""Test inline citations with vector_search tool."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.tools.vector_store import VectorStoreTool
from linus.settings.settings import Settings

async def test_inline_citations():
    """Test inline citation formatting in responses."""
    print("="*80)
    print("Testing Inline Citations with Vector Search")
    print("="*80)

    # Load settings
    settings = Settings()

    # Get tools including vector_search
    tools = get_default_tools() + [VectorStoreTool()]
    print(f"\nTools available: {[tool.name for tool in tools]}")

    # Create agent with vector_search tool
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
    query = "What is the Ukraine conflict about?"
    print(f"\nQuery: {query}")
    print("="*80)

    # Run agent
    response = await agent.run(query, return_metrics=True)

    # Display results
    print("\n" + "="*80)
    print("FINAL RESPONSE")
    print("="*80)
    print(f"\n{response.result}\n")

    # Check citations
    print("\n" + "="*80)
    print("EXTRACTED CITATIONS")
    print("="*80)

    if response.citations:
        print(f"\n✅ Extracted {len(response.citations)} citation(s):\n")
        for idx, citation in enumerate(response.citations, 1):
            print(f"[{idx}] Document: {citation.document_id}, Chunk: {citation.chunk_number}")
            if citation.score:
                print(f"    Score: {citation.score:.4f}")
            if citation.content_preview:
                print(f"    Preview: {citation.content_preview[:150]}...")
            print()
    else:
        print("❌ No citations found!")

    # Check if response contains inline citations
    print("\n" + "="*80)
    print("INLINE CITATION CHECK")
    print("="*80)

    result_str = str(response.result)
    has_inline_citations = False
    citation_count = 0

    for i in range(1, 20):  # Check for [1] through [20]
        if f"[{i}]" in result_str:
            has_inline_citations = True
            citation_count = max(citation_count, i)

    if has_inline_citations:
        print(f"✅ Response contains inline citations: {citation_count} citation reference(s) found")
    else:
        print("⚠️  Response does not contain inline citations like [1], [2], etc.")
        print("    The LLM may need better prompting or the format may differ.")

    # Check for References section
    if "## References" in result_str or "References:" in result_str or "Citations:" in result_str:
        print("✅ Response contains a References/Citations section at the bottom")
    else:
        print("⚠️  Response does not contain a References section at the bottom")

    # Display metrics
    print("\n" + "="*80)
    print("EXECUTION METRICS")
    print("="*80)
    print(f"\nIterations: {response.metrics.total_iterations}")
    print(f"Tool executions: {response.metrics.tool_executions}")
    print(f"Execution time: {response.metrics.execution_time_seconds:.2f}s")

    return response

if __name__ == "__main__":
    try:
        asyncio.run(test_inline_citations())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
