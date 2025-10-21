#!/usr/bin/env python3
"""Test inline citations with a mock tool that returns citations."""

import asyncio
import sys
import os
import json
from typing import Type, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pydantic import BaseModel, Field
from linus.agents.agent.factory import Agent
from linus.agents.agent.tool_base import BaseTool
from linus.agents.agent.tools import get_default_tools
from linus.settings.settings import Settings


class MockSearchInput(BaseModel):
    """Input for the mock search tool."""
    query: str = Field(description="Search query")


class MockSearchWithCitationsTool(BaseTool):
    """Mock tool that returns structured data with citations."""

    name: str = "mock_search"
    description: str = "Search for information (mock tool with citations)"
    args_schema: Type[BaseModel] = MockSearchInput

    def _run(self, query: str) -> str:
        """Return mock data with citations."""
        # Simulate a tool that returns results with citations
        results = {
            "status": "success",
            "query": query,
            "results": [
                {
                    "content": "Ukraine is a country in Eastern Europe. It shares borders with Russia to the east and northeast.",
                    "document_id": "doc_ukraine_001",
                    "chunk_number": 1
                },
                {
                    "content": "The conflict began in 2014 when Russia annexed Crimea from Ukraine.",
                    "document_id": "doc_conflict_002",
                    "chunk_number": 3
                },
                {
                    "content": "The situation escalated in February 2022 with a full-scale invasion by Russian forces.",
                    "document_id": "doc_conflict_002",
                    "chunk_number": 5
                }
            ],
            "citations": [
                {
                    "document_id": "doc_ukraine_001",
                    "chunk_number": 1,
                    "score": 0.95,
                    "content_preview": "Ukraine is a country in Eastern Europe. It shares borders with Russia..."
                },
                {
                    "document_id": "doc_conflict_002",
                    "chunk_number": 3,
                    "score": 0.92,
                    "content_preview": "The conflict began in 2014 when Russia annexed Crimea from Ukraine..."
                },
                {
                    "document_id": "doc_conflict_002",
                    "chunk_number": 5,
                    "score": 0.88,
                    "content_preview": "The situation escalated in February 2022 with a full-scale invasion..."
                }
            ]
        }
        return json.dumps(results, indent=2)

    async def _arun(self, query: str) -> str:
        """Async version."""
        return self._run(query)


async def test_inline_citations():
    """Test inline citation formatting with mock tool."""
    print("="*80)
    print("Testing Inline Citations with Mock Search Tool")
    print("="*80)

    # Load settings
    settings = Settings()

    # Create tools with our mock tool
    tools = [MockSearchWithCitationsTool()]
    print(f"\nTools available: {[tool.name for tool in tools]}")

    # Create agent (override model to use gemma3:27b which is available)
    agent = Agent(
        api_base=settings.llm_api_base,
        model="gemma3:27b",  # Use a model that's available
        api_key=settings.llm_api_key,
        temperature=0.7,
        tools=tools,
        verbose=True,
        use_async=True
    )

    # Test query
    query = "Tell me about Ukraine and the conflict"
    print(f"\nQuery: {query}")
    print("="*80)

    # Run agent
    response = await agent.run(query, return_metrics=True)

    # Display results
    print("\n" + "="*80)
    print("FINAL RESPONSE")
    print("="*80)
    print(f"\n{response.result}\n")

    # Check extracted citations
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
                print(f"    Preview: {citation.content_preview[:100]}...")
            print()
    else:
        print("❌ No citations extracted!")

    # Check if response contains inline citations
    print("\n" + "="*80)
    print("INLINE CITATION ANALYSIS")
    print("="*80)

    result_str = str(response.result)

    # Check for inline citation markers
    citation_markers = []
    for i in range(1, 20):
        if f"[{i}]" in result_str:
            citation_markers.append(i)

    if citation_markers:
        print(f"✅ Found inline citation markers: {citation_markers}")
    else:
        print("⚠️  No inline citation markers like [1], [2] found in response")

    # Check for References section
    has_references = False
    for keyword in ["References", "Citations", "Sources"]:
        if keyword in result_str:
            has_references = True
            print(f"✅ Found '{keyword}' section in response")
            break

    if not has_references:
        print("⚠️  No References/Citations section found at bottom")

    # Display execution history
    print("\n" + "="*80)
    print("EXECUTION HISTORY")
    print("="*80)

    for item in response.execution_history:
        print(f"\nTask: {item['task']}")
        print(f"Tool: {item['tool']}")
        print(f"Status: {item['status']}")
        if item.get('result'):
            result_preview = str(item['result'])[:200]
            print(f"Result: {result_preview}...")

    # Display metrics
    print("\n" + "="*80)
    print("METRICS")
    print("="*80)
    print(f"Iterations: {response.metrics.total_iterations}")
    print(f"Tool executions: {response.metrics.tool_executions}")
    print(f"Execution time: {response.metrics.execution_time_seconds:.2f}s")
    print(f"Total tokens: {response.metrics.total_tokens}")

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
