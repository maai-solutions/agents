"""Test script to demonstrate citation support in the agent system."""

import asyncio
from linus.agents.agent.factory import Agent
from linus.agents.tools.vector_store import VectorStoreTool
from linus.settings.settings import Settings


async def test_citations():
    """Test citation extraction from vector search."""
    print("=" * 80)
    print("Testing Citation Support")
    print("=" * 80)

    # Load settings
    settings = Settings()

    # Create vector search tool
    vector_tool = VectorStoreTool()

    # Create agent with vector search tool
    agent = Agent(
        api_base=settings.llm_api_base,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0.7,
        tools=[vector_tool],
        verbose=True,
        use_async=True
    )

    # Test query
    query = "What is the relationship between Russia and Ukraine?"

    print(f"\nQuery: {query}\n")

    # Run agent
    response = await agent.run(query, return_metrics=True)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nResponse:\n{response.result}")

    # Display citations
    if response.citations:
        print("\n" + "=" * 80)
        print(f"CITATIONS ({len(response.citations)} found)")
        print("=" * 80)
        for idx, citation in enumerate(response.citations, 1):
            print(f"\n{idx}. Document: {citation.document_id}")
            print(f"   Chunk: {citation.chunk_number}")
            if citation.score:
                print(f"   Score: {citation.score:.4f}")
            if citation.content_preview:
                print(f"   Preview: {citation.content_preview}")
    else:
        print("\nNo citations found.")

    # Display metrics
    print("\n" + "=" * 80)
    print("METRICS")
    print("=" * 80)
    metrics_dict = response.metrics.to_dict()
    for key, value in metrics_dict.items():
        print(f"  {key}: {value}")

    # Display execution history
    if response.execution_history:
        print("\n" + "=" * 80)
        print("EXECUTION HISTORY")
        print("=" * 80)
        for item in response.execution_history:
            print(f"\nIteration {item['iteration']}: {item['task']}")
            print(f"  Tool: {item['tool']}")
            print(f"  Status: {item['status']}")
            # Only show first 200 chars of result
            result_preview = str(item['result'])[:200]
            print(f"  Result: {result_preview}...")


async def test_citations_api_response():
    """Test citation format in API response model."""
    from linus.agents.agent.models import AgentResponse, AgentMetrics, Citation

    print("\n" + "=" * 80)
    print("Testing AgentResponse with Citations")
    print("=" * 80)

    # Create sample citations
    citations = [
        Citation(
            document_id="doc_123",
            chunk_number=5,
            score=0.95,
            content_preview="This is a preview of the content from the document..."
        ),
        Citation(
            document_id="doc_456",
            chunk_number=12,
            score=0.87,
            content_preview="Another relevant piece of information..."
        )
    ]

    # Create agent response with citations
    metrics = AgentMetrics(
        total_iterations=1,
        total_tokens=1500,
        execution_time_seconds=2.5,
        llm_calls=3,
        tool_executions=1,
        successful_tool_calls=1,
        task_completed=True
    )

    response = AgentResponse(
        result="The search found relevant information in 2 documents.",
        metrics=metrics,
        execution_history=[{
            "iteration": 1,
            "task": "Search for information",
            "tool": "vector_search",
            "result": "Found 2 results",
            "status": "completed"
        }],
        citations=citations
    )

    # Convert to dictionary (as would be sent in API response)
    response_dict = response.to_dict()

    print("\nAgentResponse as dictionary:")
    import json
    print(json.dumps(response_dict, indent=2))

    print("\n" + "=" * 80)
    print("Citations in response:")
    print("=" * 80)
    for citation_dict in response_dict['citations']:
        print(f"\n- Document: {citation_dict['document_id']}")
        print(f"  Chunk: {citation_dict['chunk_number']}")
        print(f"  Score: {citation_dict['score']}")
        print(f"  Preview: {citation_dict['content_preview'][:50]}...")


if __name__ == "__main__":
    # Test citation extraction from agent run
    print("\n=== Test 1: Agent Citation Extraction ===")
    asyncio.run(test_citations())

    # Test API response format
    print("\n\n=== Test 2: API Response Format ===")
    asyncio.run(test_citations_api_response())

    print("\n" + "=" * 80)
    print("Citation Testing Complete!")
    print("=" * 80)
