#!/usr/bin/env python3
"""Test script to verify citation extraction from vector_search tool."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.agent.factory import Agent
from linus.agents.agent.tools import get_default_tools
from linus.agents.tools.vector_store import VectorStoreTool
from linus.agents.logging_config import setup_rich_logging
from linus.settings.settings import Settings
from loguru import logger

# Setup logging
setup_rich_logging(level="INFO")

async def main():
    """Test citation extraction with vector_search tool."""
    logger.info("="*80)
    logger.info("Citation Extraction Test")
    logger.info("="*80)

    # Load settings
    settings = Settings()

    # Create tools list including vector_search
    try:
        vector_tool = VectorStoreTool()
        tools = get_default_tools() + [vector_tool]
        logger.info(f"✅ Created agent with {len(tools)} tools including vector_search")
    except Exception as e:
        logger.error(f"❌ Failed to create VectorStoreTool: {e}")
        logger.error("Make sure Weaviate is running and configured properly")
        return

    # Create agent
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
    query = "What is the current situation in Ukraine and the conflict with Russia?"
    logger.info(f"\nQuery: {query}\n")

    # Run agent
    logger.info("Running agent...")
    response = await agent.run(query, return_metrics=True)

    # Display results
    logger.info("\n" + "="*80)
    logger.info("RESPONSE")
    logger.info("="*80)
    print(f"\n{response.result}\n")

    # Display citations
    logger.info("="*80)
    logger.info("CITATIONS")
    logger.info("="*80)

    if response.citations:
        logger.info(f"✅ Found {len(response.citations)} citation(s)!")
        print()
        for idx, citation in enumerate(response.citations, 1):
            print(f"{idx}. Document: {citation.document_id}")
            print(f"   Chunk: {citation.chunk_number}")
            if citation.score:
                print(f"   Score: {citation.score:.4f}")
            if citation.content_preview:
                preview = citation.content_preview[:100] + "..." if len(citation.content_preview) > 100 else citation.content_preview
                print(f"   Preview: {preview}")
            print()
    else:
        logger.warning("❌ No citations found!")
        logger.info("\nCheck the logs above for [CITATIONS] messages to debug")

    # Display metrics
    logger.info("="*80)
    logger.info("METRICS")
    logger.info("="*80)
    metrics_dict = response.metrics.to_dict()
    for key, value in metrics_dict.items():
        print(f"  {key}: {value}")

    # Display execution history
    if response.execution_history:
        logger.info("\n" + "="*80)
        logger.info("EXECUTION HISTORY")
        logger.info("="*80)
        for item in response.execution_history:
            print(f"\nIteration {item['iteration']}: {item['task']}")
            print(f"  Tool: {item['tool']}")
            print(f"  Status: {item['status']}")
            # Show whether vector_search was used
            if item['tool'] == 'vector_search':
                logger.info("  ✅ vector_search tool was used")

if __name__ == "__main__":
    asyncio.run(main())
