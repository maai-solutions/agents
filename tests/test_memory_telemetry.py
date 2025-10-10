"""Test memory telemetry integration."""

import os
import sys
from openai import OpenAI

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.agent.memory import create_memory_manager
from linus.agents.telemetry import initialize_telemetry
from loguru import logger


def test_memory_with_langfuse():
    """Test memory operations with Langfuse telemetry."""
    logger.info("=== Testing Memory with Langfuse Telemetry ===")

    # Initialize Langfuse tracer
    tracer = initialize_telemetry(
        service_name="memory-test",
        exporter_type="langfuse",
        session_id="test-session-123",
        enabled=True
    )

    # Create LLM client for memory summarization
    llm = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed"
    )

    # Create memory manager with telemetry
    memory = create_memory_manager(
        backend_type="in_memory",
        llm=llm,
        model="gemma3:27b",
        max_context_tokens=2048,
        max_size=100,
        tracer=tracer
    )

    logger.info("Memory manager created with Langfuse tracer")

    # Test 1: Add some memories
    logger.info("\n--- Test 1: Adding memories ---")
    memory.add_memory(
        "User asked about the weather",
        entry_type="interaction",
        importance=0.8
    )
    memory.add_memory(
        "Provided weather information for San Francisco",
        entry_type="interaction",
        importance=0.7
    )
    memory.add_memory(
        "User requested code example for API call",
        entry_type="interaction",
        importance=0.9
    )

    # Test 2: Search memories
    logger.info("\n--- Test 2: Searching memories ---")
    results = memory.search_memories("weather", limit=5)
    logger.info(f"Found {len(results)} results for 'weather'")

    # Test 3: Get context
    logger.info("\n--- Test 3: Getting context ---")
    context = memory.get_context(max_tokens=1000, include_summary=False)
    logger.info(f"Generated context: {len(context)} characters")

    # Test 4: Get memory stats
    logger.info("\n--- Test 4: Memory statistics ---")
    stats = memory.get_memory_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Flush traces to Langfuse
    logger.info("\n--- Flushing traces to Langfuse ---")
    tracer.flush()

    logger.success("Memory telemetry test completed!")
    logger.info("Check Langfuse dashboard for memory operation traces")


def test_memory_with_console():
    """Test memory operations with console telemetry."""
    logger.info("\n\n=== Testing Memory with Console Telemetry ===")

    # Initialize console tracer
    tracer = initialize_telemetry(
        service_name="memory-test",
        exporter_type="console",
        enabled=True
    )

    # Create memory manager with telemetry
    memory = create_memory_manager(
        backend_type="in_memory",
        max_context_tokens=1024,
        max_size=50,
        tracer=tracer
    )

    logger.info("Memory manager created with console tracer")

    # Add and search memories
    memory.add_memory("Test memory entry 1", importance=0.5)
    memory.add_memory("Test memory entry 2", importance=0.8)

    results = memory.search_memories("test", limit=2)
    logger.info(f"Search found {len(results)} results")

    context = memory.get_context(max_tokens=500)
    logger.info(f"Context: {len(context)} characters")

    logger.success("Console telemetry test completed!")


def test_memory_without_telemetry():
    """Test memory operations without telemetry (baseline)."""
    logger.info("\n\n=== Testing Memory without Telemetry ===")

    # Create memory manager without tracer
    memory = create_memory_manager(
        backend_type="in_memory",
        max_context_tokens=1024,
        max_size=50
    )

    logger.info("Memory manager created without tracer")

    # Add and search memories
    memory.add_memory("No telemetry entry 1")
    memory.add_memory("No telemetry entry 2")

    results = memory.search_memories("telemetry", limit=2)
    logger.info(f"Search found {len(results)} results")

    context = memory.get_context(max_tokens=500)
    logger.info(f"Context: {len(context)} characters")

    logger.success("No-telemetry test completed!")


if __name__ == "__main__":
    # Test without telemetry first (baseline)
    test_memory_without_telemetry()

    # Test with console telemetry
    test_memory_with_console()

    # Test with Langfuse telemetry (requires credentials)
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        test_memory_with_langfuse()
    else:
        logger.warning("Skipping Langfuse test - credentials not found in environment")
        logger.info("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to test Langfuse integration")
