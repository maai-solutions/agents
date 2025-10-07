"""Test the ReasoningAgent's memory capabilities."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.agent import create_gemma_agent
from linus.agents.agent.tools import get_default_tools
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/agent_memory_test.log", rotation="10 MB", level="DEBUG")


def test_basic_memory():
    """Test basic memory functionality."""
    print("\n" + "="*80)
    print("TEST 1: Basic Memory - Conversation Context")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        max_iterations=5,
        enable_memory=True,
        max_context_tokens=4096,
        memory_context_ratio=0.3  # Use 30% of context for memory
    )

    # First interaction
    print("\n>>> User: What is 25 + 17?")
    response1 = agent.run("What is 25 + 17?", return_metrics=False)
    print(f"<<< Agent: {response1}\n")

    # Second interaction - should remember context
    print(">>> User: What was the previous calculation?")
    response2 = agent.run("What was the previous calculation?", return_metrics=False)
    print(f"<<< Agent: {response2}\n")

    # Third interaction - reference to earlier context
    print(">>> User: Add 10 to that result")
    response3 = agent.run("Add 10 to that result", return_metrics=False)
    print(f"<<< Agent: {response3}\n")

    # Check memory stats
    if agent.memory_manager:
        stats = agent.memory_manager.get_memory_stats()
        print(f"\n📊 Memory Stats:")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Has summary: {stats['has_summary']}")
        print(f"  Utilization: {stats['utilization']:.1%}")


def test_memory_with_metrics():
    """Test memory with full metrics tracking."""
    print("\n" + "="*80)
    print("TEST 2: Memory with Metrics Tracking")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        max_iterations=5,
        enable_memory=True,
        max_context_tokens=4096,
        memory_context_ratio=0.25
    )

    conversations = [
        "Calculate 42 * 17",
        "What was the result of the previous calculation?",
        "Is that number prime?",
        "What calculations have we done so far?"
    ]

    for i, question in enumerate(conversations, 1):
        print(f"\n[{i}] User: {question}")
        response = agent.run(question, return_metrics=True)
        print(f"    Agent: {str(response.result)[:200]}...")
        print(f"    Metrics: {response.metrics.total_iterations} iterations, "
              f"{response.metrics.total_tokens} tokens, "
              f"{response.metrics.execution_time_seconds:.2f}s")

    # Final memory stats
    if agent.memory_manager:
        stats = agent.memory_manager.get_memory_stats()
        print(f"\n📊 Final Memory Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def test_memory_search():
    """Test memory search functionality."""
    print("\n" + "="*80)
    print("TEST 3: Memory Search")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        enable_memory=True
    )

    # Add several interactions
    interactions = [
        "What is the capital of France?",
        "Calculate 100 / 4",
        "Search for information about Python programming",
        "What is 50 * 2?",
        "Tell me about Paris"
    ]

    print("\nAdding interactions to memory...")
    for q in interactions:
        print(f"  - {q}")
        agent.run(q, return_metrics=False)

    # Search for specific topics
    if agent.memory_manager:
        print("\n🔍 Searching memory for 'calculation':")
        calc_memories = agent.memory_manager.search_memories("calculation", limit=3)
        for mem in calc_memories:
            print(f"  [{mem.timestamp.strftime('%H:%M:%S')}] {mem.content[:100]}")

        print("\n🔍 Searching memory for 'Paris':")
        paris_memories = agent.memory_manager.search_memories("Paris", limit=3)
        for mem in paris_memories:
            print(f"  [{mem.timestamp.strftime('%H:%M:%S')}] {mem.content[:100]}")


def test_memory_export_import():
    """Test exporting and importing memories."""
    print("\n" + "="*80)
    print("TEST 4: Memory Export/Import")
    print("="*80)

    tools = get_default_tools()

    # Create agent and add some memories
    agent1 = create_gemma_agent(tools=tools, verbose=False, enable_memory=True)

    print("\nAgent 1: Adding memories...")
    agent1.run("Calculate 10 + 20", return_metrics=False)
    agent1.run("What is 5 * 6?", return_metrics=False)

    # Export memories
    if agent1.memory_manager:
        exported = agent1.memory_manager.export_memories()
        print(f"Exported {len(exported)} memories")

        # Create new agent and import
        agent2 = create_gemma_agent(tools=tools, verbose=False, enable_memory=True)
        if agent2.memory_manager:
            agent2.memory_manager.import_memories(exported)
            print(f"Imported {agent2.memory_manager.backend.count()} memories to Agent 2")

            # Verify Agent 2 has the context
            print("\nAgent 2: Using imported memory...")
            response = agent2.run("What calculations did we do?", return_metrics=False)
            print(f"Response: {response[:200]}...")


def test_memory_token_management():
    """Test that memory respects token limits."""
    print("\n" + "="*80)
    print("TEST 5: Memory Token Management")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        enable_memory=True,
        max_context_tokens=2048,  # Smaller context window
        memory_context_ratio=0.4  # 40% for memory
    )

    # Add many interactions to fill up memory
    print("\nAdding multiple interactions...")
    for i in range(20):
        agent.run(f"Calculate {i} * {i+1}", return_metrics=False)

    if agent.memory_manager:
        stats = agent.memory_manager.get_memory_stats()
        print(f"\n📊 Memory Stats after 20 interactions:")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Max context tokens: {stats['max_context_tokens']}")
        print(f"  Utilization: {stats['utilization']:.1%}")
        print(f"  Has summary: {stats['has_summary']}")

        # Test that context doesn't exceed limits
        max_memory_tokens = int(stats['max_context_tokens'] * 0.4)
        context = agent.memory_manager.get_context(max_tokens=max_memory_tokens)
        context_tokens = agent.memory_manager.count_tokens(context)
        print(f"\n✅ Context token limit: {max_memory_tokens}")
        print(f"   Actual context tokens: {context_tokens}")
        print(f"   Within limit: {context_tokens <= max_memory_tokens}")


def test_memory_types():
    """Test different memory backends."""
    print("\n" + "="*80)
    print("TEST 6: Different Memory Backends")
    print("="*80)

    tools = get_default_tools()

    # Test in-memory backend
    print("\n1. In-Memory Backend:")
    agent_mem = create_gemma_agent(
        tools=tools,
        verbose=False,
        enable_memory=True,
        memory_backend="in_memory",
        max_memory_size=50
    )
    agent_mem.run("Test in-memory backend", return_metrics=False)
    if agent_mem.memory_manager:
        print(f"   Memories: {agent_mem.memory_manager.backend.count()}")

    # Test vector store backend (stub)
    print("\n2. Vector Store Backend (stub):")
    agent_vec = create_gemma_agent(
        tools=tools,
        verbose=False,
        enable_memory=True,
        memory_backend="vector_store"
    )
    agent_vec.run("Test vector store backend", return_metrics=False)
    if agent_vec.memory_manager:
        print(f"   Memories: {agent_vec.memory_manager.backend.count()}")


def test_memory_disabled():
    """Test agent without memory."""
    print("\n" + "="*80)
    print("TEST 7: Agent Without Memory")
    print("="*80)

    tools = get_default_tools()
    agent = create_gemma_agent(
        tools=tools,
        verbose=False,
        enable_memory=False  # Memory disabled
    )

    print("\n>>> User: Calculate 10 + 5")
    response1 = agent.run("Calculate 10 + 5", return_metrics=False)
    print(f"<<< Agent: {response1}")

    print("\n>>> User: What was the previous calculation?")
    response2 = agent.run("What was the previous calculation?", return_metrics=False)
    print(f"<<< Agent: {response2}")

    print("\n(Agent should NOT remember the previous calculation)")
    print(f"Has memory manager: {agent.memory_manager is not None}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("REASONING AGENT - MEMORY TESTS")
    print("Testing memory management and context persistence")
    print("="*80)

    os.makedirs("logs", exist_ok=True)

    # Run tests
    try:
        test_basic_memory()
    except Exception as e:
        logger.error(f"Test 1 failed: {e}", exc_info=True)

    try:
        test_memory_with_metrics()
    except Exception as e:
        logger.error(f"Test 2 failed: {e}", exc_info=True)

    try:
        test_memory_search()
    except Exception as e:
        logger.error(f"Test 3 failed: {e}", exc_info=True)

    try:
        test_memory_export_import()
    except Exception as e:
        logger.error(f"Test 4 failed: {e}", exc_info=True)

    try:
        test_memory_token_management()
    except Exception as e:
        logger.error(f"Test 5 failed: {e}", exc_info=True)

    try:
        test_memory_types()
    except Exception as e:
        logger.error(f"Test 6 failed: {e}", exc_info=True)

    try:
        test_memory_disabled()
    except Exception as e:
        logger.error(f"Test 7 failed: {e}", exc_info=True)

    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
