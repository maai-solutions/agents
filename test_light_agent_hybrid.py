"""Test script for LightAgent with hybrid tool calling support.

This script demonstrates:
1. Auto-detection of tool calling mode
2. Manual tool calling for Gemma3:27b
3. Native tool calling for GPT-4 (if API key available)
"""

import asyncio
import os
from openai import AsyncOpenAI

# Import LightAgent and tools
from src.linus.agents.agent.light_agent import LightAgent, ToolCallingMode
from src.linus.agents.agent.tools import get_default_tools


async def test_manual_mode_with_gemma():
    """Test manual mode with Gemma3:27b (Ollama)."""
    print("\n" + "=" * 80)
    print("TEST 1: Manual Mode with Gemma3:27b (Ollama)")
    print("=" * 80)

    # Create LightAgent for Ollama Gemma (will auto-detect MANUAL mode)
    llm = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed"
    )

    agent = LightAgent(
        llm=llm,
        model="gemma3:27b",
        tools=get_default_tools(),
        instructions="You are a helpful assistant that can use tools.",
        max_tool_iterations=5,
        temperature=0.7,
        tool_calling_mode=ToolCallingMode.AUTO,  # Will auto-detect MANUAL
        agent_name="gemma_agent"
    )

    print(f"✓ Agent created with mode: {agent.active_mode.value}")
    print(f"✓ Model: {agent.model}")

    # Test with a simple calculation
    query = "What is 42 multiplied by 17?"
    print(f"\n📝 Query: {query}")

    try:
        response = await agent.run(query, return_metrics=True)
        print(f"\n✅ Response: {response.result}")
        print(f"\n📊 Metrics:")
        print(f"   - Iterations: {response.metrics.total_iterations}")
        print(f"   - Tool executions: {response.metrics.tool_executions}")
        print(f"   - Execution time: {response.metrics.execution_time_seconds:.2f}s")
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def test_explicit_manual_mode():
    """Test explicitly setting manual mode."""
    print("\n" + "=" * 80)
    print("TEST 2: Explicit Manual Mode")
    print("=" * 80)

    llm = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed"
    )

    # Explicitly set MANUAL mode
    agent = LightAgent(
        llm=llm,
        model="gemma3:27b",
        tools=get_default_tools(),
        instructions="You are a calculator assistant.",
        tool_calling_mode="manual",  # Explicit string mode
        agent_name="manual_agent"
    )

    print(f"✓ Agent created with mode: {agent.active_mode.value}")

    query = "Calculate 100 + 250"
    print(f"\n📝 Query: {query}")

    try:
        response = await agent.run(query, return_metrics=True)
        print(f"\n✅ Response: {response.result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def test_native_mode_simulation():
    """Test native mode (will work with GPT-4 if API key is available)."""
    print("\n" + "=" * 80)
    print("TEST 3: Native Mode Detection")
    print("=" * 80)

    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("⚠️  OpenAI API key not found, skipping native mode test")
        print("   (This test requires OPENAI_API_KEY environment variable)")
        return

    llm = AsyncOpenAI(api_key=api_key)

    # Auto-detect should choose NATIVE for GPT-4
    agent = LightAgent(
        llm=llm,
        model="gpt-4",
        tools=get_default_tools(),
        instructions="You are a helpful assistant.",
        tool_calling_mode=ToolCallingMode.AUTO,
        agent_name="gpt4_agent"
    )

    print(f"✓ Agent created with mode: {agent.active_mode.value}")
    print(f"✓ Model: {agent.model}")

    query = "What is 15 times 8?"
    print(f"\n📝 Query: {query}")

    try:
        response = await agent.run(query, return_metrics=True)
        print(f"\n✅ Response: {response.result}")
        print(f"\n📊 Metrics:")
        print(f"   - Iterations: {response.metrics.total_iterations}")
        print(f"   - Tool executions: {response.metrics.tool_executions}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def test_mode_detection():
    """Test auto-detection for various model names."""
    print("\n" + "=" * 80)
    print("TEST 4: Auto-Detection for Various Models")
    print("=" * 80)

    test_models = [
        ("gpt-4", "should detect NATIVE"),
        ("gpt-3.5-turbo", "should detect NATIVE"),
        ("claude-3-opus", "should detect NATIVE"),
        ("gemma3:27b", "should detect MANUAL"),
        ("llama2:13b", "should detect MANUAL"),
        ("mistral-large", "should detect NATIVE"),
        ("phi-2", "should detect MANUAL"),
        ("unknown-model", "should default to MANUAL (safe fallback)"),
    ]

    llm = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed"
    )

    print("\nDetection Results:")
    print("-" * 80)

    for model_name, expected in test_models:
        agent = LightAgent(
            llm=llm,
            model=model_name,
            tools=[],  # No tools needed for detection test
            tool_calling_mode=ToolCallingMode.AUTO,
            agent_name="test_agent"
        )

        detected = agent.active_mode.value.upper()
        status = "✓" if detected in expected.upper() else "✗"
        print(f"{status} {model_name:20s} → {detected:8s} ({expected})")


async def main():
    """Run all tests."""
    print("\n🚀 LightAgent Hybrid Tool Calling Tests")
    print("=" * 80)

    # Test 1: Manual mode with Gemma
    await test_manual_mode_with_gemma()

    # Test 2: Explicit manual mode
    await test_explicit_manual_mode()

    # Test 3: Native mode (if API key available)
    await test_native_mode_simulation()

    # Test 4: Mode detection
    await test_mode_detection()

    print("\n" + "=" * 80)
    print("✨ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
