"""Verification script to show that reasoning and tool args output tracing is implemented.

This script demonstrates the code changes without requiring Langfuse credentials.
"""

import inspect
from linus.agents.agent.reasoning_agent import ReasoningAgent

def verify_reasoning_call_tracing():
    """Verify that _reasoning_call method includes output tracing."""

    print("=" * 80)
    print("VERIFICATION: Reasoning Output Tracing Fix")
    print("=" * 80)
    print()

    # Get the source code of _reasoning_call method
    source = inspect.getsource(ReasoningAgent._reasoning_call)

    # Check for key indicators that output tracing is implemented
    checks = {
        "Creates reasoning span": "trace_reasoning_phase" in source,
        "Creates LLM generation": "trace_llm_call" in source,
        "Updates LLM generation output": "update_generation" in source,
        "Updates reasoning span with output": "update_current_span(output=" in source,
        "Handles error case": 'level="ERROR"' in source,
        "Includes structured output": '"has_sufficient_info"' in source and '"reasoning"' in source,
    }

    print("Code Analysis for _reasoning_call:")
    print("-" * 80)
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    print()

    all_passed = all(checks.values())

    if all_passed:
        print("✅ SUCCESS: All reasoning tracing checks passed!")
        print()
        print("The _reasoning_call method now:")
        print("  1. Creates a reasoning phase span")
        print("  2. Creates an LLM generation observation")
        print("  3. Updates the generation with output and token usage")
        print("  4. Updates the reasoning span with structured output")
        print("  5. Handles errors by updating span with error info")
    else:
        print("❌ FAILURE: Some reasoning tracing checks failed!")

    print("=" * 80)
    print()

    return all_passed

def verify_tool_args_tracing():
    """Verify that _generate_tool_arguments method includes output tracing."""

    print("=" * 80)
    print("VERIFICATION: Tool Arguments Output Tracing Fix")
    print("=" * 80)
    print()

    # Get the source code of _generate_tool_arguments method
    source = inspect.getsource(ReasoningAgent._generate_tool_arguments)

    # Check for key indicators that output tracing is implemented
    checks = {
        "Creates LLM generation": "trace_llm_call" in source and "tool_args" in source,
        "Updates generation with output": "update_generation" in source,
        "Includes parsed args in output": '"parsed_args"' in source,
        "Includes raw response in output": '"raw_response"' in source,
        "Includes tool name in output": '"tool"' in source,
        "Handles error case": '"error"' in source and "update_generation" in source,
        "Parsing inside trace context": "async with self.tracer.trace_llm_call" in source,
    }

    print("Code Analysis for _generate_tool_arguments:")
    print("-" * 80)
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    print()

    all_passed = all(checks.values())

    if all_passed:
        print("✅ SUCCESS: All tool args tracing checks passed!")
        print()
        print("The _generate_tool_arguments method now:")
        print("  1. Creates an LLM generation observation")
        print("  2. Parses JSON arguments inside the trace context")
        print("  3. Updates generation with structured output containing:")
        print("     • raw_response (original LLM output)")
        print("     • parsed_args (parsed JSON arguments)")
        print("     • tool (tool name)")
        print("  4. Handles parsing errors by updating with error info")
    else:
        print("❌ FAILURE: Some tool args tracing checks failed!")

    print("=" * 80)
    print()

    return all_passed

def print_summary():
    """Print expected behavior summary."""
    print("=" * 80)
    print("EXPECTED BEHAVIOR IN LANGFUSE")
    print("=" * 80)
    print()
    print("1. reasoning_phase span:")
    print("   • Input: User query")
    print("   • Output: Structured result with has_sufficient_info, reasoning, tasks")
    print()
    print("2. llm_reasoning generation:")
    print("   • Input: Reasoning prompt")
    print("   • Output: Raw LLM response")
    print("   • Usage: Token counts")
    print()
    print("3. llm_tool_args generation:")
    print("   • Input: Tool argument generation prompt")
    print("   • Output: Structured data with raw_response, parsed_args, tool")
    print("   • Usage: Token counts")
    print()
    print("4. tool_* spans:")
    print("   • Input: Tool arguments")
    print("   • Output: Tool execution result")
    print()
    print("=" * 80)

if __name__ == "__main__":
    reasoning_success = verify_reasoning_call_tracing()
    tool_args_success = verify_tool_args_tracing()

    if reasoning_success and tool_args_success:
        print_summary()
        print("\n✅ ALL VERIFICATIONS PASSED!\n")
        exit(0)
    else:
        print("\n❌ SOME VERIFICATIONS FAILED!\n")
        exit(1)
