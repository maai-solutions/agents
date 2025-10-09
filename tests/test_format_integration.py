#!/usr/bin/env python3
"""
Test that the _format_final_response_with_history method works correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.agent.reasoning_agent import ReasoningAgent
from linus.agents.agent.tool_base import BaseTool

def test_format_method():
    """Test the _format_final_response_with_history method."""
    
    # Create a mock LLM class
    class MockLLM:
        class Chat:
            class Completions:
                def create(self, **kwargs):
                    class MockResponse:
                        class Choice:
                            class Message:
                                content = "Based on my research, whales face several dangers including pollution, climate change, and fishing nets."
                        choices = [Choice()]
                    return MockResponse()
            completions = Completions()
        chat = Chat()
    
    # Create a simple mock tool class
    class MockTool(BaseTool):
        def __init__(self):
            self.name = "test_tool"
            self.description = "Test tool"
            super().__init__()
        
        async def _arun(self, *args, **kwargs):
            return "Mock result"
    
    # Create agent with minimal setup
    mock_llm = MockLLM()
    agent = ReasoningAgent(
        llm=mock_llm,
        model="test-model",
        tools=[MockTool()]
    )
    
    # Test data
    input_text = "why are the whales in danger?"
    execution_history = [
        {
            "iteration": 1,
            "task": "Search for information about whale dangers",
            "tool": "vector-store_vector_search",
            "result": "Whales face threats from pollution, climate change, and fishing activities.",
            "status": "completed"
        }
    ]
    completion_status = {
        "is_complete": True,
        "reasoning": "Successfully found information about whale dangers"
    }
    
    try:
        # Test the method
        result = agent._format_final_response_with_history(input_text, execution_history, completion_status)
        print(f"✅ Method executed successfully")
        print(f"✅ Result: {result[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Method failed: {e}")
        return False

if __name__ == "__main__":
    success = test_format_method()
    print(f"\nIntegration test result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)