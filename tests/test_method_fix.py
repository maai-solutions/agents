#!/usr/bin/env python3
"""
Quick test to verify the _format_final_response_with_history method is available.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.agent.reasoning_agent import ReasoningAgent

def test_method_exists():
    """Test that the missing method now exists."""
    
    print("Testing ReasoningAgent method availability...")
    
    # Check if method exists
    if hasattr(ReasoningAgent, '_format_final_response_with_history'):
        print("✅ _format_final_response_with_history method exists")
        
        # Check method signature
        import inspect
        sig = inspect.signature(ReasoningAgent._format_final_response_with_history)
        print(f"✅ Method signature: {sig}")
        
        # Test basic instantiation (without LLM for now)
        try:
            agent = ReasoningAgent(llm=None)
            print("✅ ReasoningAgent can be instantiated")
        except Exception as e:
            print(f"⚠️  ReasoningAgent instantiation issue: {e}")
            
    else:
        print("❌ _format_final_response_with_history method is missing")
        return False
    
    return True

if __name__ == "__main__":
    success = test_method_exists()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)