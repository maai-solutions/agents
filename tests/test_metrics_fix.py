"""Simple test to verify the attribute error is fixed."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from linus.agents.agent.models import AgentMetrics


def test_agent_metrics():
    """Test AgentMetrics attributes."""
    print("Testing AgentMetrics attributes...")
    
    metrics = AgentMetrics()
    print(f"total_iterations: {metrics.total_iterations}")
    print(f"Has 'iterations' attribute: {hasattr(metrics, 'iterations')}")
    print(f"Has 'total_iterations' attribute: {hasattr(metrics, 'total_iterations')}")
    
    # Test setting total_iterations
    metrics.total_iterations = 5
    print(f"After setting: total_iterations = {metrics.total_iterations}")
    
    print("✅ AgentMetrics test passed!")


if __name__ == "__main__":
    test_agent_metrics()