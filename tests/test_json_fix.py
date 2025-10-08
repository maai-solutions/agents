#!/usr/bin/env python3
"""Test JSON parsing fix for gpt-oss:20b"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Install opentelemetry if needed
try:
    from linus.agents.agent.factory import create_gemma_agent
    from linus.agents.agent.tools import get_default_tools
except Exception as e:
    print(f"Import error: {e}")
    print("Installing dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "opentelemetry-api", "opentelemetry-sdk"])
    from linus.agents.agent.factory import create_gemma_agent
    from linus.agents.agent.tools import get_default_tools

def main():
    print("Creating agent with gpt-oss:20b...")

    agent = create_gemma_agent(
        api_base='http://localhost:11434/v1',
        model='gpt-oss:20b',
        api_key='not-needed',
        temperature=0.7,
        tools=get_default_tools(),
        verbose=True
    )

    print("\nTesting with simple query: 'What is 5 + 3?'")
    response = agent.run('What is 5 + 3?', return_metrics=True)

    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(f"Answer: {response.result}")
    print(f"\nMetrics: {response.metrics}")

    if response.execution_history:
        print("\nExecution History:")
        for item in response.execution_history:
            print(f"  - {item}")

if __name__ == "__main__":
    main()
