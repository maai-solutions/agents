"""Example usage of the ReasoningAgent with Gemma3:27b."""

import os
from src.linus.agents.agent.agent import ReasoningAgent, Agent
from src.linus.agents.agent.tools import get_default_tools, create_custom_tool
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from loguru import logger


def weather_tool(location: str) -> str:
    """Mock weather tool for demonstration."""
    return f"The weather in {location} is sunny with a temperature of 72°F"


class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    location: str = Field(description="Location to get weather for")


def main():
    """Demonstrate the ReasoningAgent with various tasks."""
    
    # Configure logging
    logger.add("agent_demo.log", rotation="10 MB")
    
    # Get default tools
    tools = get_default_tools()
    
    # Add a custom tool
    weather = create_custom_tool(
        name="weather",
        description="Get current weather for a location",
        func=weather_tool,
        args_schema=WeatherInput
    )
    tools.append(weather)
    
    # Create agent for Gemma3:27b
    # Assuming Ollama is running with Gemma3:27b model
    agent = Agent(
        api_base="http://localhost:11434/v1",  # Ollama OpenAI-compatible endpoint
        model="gemma3:27b",  # or "gemma2:27b" depending on your setup
        tools=tools,
        verbose=True
    )
    
    # Example tasks to demonstrate the agent
    tasks = [
        "What is the weather in New York?",
        "Calculate the result of 25 * 4 + 10",
        "Search for information about langchain agents",
        "Read the contents of the requirements.txt file",
        "First search for Python decorators, then calculate 15^2, and finally tell me the weather in London"
    ]
    
    print("=" * 60)
    print("ReasoningAgent Demo for Gemma3:27b")
    print("=" * 60)
    
    for task in tasks:
        print(f"\n📋 Task: {task}")
        print("-" * 40)
        
        try:
            result = agent.run(task)
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            logger.error(f"Failed to process task '{task}': {e}")
        
        print("-" * 40)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode (type 'exit' to quit)")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🤖 Enter your request: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            result = agent.run(user_input)
            print(f"\n💡 Response: {result}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            logger.error(f"Interactive mode error: {e}")


def test_reasoning_only():
    """Test just the reasoning phase without execution."""
    print("\n" + "=" * 60)
    print("Testing Reasoning Phase Only")
    print("=" * 60)
    
    # Create a simple agent
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        model="gemma3:27b",
        temperature=0.7,
        api_key="not-needed"
    )
    
    agent = ReasoningAgent(
        llm=llm,
        tools=get_default_tools(),
        verbose=True
    )
    
    test_requests = [
        "I need to analyze sales data from multiple CSV files and create a report",
        "Deploy a web application to AWS",
        "Debug a memory leak in a Python application"
    ]
    
    for request in test_requests:
        print(f"\n📝 Request: {request}")
        reasoning_result = agent._reasoning_call(request)
        
        print(f"✓ Has sufficient info: {reasoning_result.has_sufficient_info}")
        print(f"✓ Reasoning: {reasoning_result.reasoning}")
        print("✓ Planned tasks:")
        for i, task in enumerate(reasoning_result.tasks, 1):
            tool_info = f" (using {task.get('tool_name')})" if task.get('tool_name') else ""
            print(f"   {i}. {task['description']}{tool_info}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-reasoning":
        test_reasoning_only()
    else:
        main()