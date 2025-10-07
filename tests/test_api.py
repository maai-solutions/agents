#!/usr/bin/env python3
"""Test script for the ReasoningAgent API."""

import requests
import json
import time
from typing import Dict, Any


API_BASE = "http://localhost:8000"


def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.json()


def test_list_tools():
    """Test listing available tools."""
    print("Listing available tools...")
    response = requests.get(f"{API_BASE}/tools")
    if response.status_code == 200:
        tools = response.json()["tools"]
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
    print()
    return response.json()


def test_query(query: str, use_tools: bool = True) -> Dict[str, Any]:
    """Test a single query."""
    print(f"Query: {query}")
    print("-" * 40)
    
    payload = {
        "query": query,
        "use_tools": use_tools,
        "stream": False
    }
    
    response = requests.post(f"{API_BASE}/agent/query", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['response']}")
        if data.get('tools_used'):
            print(f"Tools used: {', '.join(data['tools_used'])}")
        print(f"Execution time: {data['execution_time']:.2f}s")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print()
    return response.json() if response.status_code == 200 else None


def test_reasoning(query: str):
    """Test the reasoning phase only."""
    print(f"Testing reasoning for: {query}")
    print("-" * 40)
    
    payload = {"query": query}
    response = requests.post(f"{API_BASE}/agent/reasoning", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Has sufficient info: {data['has_sufficient_info']}")
        print(f"Reasoning: {data['reasoning']}")
        print("Planned tasks:")
        for i, task in enumerate(data['planned_tasks'], 1):
            tool = f" (tool: {task.get('tool_name')})" if task.get('tool_name') else ""
            print(f"  {i}. {task['description']}{tool}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print()
    return response.json() if response.status_code == 200 else None


def test_tool(tool_name: str, tool_args: Dict[str, Any]):
    """Test a specific tool."""
    print(f"Testing tool: {tool_name}")
    print(f"Arguments: {tool_args}")
    print("-" * 40)
    
    payload = {
        "tool_name": tool_name,
        "tool_args": tool_args
    }
    
    response = requests.post(f"{API_BASE}/tools/test", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Result: {data['result']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print()
    return response.json() if response.status_code == 200 else None


def test_batch_queries(queries: list):
    """Test batch query processing."""
    print("Testing batch queries...")
    print("-" * 40)
    
    response = requests.post(f"{API_BASE}/agent/batch", json=queries)
    
    if response.status_code == 200:
        data = response.json()
        for result in data['results']:
            print(f"Query: {result['query']}")
            if result['status'] == 'success':
                print(f"Response: {result['response'][:100]}...")
            else:
                print(f"Error: {result['error']}")
            print()
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    return response.json() if response.status_code == 200 else None


def test_scenarios():
    """Run predefined test scenarios."""
    print("Running test scenarios...")
    response = requests.get(f"{API_BASE}/test/scenarios")
    
    if response.status_code == 200:
        data = response.json()
        for result in data['test_results']:
            print(f"Scenario: {result.get('scenario')}")
            if 'result' in result:
                print(f"Result: {result['result']}")
            else:
                print(f"Error: {result.get('error')}")
            print()
    
    return response.json() if response.status_code == 200 else None


def main():
    """Run all tests."""
    print("=" * 60)
    print("ReasoningAgent API Test Suite")
    print("=" * 60)
    print()
    
    # Check if API is running
    try:
        requests.get(f"{API_BASE}/")
    except requests.ConnectionError:
        print("Error: API is not running. Start it with:")
        print("  cd /Users/udg/Projects/ai/agents")
        print("  python src/app.py")
        return
    
    # Run tests
    print("1. Health Check")
    print("=" * 60)
    test_health()
    
    print("2. Available Tools")
    print("=" * 60)
    test_list_tools()
    
    print("3. Simple Queries")
    print("=" * 60)
    test_query("What is 25 * 4?")
    test_query("What is the current time?")
    
    print("4. Reasoning Tests")
    print("=" * 60)
    test_reasoning("I need to analyze sales data and create a report")
    test_reasoning("Calculate the fibonacci sequence and save it to a file")
    
    print("5. Tool Tests")
    print("=" * 60)
    test_tool("calculator", {"expression": "2**10"})
    test_tool("search", {"query": "Python decorators", "limit": 3})
    
    print("6. Complex Multi-Step Query")
    print("=" * 60)
    test_query("First get the current time, then calculate 15^2, and finally search for information about langchain")
    
    print("7. Batch Processing")
    print("=" * 60)
    test_batch_queries([
        "What is 100 / 4?",
        "Search for FastAPI",
        "Get the current time"
    ])
    
    print("8. Predefined Scenarios")
    print("=" * 60)
    test_scenarios()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()