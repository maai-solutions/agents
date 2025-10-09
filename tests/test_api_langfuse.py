"""Test the API with real Langfuse integration."""

import requests
import json

def test_api_with_langfuse():
    """Test the FastAPI endpoint with Langfuse telemetry."""
    
    # Test data
    test_request = {
        "query": "What is 2 + 2?",
        "use_tools": True,
        "stream": False,
        "session_id": "test-session-123"
    }
    
    try:
        print("🚀 Testing API endpoint with Langfuse telemetry...")
        print(f"Request: {test_request}")
        
        # Make request to the API
        response = requests.post(
            "http://localhost:8000/agent/query",
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            print(f"Response: {result.get('response', 'No response')}")
            print(f"Tools used: {result.get('tools_used', [])}")
            print(f"Execution time: {result.get('execution_time', 'Unknown')}s")
            return True
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server at http://localhost:8000")
        print("Make sure the server is running with: uvicorn src.app:app --reload")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("LANGFUSE TELEMETRY API TEST")
    print("=" * 60)
    
    success = test_api_with_langfuse()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("Check your Langfuse dashboard for traces with session 'test-session-123'")
    else:
        print("\n❌ Test failed!")
    
    print("=" * 60)