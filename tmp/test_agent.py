import requests
import uuid

def test_agent():
    session_id = str(uuid.uuid4())
    print(f"Testing with session_id: {session_id}")
    
    # Test chat endpoint
    url = "http://localhost:8000/agent/chat"
    payload = {
        "session_id": session_id,
        "user_query": "Hello, who are you?"
    }
    
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test history endpoint
    print("\nFetching history...")
    url_history = f"http://localhost:8000/conversations/{session_id}"
    response_history = requests.get(url_history)
    print(f"Status Code (History): {response_history.status_code}")
    print(f"Response (History): {response_history.json()}")

if __name__ == "__main__":
    test_agent()
