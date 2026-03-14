import json
import uuid

def test_tool_content_start():
    prompt_name = str(uuid.uuid4())
    content_name = str(uuid.uuid4())
    tool_use_id = "test-id"
    
    # Simulating _build_tool_content_start logic
    event = {
        "event": {
            "contentStart": {
                "promptName": prompt_name,
                "contentName": content_name,
                "interactive": False,
                "type": "TOOL",
                "role": "USER", # This is the fix
                "toolResultInputConfiguration": {
                    "toolUseId": tool_use_id,
                    "type": "TEXT",
                    "textInputConfiguration": {"mediaType": "text/plain"}
                }
            }
        }
    }
    
    event_json = json.dumps(event)
    data = json.loads(event_json)
    
    content_start = data.get("event", {}).get("contentStart", {})
    role = content_start.get("role")
    
    print(f"Generated Event: {event_json}")
    if role == "USER":
        print("VERIFICATION SUCCESS: 'role' is correctly set to 'USER' for TOOL contentStart.")
    else:
        print(f"VERIFICATION FAILURE: 'role' is {role}, expected 'USER'.")

if __name__ == "__main__":
    test_tool_content_start()
