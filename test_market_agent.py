import requests
import json
import time

url = "http://localhost:8000/market/start"
payload = {"goal": "AI meeting summarizer tool for dentists"}
headers = {"Content-Type": "application/json"}

print(f"Triggering {url} with {payload}")
res = requests.post(url, json=payload, headers=headers)
print("Response:", res.status_code, res.text)

if res.ok:
    data = res.json()
    session_id = data["id"]
    print(f"Session queued. Monitoring stream for session {session_id}...")
    
    # We could simulate SSE streaming by doing a GET, but here we just poll the session
    for _ in range(30):
        time.sleep(3)
        chk = requests.get(f"http://localhost:8000/market/{session_id}")
        if chk.ok:
            info = chk.json()
            print(f"Status: {info['status']}, Vendors found: {len(info['results'])}")
            if info['status'] == 'done' or info['status'] == 'failed':
                print(f"Finished with status {info['status']}. Logs: {len(info['logs'])}")
                for log in info['logs']:
                    print(f"[{log['step']}] {log['message']}")
                for i, r in enumerate(info["results"]):
                    print(f"URL: {r['url']}")
                    print(f"Data: {json.dumps(r.get('data', {}), indent=2)}")
                    print("-" * 40)
                break
