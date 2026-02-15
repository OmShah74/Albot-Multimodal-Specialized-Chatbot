import requests
import time
import uuid
import json

BASE_URL = "http://localhost:8010"

def test_memory_pipeline():
    chat_id = f"test-memory-{uuid.uuid4().hex[:8]}"
    print(f"🚀 Starting memory verification for chat_id: {chat_id}")

    # 1. Send a query that should trigger memory
    query_text = "What is the capital of France and what are its main architectural landmarks?"
    print(f"\nStep 1: Sending query: '{query_text}'")
    
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "query": query_text,
            "chat_id": chat_id,
            "search_mode": "web_search"
        }
    )
    
    if response.status_code != 200:
        print(f"[FAIL] Query failed with status {response.status_code}: {response.text}")
        return

    print(f"[OK] Query successful (took {time.time()-start:.2f}s)")
    
    # 2. Wait for non-blocking fragment extraction
    print("\nStep 2: Waiting 3s for asynchronous memory processing...")
    time.sleep(3)

    # 3. Verify reasoning traces
    print("\nStep 3: Checking reasoning traces...")
    trace_resp = requests.get(f"{BASE_URL}/memory/{chat_id}/traces")
    if trace_resp.status_code == 200:
        traces = trace_resp.json().get("traces", [])
        if traces:
            print(f"[OK] Found {len(traces)} reasoning traces")
            print(f"   Latest trace summary: {traces[-1].get('answer_summary', '')[:100]}...")
        else:
            print("[WARN] No reasoning traces found")
    else:
        print(f"[FAIL] Failed to get traces: {trace_resp.text}")

    # 4. Verify memory fragments
    print("\nStep 4: Checking memory fragments...")
    frag_resp = requests.get(f"{BASE_URL}/memory/{chat_id}/fragments")
    if frag_resp.status_code == 200:
        fragments = frag_resp.json().get("fragments", [])
        if fragments:
            print(f"[OK] Found {len(fragments)} memory fragments")
            for i, f in enumerate(fragments):
                print(f"   [{i+1}] {f.get('fragment_type')}: {f.get('content')[:100]}...")
        else:
            print("[WARN] No memory fragments extracted (too short or trivial answer?)")
    else:
        print(f"[FAIL] Failed to get fragments: {frag_resp.text}")

    # 5. Verify web history
    print("\nStep 5: Checking web history...")
    web_resp = requests.get(f"{BASE_URL}/memory/{chat_id}/web-history")
    if web_resp.status_code == 200:
        web_logs = web_resp.json().get("web_interactions", [])
        if web_logs:
            print(f"[OK] Found {len(web_logs)} web interaction logs")
        else:
            print("[INFO] No web results logged (local context might have been sufficient)")

    # 6. Test namespace listing
    print("\nStep 6: Checking available namespaces...")
    ns_resp = requests.get(f"{BASE_URL}/memory/namespaces?session_id={chat_id}")
    if ns_resp.status_code == 200:
        namespaces = ns_resp.json().get("namespaces", {})
        print(f"[OK] Available namespaces: {list(namespaces.keys())}")
        print(f"   Fragment counts: {namespaces}")
    else:
        print(f"[FAIL] Failed to get namespaces: {ns_resp.text}")

    print("\nVerification complete!")

if __name__ == "__main__":
    try:
        test_memory_pipeline()
    except Exception as e:
        print(f"❌ An error occurred during verification: {e}")
        print("   Is the backend running at http://localhost:8010?")
