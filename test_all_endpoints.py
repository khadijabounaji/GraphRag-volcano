#!/usr/bin/env python3
import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Wait for server to start
time.sleep(3)

tests = [
    "Qu'est-ce que la méthode ZDD géochronologique ?",
    "Quels sont les minéraux composant le basalte ?",
    "Comment fonctionne le système hydrothermal du Piton des Neiges ?",
    "Quelle est la caldera du Piton des Neiges ?",
    "C'est quoi?",  # Should be off-topic
    "Bonjour",  # Should be off-topic
]

for question in tests:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    try:
        resp = requests.post(
            f"{BASE_URL}/query",
            json={"question": question},
            timeout=30
        )
        
        if resp.status_code != 200:
            print(f"ERROR: Status {resp.status_code}")
            print(resp.text)
            continue
        
        data = resp.json()
        
        print(f"Analysis Type: {data.get('analysis_type')}")
        print(f"Routing: {data.get('routing')}")
        print(f"Confidence: {data.get('confidence')}")
        print(f"State: {data.get('state')}")
        print(f"Action: {data.get('action')}")
        print(f"Reward: {data.get('reward')}")
        print(f"\nAnswer:\n{data.get('answer', '')[:400]}...")
        
    except Exception as e:
        print(f"ERROR: {e}")

print("\n\nTesting /vectorial endpoint...")
resp = requests.get(
    f"{BASE_URL}/vectorial",
    params={
        "question": "Qu'est-ce que la méthode ZDD géochronologique ?",
        "k": 3
    },
    timeout=30
)

if resp.status_code == 200:
    data = resp.json()
    print(f"Status: {data.get('status')}")
    print(f"Total chunks: {data.get('stats', {}).get('total_chunks')}")
    print(f"Query results: {len(data.get('query_results', []))} results")
else:
    print(f"ERROR: Status {resp.status_code}")
    print(resp.text)

print("\n\nTesting /graph endpoint...")
resp = requests.get(
    f"{BASE_URL}/graph?question=Qu'est-ce%20que%20la%20méthode%20ZDD",
    timeout=30
)

if resp.status_code == 200:
    data = resp.json()
    print(f"Status: {data.get('status')}")
    print(f"Nodes: {data.get('stats', {}).get('nodes')}")
    print(f"Relations: {data.get('stats', {}).get('relations')}")
    print(f"Query results: {len(data.get('query_results', []))} results")
else:
    print(f"ERROR: Status {resp.status_code}")
    print(resp.text)

print("\n\nTesting /agentic endpoint...")
resp = requests.get(f"{BASE_URL}/agentic", timeout=30)

if resp.status_code == 200:
    data = resp.json()
    print(f"Status: {data.get('status')}")
    print(f"Q-Learning states: {len(data.get('q_learning', {}).get('states', []))}")
    print(f"Total episodes: {data.get('reward_monitor', {}).get('total_episodes')}")
else:
    print(f"ERROR: Status {resp.status_code}")
    print(resp.text)

print("\n\nAll tests completed!")
