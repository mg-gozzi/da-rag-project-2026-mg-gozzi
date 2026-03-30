#!/usr/bin/env python3
"""Test /answer endpoint."""
import json
import urllib.request

try:
    # Check /answer endpoint directly
    print("Testing /answer endpoint...")
    req = urllib.request.Request(
        'http://127.0.0.1:9000/answer',
        data=json.dumps({'question': 'What is France?', 'k': 3}).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.load(resp)
            print(f"✓ /answer endpoint WORKS!")
            print(f"Status: 200")
            print(f"Answer: {result.get('answer', 'N/A')[:100]}")
    except urllib.error.HTTPError as e:
        print(f"✗ /answer endpoint returned {e.code}")
        print(f"Detail: {e.read().decode()}")
    
    # Check registered paths
    print("\n\nRegistered paths:")
    with urllib.request.urlopen('http://127.0.0.1:9000/openapi.json', timeout=5) as resp:
        openapi = json.load(resp)
        paths = sorted(openapi.get('paths', {}).keys())
        for path in paths:
            print(f"  {path}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
