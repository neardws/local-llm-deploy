#!/usr/bin/env python3
"""Test vLLM OpenAI-compatible API"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_models():
    """List available models"""
    resp = requests.get(f"{BASE_URL}/v1/models")
    print("Available models:")
    print(json.dumps(resp.json(), indent=2))
    return resp.json()

def test_completion(model_id: str, prompt: str = "Hello, how are you?"):
    """Test chat completion"""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    result = resp.json()
    print("\nChat Completion Response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

def test_streaming(model_id: str, prompt: str = "Write a short poem about AI"):
    """Test streaming completion"""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": True
    }
    
    print("\nStreaming Response:")
    with requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True
    ) as resp:
        for line in resp.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data != '[DONE]':
                        chunk = json.loads(data)
                        if chunk['choices'][0]['delta'].get('content'):
                            print(chunk['choices'][0]['delta']['content'], end='', flush=True)
    print()

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, how are you?"
    
    try:
        models = test_models()
        if models.get('data'):
            model_id = models['data'][0]['id']
            test_completion(model_id, prompt)
            test_streaming(model_id, "Write a haiku about coding")
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to vLLM server. Make sure it's running on port 8000")
        sys.exit(1)
