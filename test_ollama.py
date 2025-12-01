import requests
import sys

def test_ollama():
    print("=== Testing Ollama Connection ===\n")
    
    base_url = "http://127.0.0.1:11434"
    
    # Test 1: Check if Ollama is running
    print("1. Checking if Ollama is running...")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ Ollama is running on {base_url}")
            models = response.json().get("models", [])
            print(f"   ✓ Found {len(models)} models:")
            for model in models:
                print(f"     - {model.get('name')}")
        else:
            print(f"   ✗ Unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Cannot connect to Ollama: {e}")
        print("   Make sure 'ollama serve' is running!")
        return False
    
    # Test 2: Test embedding
    print("\n2. Testing embedding with 'nomic-embed-text'...")
    try:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "test embedding"},
            timeout=30
        )
        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            print(f"   ✓ Embedding successful (length: {len(embedding)})")
        else:
            print(f"   ✗ Embedding failed: {response.status_code}")
            print(f"     Response: {response.text[:200]}")
            
            # Try to pull the model
            print("   Trying to pull nomic-embed-text model...")
            import subprocess
            result = subprocess.run(["ollama", "pull", "nomic-embed-text"], capture_output=True, text=True)
            print(f"     Pull result: {result.returncode}")
            
            return False
    except Exception as e:
        print(f"   ✗ Embedding error: {e}")
        return False
    
    # Test 3: Test generation
    print("\n3. Testing generation with 'qwen2.5:0.5b-instruct'...")
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": "qwen2.5:0.5b-instruct", "prompt": "Say hello in one word.", "stream": False},
            timeout=30
        )
        if response.status_code == 200:
            answer = response.json().get("response", "")
            print(f"   ✓ Generation successful: {answer}")
        else:
            print(f"   ✗ Generation failed: {response.status_code}")
            print(f"     Trying to pull qwen2.5:0.5b-instruct model...")
            import subprocess
            result = subprocess.run(["ollama", "pull", "qwen2.5:0.5b-instruct"], capture_output=True, text=True)
            print(f"     Pull result: {result.returncode}")
    except Exception as e:
        print(f"   ✗ Generation error: {e}")
    
    print("\n=== Tests Complete ===")
    return True

if __name__ == "__main__":
    if test_ollama():
        print("\n✓ All tests passed! You can run your main code.")
    else:
        print("\n✗ Tests failed. Please fix issues before running main code.")
        sys.exit(1)