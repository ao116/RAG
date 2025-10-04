import os
import requests
import numpy as np

BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")

def llm_generate(prompt: str, model: str = "llama3.1:latest") -> str:
    """Call Ollama /api/generate (non-stream) and return the text."""
    url = f"{BASE}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]

def embedding(text: str | list[str], model: str = "nomic-embed-text:latest") -> np.ndarray:
    """Call Ollama /api/embed and return an array [n, d] of float32."""
    inputs = [text] if isinstance(text, str) else text
    url = f"{BASE}/api/embed"
    payload = {"model": model, "input": inputs}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    vecs = np.array(r.json()["embeddings"], dtype="float32")
    assert vecs.ndim == 2 and vecs.shape[0] == len(inputs)
    return vecs