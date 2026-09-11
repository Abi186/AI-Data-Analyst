"""
ollama_client.py
-----------------
Thin wrapper around the local Ollama REST API.
No API key. No internet. ₹0 cost. Runs Qwen (or any pulled model) locally.

Ollama exposes a local server at http://localhost:11434 once installed
and running. This module just sends prompts to it and returns text.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"   # change to qwen2.5:7b / qwen2:1.5b etc. based on your RAM


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    """
    Sends a prompt to the local Ollama server and returns the generated text.
    Raises a clear RuntimeError if Ollama isn't running / model isn't pulled.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "❌ Could not reach Ollama at http://localhost:11434.\n"
            "Make sure Ollama is installed and running:\n"
            "  1) Install: https://ollama.com/download\n"
            "  2) Run:     ollama serve   (if not already running)\n"
            "  3) Pull:    ollama pull qwen2.5:3b"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("⏳ Ollama took too long to respond. Try a smaller model (e.g. qwen2:1.5b).")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def check_ollama_alive() -> bool:
    """Quick health check used by the Streamlit app to show a status badge."""
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
