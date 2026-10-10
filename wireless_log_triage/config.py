import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("TRIAGE_DATA_DIR", "data"))
MODEL_NAME = os.environ.get("TRIAGE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
OLLAMA_MODEL = os.environ.get("TRIAGE_OLLAMA_MODEL", "qwen3:8b")
OLLAMA_URL = os.environ.get("TRIAGE_OLLAMA_URL", "http://localhost:11434")
