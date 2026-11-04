import os
import sys
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configured to use Ollama Cloud model gemma4:31b-cloud via local Ollama daemon
model_name = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

model = ChatOllama(model=model_name, base_url=base_url)
