import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import threading

# Load environment variables
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
MODEL_FILE = os.getenv("MODEL_FILE", "model.gguf")
MODEL_PATH = str(Path(MODEL_DIR) / MODEL_FILE)

# Default generation settings
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

app = FastAPI(title="Railway Mistral Q4_0 Inference")

# Lazy model loading
_llm = None
_llm_lock = threading.Lock()

class Prompt(BaseModel):
    prompt: str
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE

def load_model():
    """Load the LLaMA model if it's not already loaded."""
    global _llm
    from llama_cpp import Llama
    with _llm_lock:
        if _llm is None:
            if not os.path.isfile(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            print(f"Loading model from {MODEL_PATH}...")
            _llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)
            print("Model loaded successfully.")
    return _llm

@app.on_event("startup")
def check_model():
    """Check if model exists, optionally preload it in background."""
    if not os.path.isfile(MODEL_PATH):
        print(f"WARNING: model file not found at {MODEL_PATH}. Waiting until available.")
    else:
        def _bg_load():
            try:
                load_model()
            except Exception as e:
                print("Model preload failed:", e)
        threading.Thread(target=_bg_load, daemon=True).start()

@app.post("/generate")
def generate(p: Prompt):
    """Generate text from the given prompt."""
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model file not available yet.")
    
    llm = load_model()
    try:
        out = llm(
            prompt=p.prompt,
            max_tokens=p.max_tokens,
            temperature=p.temperature,
            stop=["</s>"],  # Optional stop sequence
            echo=False
        )
        text = out["choices"][0]["text"]
        return {"text": text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_exists": os.path.isfile(MODEL_PATH)}
