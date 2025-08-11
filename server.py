import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import threading
import time
import signal

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

@app.post("/test-generate")
def test_generate():
    """Simple test endpoint to check if model loading works."""
    print("Test generate endpoint called")
    
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model file not available yet.")
    
    try:
        print("Attempting to load model...")
        llm = load_model()
        print("Model loaded successfully!")
        return {"status": "success", "message": "Model loaded and ready for inference"}
    except Exception as e:
        print(f"Model loading error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")

@app.post("/generate")
def generate(p: Prompt):
    """Generate text from the given prompt."""
    print(f"Received generate request: {p.prompt[:50]}...")
    
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model file not available yet.")
    
    try:
        print("Loading model...")
        llm = load_model()
        print("Model loaded, starting generation...")
        
        out = llm(
            prompt=p.prompt,
            max_tokens=p.max_tokens,
            temperature=p.temperature,
            stop=["</s>"],  # Optional stop sequence
            echo=False
        )
        text = out["choices"][0]["text"]
        print(f"Generation complete: {text[:50]}...")
        return {"text": text.strip()}
    except Exception as e:
        print(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_exists": os.path.isfile(MODEL_PATH)}

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Mistral Q4_0 Inference API", 
        "endpoints": {
            "/generate": "POST - Generate text from prompt",
            "/health": "GET - Health check",
            "/status": "GET - Detailed status"
        }
    }

@app.get("/status")
def status():
    """Detailed status endpoint."""
    status_info = {
        "model_path": MODEL_PATH,
        "model_exists": os.path.isfile(MODEL_PATH),
        "model_loaded": _llm is not None,
        "model_size_mb": round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 2) if os.path.isfile(MODEL_PATH) else None,
        "environment": {
            "MODEL_DIR": MODEL_DIR,
            "MODEL_FILE": MODEL_FILE,
            "MAX_TOKENS": MAX_TOKENS,
            "TEMPERATURE": TEMPERATURE
        }
    }
    
    try:
        import psutil
        status_info["memory_usage"] = {
            "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
            "total_mb": round(psutil.virtual_memory().total / 1024 / 1024, 2),
            "percent": psutil.virtual_memory().percent
        }
    except ImportError:
        status_info["memory_usage"] = "psutil not available"
    
    return status_info
