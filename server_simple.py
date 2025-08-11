import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import asyncio

# Load environment variables
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
MODEL_FILE = os.getenv("MODEL_FILE", "model.gguf")
MODEL_PATH = str(Path(MODEL_DIR) / MODEL_FILE)

# Default generation settings
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))  # Reduced default
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

app = FastAPI(title="Mistral Q4_0 Inference API")

# Global model variable
_llm = None

class Prompt(BaseModel):
    prompt: str
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global _llm
    print("Starting up...")
    
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
    
    try:
        print(f"Loading model from {MODEL_PATH}...")
        from llama_cpp import Llama
        
        # Use minimal settings for memory efficiency
        _llm = Llama(
            model_path=MODEL_PATH, 
            n_ctx=1024,  # Reduced context
            n_threads=2,  # Reduced threads
            verbose=False
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Mistral Q4_0 Inference API",
        "status": "Model loaded" if _llm else "Model not loaded",
        "endpoints": {
            "/generate": "POST - Generate text",
            "/health": "GET - Health check",
            "/quick-test": "GET - Quick model test"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_exists": os.path.isfile(MODEL_PATH),
        "model_loaded": _llm is not None,
        "model_size_mb": round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 2) if os.path.isfile(MODEL_PATH) else None
    }

@app.get("/quick-test")
def quick_test():
    """Quick model test with minimal generation."""
    if not _llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = _llm("Hello", max_tokens=5, temperature=0.1, echo=False)
        return {"status": "success", "test_output": result["choices"][0]["text"].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@app.post("/generate")
async def generate(p: Prompt):
    """Generate text from the given prompt."""
    if not _llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Limit max_tokens to prevent memory issues
        max_tokens = min(p.max_tokens, 256)
        
        print(f"Generating for prompt: '{p.prompt[:50]}...' with max_tokens={max_tokens}")
        
        result = _llm(
            prompt=p.prompt,
            max_tokens=max_tokens,
            temperature=p.temperature,
            stop=["</s>", "\n\n"],
            echo=False
        )
        
        text = result["choices"][0]["text"].strip()
        print(f"Generated: '{text[:50]}...'")
        
        return {"text": text}
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
