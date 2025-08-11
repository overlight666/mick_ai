import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import threading

load_dotenv()

MODEL_DIR = os.getenv('MODEL_DIR', '/app/model')
MODEL_FILE = os.getenv('MODEL_FILE', 'model.gguf')
MODEL_PATH = str(Path(MODEL_DIR) / MODEL_FILE)
# generation defaults
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '256'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))

app = FastAPI(title='Railway Mistral Q4_0 Inference')

# Lazy model loader to avoid startup failure if model is missing
_llm = None
_llm_lock = threading.Lock()

class Prompt(BaseModel):
    prompt: str
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE

def load_model():
    global _llm
    from llama_cpp import Llama
    if _llm is None:
        print(f"Loading model from {MODEL_PATH}...")
        _llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)
        print("Model loaded.")
    return _llm

@app.on_event('startup')
def check_model():
    # do not eagerly fail; just log if missing - start.sh should ensure it's downloaded
    if not os.path.isfile(MODEL_PATH):
        print(f"WARNING: model file not found at {MODEL_PATH}. The server will wait until file is present.")
    else:
        # pre-load in background to warm up (optional)
        def _bg_load():
            try:
                load_model()
            except Exception as e:
                print('Model load failed in background:', e)
        t = threading.Thread(target=_bg_load, daemon=True)
        t.start()

@app.post('/generate')
def generate(p: Prompt):
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(status_code=503, detail='Model file not available yet.')
    llm = load_model()
    try:
        out = llm.create(prompt=p.prompt, max_tokens=p.max_tokens, temperature=p.temperature)
        # llama-cpp-python returns dict with 'choices' list for create()
        text = out['choices'][0]['text']
        return { 'text': text }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
def health():
    return { 'status': 'ok', 'model_exists': os.path.isfile(MODEL_PATH) }
