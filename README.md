# Railway Mistral Q4_0 Deployment Template (GGUF auto-download)

## Overview
This template downloads a GGUF-quantized model at container startup from a public URL (Hugging Face or other HTTP host), and serves a small FastAPI inference endpoint using `llama-cpp-python` (llama.cpp bindings).

**Important:** This is designed for free-tier / CPU deployments where the model file is hosted externally (Hugging Face) and downloaded at startup. Use a Q4_0 or similar quantized GGUF file to reduce memory usage.

## Required environment variables (set in Railway project settings)
- `MODEL_URL` (required): Direct download link to the GGUF file (e.g. https://huggingface.co/your-user/your-model/resolve/main/model.gguf)
- `MODEL_FILE` (optional): filename to save the model as under `/app/model` (default: model.gguf)
- `MODEL_DIR` (optional): directory where model is stored (default: /app/model)
- `PORT` (optional): port for the server (default: 8080)
- `MAX_TOKENS`, `TEMPERATURE` (optional): generation params defaults

## How it works
1. On container start, `start.sh` downloads the GGUF file from `MODEL_URL` into `/app/model/` (if not already present).
2. `start.sh` then launches `uvicorn server:app`.
3. The FastAPI server lazily loads the model using `llama-cpp-python` when a `/generate` request is received.

## Usage example (after deploy)
```bash
curl -X POST https://<your-railway-host>/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a 100-word story about a lighthouse.", "max_tokens":120}'
```

## Notes & Caveats
- Railway free tier provides limited RAM/disk. Use the smallest GGUF quantization available (q4_0, q4_k) to increase the chance of successful runs.
- Compilation of `llama-cpp-python` may take time during `pip install` — the Docker build includes the build step.
- If your Hugging Face model repo is private, you must provide an authenticated download link (or use `hf_hub_download` with token); this template expects a public direct link.
