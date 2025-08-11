#!/usr/bin/env bash
set -euo pipefail

# Environment variables:
# MODEL_URL     (required) - full direct URL to GGUF file on Hugging Face (or other HTTP host)
# MODEL_FILE    (optional) - filename to save as under /app/model/, default: model.gguf
# MODEL_DIR     (optional) - directory to store model, default: /app/model
# PORT          (optional) - server port, default: 8080
MODEL_URL=${MODEL_URL:-"https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"}
MODEL_FILE=${MODEL_FILE:-model.gguf}
MODEL_DIR=${MODEL_DIR:-/app/model}
PORT=${PORT:-8080}

if [ -z "$MODEL_URL" ]; then
  echo "ERROR: MODEL_URL environment variable is required (direct link to GGUF file)."
  exit 1
fi

mkdir -p "$MODEL_DIR"
TARGET="$MODEL_DIR/$MODEL_FILE"

if [ ! -f "$TARGET" ]; then
  echo "Downloading model from $MODEL_URL to $TARGET ..."
  # retry download with curl
  curl -L --retry 5 --retry-delay 5 -o "$TARGET" "$MODEL_URL"
else
  echo "Model already present at $TARGET - skipping download."
fi

echo "Starting FastAPI server (uvicorn) ..."
# run uvicorn; server.py will load model from MODEL_DIR/MODEL_FILE
exec uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1
