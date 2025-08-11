FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# system dependencies for building llama-cpp-python and running inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl unzip libopenblas-dev libgomp1 libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy project files
COPY requirements.txt /app/requirements.txt
COPY start.sh /app/start.sh
COPY server.py /app/server.py
COPY .dockerignore /app/.dockerignore
COPY README.md /app/README.md

RUN chmod +x /app/start.sh

# Install python deps (llama-cpp-python will compile during pip install)
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080

# start script will download model and then run the FastAPI server
CMD ["/bin/bash", "/app/start.sh"]
