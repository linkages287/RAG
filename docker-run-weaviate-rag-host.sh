#!/bin/bash
# Docker run command for app_weaviate_rag.py
# Connects to Weaviate running on the host machine at 127.0.0.1:8080

# Option 1: Use --network host (Linux only) - RECOMMENDED
# This makes the container use the host's network stack directly
docker run --rm -it \
  --network host \
  -v "$(pwd)":/app \
  pdftext-rag:latest \
  python3 app_weaviate_rag.py \
    --collections "countrymodels:cms" "copd:structdoc" \
    --model-path /app/models/mxbai-embed-large-v1 \
    --ollama-model llama3.2 \
    --weaviate-url http://127.0.0.1:8080 \
    --chat-log /app/weaviate_chat_history.json \
    --host 0.0.0.0 \
    --port 5003
