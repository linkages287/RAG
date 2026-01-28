#!/bin/bash
# Docker run command for app_weaviate_rag.py
# This connects to Weaviate running on the host machine at 127.0.0.1:8080

# Option 1: Use --network host (Linux only) - simplest approach
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

# Option 2: Use host.docker.internal (if Docker supports it)
# docker run --rm -it \
#   --add-host=host.docker.internal:host-gateway \
#   -p 5003:5003 \
#   -v "$(pwd)":/app \
#   pdftext-rag:latest \
#   python3 app_weaviate_rag.py \
#     --collections "countrymodels:cms" "copd:structdoc" \
#     --model-path /app/models/mxbai-embed-large-v1 \
#     --ollama-model llama3.2 \
#     --weaviate-url http://host.docker.internal:8080 \
#     --chat-log /app/weaviate_chat_history.json \
#     --host 0.0.0.0 \
#     --port 5003
