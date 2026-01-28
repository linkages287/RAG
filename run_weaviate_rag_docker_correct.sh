#!/bin/bash
# Correct Docker command for app_weaviate_rag.py
# Uses port mapping and host.docker.internal to connect to host Weaviate

echo "=========================================="
echo "Starting Weaviate RAG Application"
echo "=========================================="

# Check if Weaviate is accessible from host
echo "Checking Weaviate connection..."
if curl -s http://127.0.0.1:8080/v1/.well-known/ready >/dev/null 2>&1; then
    echo "✓ Weaviate is running and accessible on host"
else
    echo "⚠ Warning: Weaviate is not accessible at http://127.0.0.1:8080"
    echo "The app will start but may not be able to connect to Weaviate"
fi

echo ""
echo "Starting Docker container..."
echo "Access the application at: http://127.0.0.1:5005"
echo ""

docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  -p 5005:5005 \
  -v "$(pwd)":/app \
  pdftext-rag:latest \
  python3 app_weaviate_rag.py \
    --collections "countrymodels:cms" "copd:structdoc" \
    --model-path /app/models/mxbai-embed-large-v1 \
    --ollama-model llama3.2 \
    --weaviate-url http://host.docker.internal:8080 \
    --chat-log /app/weaviate_chat_history.json \
    --host 0.0.0.0 \
    --port 5005
