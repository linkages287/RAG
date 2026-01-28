#!/bin/bash
# Script to run app_weaviate_rag.py in Docker with port mapping
# This avoids --network host issues

PORT=5003
CONTAINER_NAME="pdftext-weaviate-rag"

echo "=========================================="
echo "Starting Weaviate RAG Application"
echo "=========================================="

# Stop and remove existing container if it exists
if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME 2>/dev/null
    docker rm $CONTAINER_NAME 2>/dev/null
    echo "✓ Container removed"
    sleep 2
fi

# Check if Weaviate is accessible from host
echo ""
echo "Checking Weaviate connection..."
if curl -s http://127.0.0.1:8080/v1/.well-known/ready >/dev/null 2>&1; then
    echo "✓ Weaviate is running and accessible"
    WEAVIATE_URL="http://host.docker.internal:8080"
else
    echo "⚠ Warning: Weaviate is not accessible at http://127.0.0.1:8080"
    echo "The app will start but may not be able to connect to Weaviate"
    WEAVIATE_URL="http://host.docker.internal:8080"
fi

echo ""
echo "Starting Docker container with port mapping..."
docker run -d \
  --name $CONTAINER_NAME \
  --add-host=host.docker.internal:host-gateway \
  -p $PORT:$PORT \
  -v "$(pwd)":/app \
  pdftext-rag:latest \
  python3 app_weaviate_rag.py \
    --collections "countrymodels:cms" "copd:structdoc" \
    --model-path /app/models/mxbai-embed-large-v1 \
    --ollama-model llama3.2 \
    --weaviate-url $WEAVIATE_URL \
    --chat-log /app/weaviate_chat_history.json \
    --host 0.0.0.0 \
    --port $PORT

sleep 3

# Check if container is running
if docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "✓ Container is running"
    echo ""
    echo "Application is available at:"
    echo "  - http://127.0.0.1:$PORT"
    echo "  - http://localhost:$PORT"
    echo ""
    echo "To view logs:"
    echo "  docker logs -f $CONTAINER_NAME"
    echo ""
    echo "To stop the container:"
    echo "  docker stop $CONTAINER_NAME"
else
    echo "✗ Container failed to start"
    echo "Checking logs..."
    docker logs $CONTAINER_NAME 2>&1 | tail -20
    exit 1
fi
