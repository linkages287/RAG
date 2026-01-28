#!/bin/bash
# Build script for Docker container

set -e

IMAGE_NAME="pdftext-rag"
IMAGE_TAG="latest"
CONTAINER_NAME="pdftext-rag-container"

echo "=========================================="
echo "Building Docker Image: $IMAGE_NAME:$IMAGE_TAG"
echo "=========================================="

# Generate requirements.txt from current environment
echo "Generating requirements.txt..."
if [ -d "venv" ]; then
    source venv/bin/activate
    pip freeze > requirements.txt
    echo "✓ Requirements file generated"
else
    echo "⚠ Warning: venv not found, using existing requirements.txt"
fi

# Build Docker image
echo ""
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .

echo ""
echo "=========================================="
echo "✓ Docker image built successfully!"
echo "=========================================="
echo ""
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "To run the container:"
echo "  docker run -it --rm $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "To run with port mapping (for Flask apps):"
echo "  docker run -it --rm -p 5000:5000 $IMAGE_NAME:$IMAGE_TAG python3 app.py"
echo ""
echo "To run with volume mounts (for data/models):"
echo "  docker run -it --rm -v \$(pwd)/models:/app/models -v \$(pwd)/pdf:/app/pdf $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "To run with Weaviate and Ollama (using docker-compose):"
echo "  docker-compose up -d"
echo ""
