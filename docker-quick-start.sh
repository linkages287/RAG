#!/bin/bash
# Quick start script for Docker setup

echo "=========================================="
echo "PDF Text RAG - Docker Quick Start"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Generate requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo "📦 Generating requirements.txt..."
    if [ -d "venv" ]; then
        source venv/bin/activate
        pip freeze > requirements.txt
        echo "✓ Requirements file generated"
    else
        echo "⚠ Warning: venv not found. Please create requirements.txt manually."
    fi
fi

# Build and start services
echo ""
echo "🐳 Building Docker image and starting services..."
echo ""

# Use docker compose (newer) or docker-compose (older)
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

$COMPOSE_CMD up -d --build

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check service status
echo ""
echo "📊 Service Status:"
$COMPOSE_CMD ps

echo ""
echo "=========================================="
echo "✓ Services started!"
echo "=========================================="
echo ""
echo "Available services:"
echo "  - Main App:        http://localhost:5000"
echo "  - Chat App:        http://localhost:5001"
echo "  - Weaviate RAG:    http://localhost:5003"
echo "  - LangChain RAG:   http://localhost:5004"
echo "  - Weaviate API:    http://localhost:8080"
echo "  - Ollama API:      http://localhost:11434"
echo ""
echo "To run an application:"
echo "  docker exec -it pdftext-rag python3 app.py"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
