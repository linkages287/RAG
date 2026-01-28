#!/bin/bash
# Test script to diagnose Docker container issues

echo "=========================================="
echo "Testing Docker Container Setup"
echo "=========================================="

echo ""
echo "1. Testing Python imports..."
docker run --rm --network host \
  -v "$(pwd)":/app \
  pdftext-rag:latest \
  python3 -c "
import sys
print('✓ Python:', sys.version.split()[0])

try:
    import torch
    print('✓ PyTorch:', torch.__version__)
except ImportError as e:
    print('✗ PyTorch import failed:', e)
    sys.exit(1)

try:
    import transformers
    print('✓ Transformers:', transformers.__version__)
except ImportError as e:
    print('✗ Transformers import failed:', e)
    sys.exit(1)

try:
    import weaviate
    print('✓ Weaviate:', weaviate.__version__)
except ImportError as e:
    print('✗ Weaviate import failed:', e)
    sys.exit(1)

try:
    from ollama_api import call_ollama_api, stream_ollama_api
    print('✓ ollama_api module imported')
except ImportError as e:
    print('✗ ollama_api import failed:', e)
    sys.exit(1)

try:
    from flask import Flask
    print('✓ Flask imported')
except ImportError as e:
    print('✗ Flask import failed:', e)
    sys.exit(1)

print('\\n✓ All imports successful!')
"

echo ""
echo "2. Testing model path..."
docker run --rm --network host \
  -v "$(pwd)":/app \
  pdftext-rag:latest \
  python3 -c "
from pathlib import Path
model_path = Path('/app/models/mxbai-embed-large-v1')
if model_path.exists():
    print('✓ Model path exists:', model_path)
    config_file = model_path / 'config.json'
    if config_file.exists():
        print('✓ config.json found')
    else:
        print('✗ config.json not found')
else:
    print('✗ Model path does not exist:', model_path)
"

echo ""
echo "3. Testing Weaviate connection..."
docker run --rm --network host \
  -v "$(pwd)":/app \
  pdftext-rag:latest \
  python3 -c "
import weaviate
try:
    client = weaviate.connect_to_custom(
        http_host='127.0.0.1',
        http_port=8080,
        http_secure=False,
    )
    if client.is_ready():
        print('✓ Weaviate connection successful')
        client.close()
    else:
        print('✗ Weaviate is not ready')
except Exception as e:
    print('✗ Weaviate connection failed:', str(e))
"

echo ""
echo "4. Testing app_weaviate_rag.py syntax..."
docker run --rm --network host \
  -v "$(pwd)":/app \
  pdftext-rag:latest \
  python3 -c "
import sys
sys.path.insert(0, '/app')
try:
    import app_weaviate_rag
    print('✓ app_weaviate_rag.py imported successfully')
except Exception as e:
    print('✗ app_weaviate_rag.py import failed:', str(e))
    import traceback
    traceback.print_exc()
"

echo ""
echo "=========================================="
echo "Diagnostic complete"
echo "=========================================="
