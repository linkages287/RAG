#!/bin/bash
# Comprehensive test script for Weaviate connection from Docker

echo "=========================================="
echo "Testing Weaviate Connection from Docker"
echo "=========================================="

echo ""
echo "1. Checking if Weaviate is running on host..."
if curl -s http://127.0.0.1:8080/v1/.well-known/ready >/dev/null 2>&1; then
    echo "✓ Weaviate is running and accessible on host"
else
    echo "✗ Weaviate is NOT accessible on host at http://127.0.0.1:8080"
    echo "  Please start Weaviate first"
    exit 1
fi

echo ""
echo "2. Testing connection with --network host..."
docker run --rm --network host pdftext-rag:latest python3 -c "
import weaviate
try:
    client = weaviate.connect_to_local()
    if client.is_ready():
        print('✓ Weaviate connection successful with --network host')
        client.close()
    else:
        print('✗ Weaviate is not ready')
except Exception as e:
    print(f'✗ Connection failed: {e}')
" 2>&1

echo ""
echo "3. Testing connection with host.docker.internal..."
docker run --rm --add-host=host.docker.internal:host-gateway pdftext-rag:latest python3 -c "
import weaviate
try:
    client = weaviate.connect_to_custom(
        http_host='host.docker.internal',
        http_port=8080,
        http_secure=False,
        grpc_host='host.docker.internal',
        grpc_port=50051,
        grpc_secure=False,
    )
    if client.is_ready():
        print('✓ Weaviate connection successful with host.docker.internal')
        client.close()
    else:
        print('✗ Weaviate is not ready')
except Exception as e:
    print(f'✗ Connection failed: {e}')
" 2>&1

echo ""
echo "4. Testing connection with 172.17.0.1 (Docker bridge)..."
docker run --rm pdftext-rag:latest python3 -c "
import weaviate
try:
    client = weaviate.connect_to_custom(
        http_host='172.17.0.1',
        http_port=8080,
        http_secure=False,
        grpc_host='172.17.0.1',
        grpc_port=50051,
        grpc_secure=False,
    )
    if client.is_ready():
        print('✓ Weaviate connection successful with 172.17.0.1')
        client.close()
    else:
        print('✗ Weaviate is not ready')
except Exception as e:
    print(f'✗ Connection failed: {e}')
" 2>&1

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
