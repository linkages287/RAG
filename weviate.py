#!/usr/bin/env python3
"""
Simple test script to verify Weaviate connection.
"""
import weaviate

if __name__ == "__main__":
    try:
        with weaviate.connect_to_local() as client:
            if client.is_ready():
                print("✓ Weaviate is ready and connected")
                print(f"  Version: {client._connection.get_meta() if hasattr(client, '_connection') else 'unknown'}")
            else:
                print("✗ Weaviate is not ready")
    except Exception as e:
        print(f"✗ Error connecting to Weaviate: {e}")
        print("\nMake sure Weaviate is running:")
        print("  docker run -d -p 8080:8080 semitechnologies/weaviate:latest")



