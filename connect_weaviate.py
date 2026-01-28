#!/usr/bin/env python3
"""
Simple program to connect to Weaviate database and perform basic operations.

Usage:
    python connect_weaviate.py
    python connect_weaviate.py --url http://localhost:8080
    python connect_weaviate.py --url http://192.168.1.100:8080
"""

import argparse
import sys
from typing import Optional

try:
    import weaviate
except ImportError:
    print("Error: weaviate package is not installed.")
    print("Install it with: pip install weaviate-client")
    sys.exit(1)


def connect_to_weaviate(weaviate_url: str = "http://localhost:8080"):
    """
    Connect to Weaviate database.
    
    Args:
        weaviate_url: Weaviate server URL (default: http://localhost:8080)
    
    Returns:
        WeaviateClient instance or None if connection fails
    """
    try:
        # For localhost connections, use connect_to_local()
        if weaviate_url in ["http://localhost:8080", "http://127.0.0.1:8080"]:
            print(f"Connecting to local Weaviate instance...")
            client = weaviate.connect_to_local()
        else:
            # For remote connections, parse URL and use connect_to_custom()
            print(f"Connecting to Weaviate at {weaviate_url}...")
            
            # Parse URL
            url_clean = weaviate_url.replace("http://", "").replace("https://", "")
            if ":" in url_clean:
                host, port_str = url_clean.split(":", 1)
                port = int(port_str)
            else:
                host = url_clean
                port = 8080
            
            http_secure = weaviate_url.startswith("https://")
            
            # Default GRPC port is 50051
            grpc_port = 50051
            
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=http_secure,
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=http_secure,
            )
        
        # Check if Weaviate is ready
        if client.is_ready():
            print("✓ Successfully connected to Weaviate!")
            return client
        else:
            print("✗ Weaviate is not ready")
            return None
            
    except Exception as e:
        print(f"✗ Error connecting to Weaviate: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Weaviate is running")
        print("  2. Check if the URL is correct")
        print("  3. Verify network connectivity")
        if weaviate_url == "http://localhost:8080":
            print("\n  To start Weaviate locally:")
            print("    docker run -d -p 8080:8080 semitechnologies/weaviate:latest")
        return None


def list_collections(client: weaviate.WeaviateClient):
    """List all collections in Weaviate."""
    try:
        collections = client.collections.list_all()
        if collections:
            print(f"\n📚 Found {len(collections)} collection(s):")
            for collection_name in collections:
                collection = client.collections.get(collection_name)
                # Get collection info
                config = collection.config.get()
                print(f"\n  Collection: {collection_name}")
                print(f"    Description: {config.description or 'No description'}")
                # Count objects
                try:
                    count = collection.query.fetch_objects(limit=1, return_metadata=weaviate.classes.query.MetadataQuery(count=True))
                    total = count.total if hasattr(count, 'total') else 'unknown'
                    print(f"    Objects: {total}")
                except:
                    print(f"    Objects: unknown")
        else:
            print("\n📚 No collections found in Weaviate")
    except Exception as e:
        print(f"✗ Error listing collections: {e}")


def get_collection_info(client: weaviate.WeaviateClient, collection_name: str):
    """Get detailed information about a specific collection."""
    try:
        if not client.collections.exists(collection_name):
            print(f"✗ Collection '{collection_name}' does not exist")
            return
        
        collection = client.collections.get(collection_name)
        config = collection.config.get()
        
        print(f"\n📖 Collection Details: {collection_name}")
        print(f"  Description: {config.description or 'No description'}")
        print(f"  Properties:")
        for prop in config.properties:
            print(f"    - {prop.name}: {prop.data_type}")
        
        # Get object count
        try:
            result = collection.query.fetch_objects(limit=1, return_metadata=weaviate.classes.query.MetadataQuery(count=True))
            total = result.total if hasattr(result, 'total') else 'unknown'
            print(f"  Total Objects: {total}")
        except:
            pass
            
    except Exception as e:
        print(f"✗ Error getting collection info: {e}")


def query_collection(client: weaviate.WeaviateClient, collection_name: str, limit: int = 5):
    """Query a collection and return sample objects."""
    try:
        if not client.collections.exists(collection_name):
            print(f"✗ Collection '{collection_name}' does not exist")
            return
        
        collection = client.collections.get(collection_name)
        
        print(f"\n🔍 Querying collection '{collection_name}' (showing first {limit} objects)...")
        
        # Fetch objects
        result = collection.query.fetch_objects(limit=limit)
        
        if result.objects:
            print(f"  Found {len(result.objects)} object(s):")
            for i, obj in enumerate(result.objects, 1):
                print(f"\n  Object {i}:")
                print(f"    UUID: {obj.uuid}")
                if obj.properties:
                    for key, value in obj.properties.items():
                        # Truncate long text values
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        print(f"    {key}: {value}")
        else:
            print(f"  No objects found in collection '{collection_name}'")
            
    except Exception as e:
        print(f"✗ Error querying collection: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Simple program to connect to Weaviate database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to local Weaviate
  python connect_weaviate.py
  
  # Connect to remote Weaviate
  python connect_weaviate.py --url http://192.168.1.100:8080
  
  # List collections and query a specific one
  python connect_weaviate.py --collection MyCollection
        """
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Weaviate server URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--collection",
        help="Collection name to query (optional)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of objects to fetch when querying (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Connect to Weaviate
    client = connect_to_weaviate(args.url)
    if not client:
        sys.exit(1)
    
    try:
        # List all collections
        list_collections(client)
        
        # If a specific collection is requested, show details and query it
        if args.collection:
            get_collection_info(client, args.collection)
            query_collection(client, args.collection, limit=args.limit)
        
        print("\n✓ Done!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close connection if it's a context manager
        if hasattr(client, '__exit__'):
            try:
                client.__exit__(None, None, None)
            except:
                pass


if __name__ == "__main__":
    main()
