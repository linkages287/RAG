#!/usr/bin/env python3
"""
Remove collections from Weaviate vector database.
Supports deleting single or multiple collections with confirmation.
"""
import argparse
import sys
from typing import List

import weaviate


def list_collections(client: weaviate.WeaviateClient) -> List[str]:
    """List all collections in Weaviate."""
    try:
        # Get all collections - try different methods based on API version
        try:
            # Try v4 API - returns a dict, convert to list
            collections = client.collections.list_all()
            if isinstance(collections, dict):
                return list(collections.keys())
            elif isinstance(collections, list):
                return collections
            else:
                return list(collections) if hasattr(collections, '__iter__') else []
        except AttributeError:
            # Fallback: try to get schema and extract class names
            try:
                schema = client.schema.get()
                if schema and 'classes' in schema:
                    return [cls['class'] for cls in schema['classes']]
                return []
            except:
                return []
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []


def get_collection_info(client: weaviate.WeaviateClient, collection_name: str) -> dict:
    """Get information about a collection."""
    try:
        if not client.collections.exists(collection_name):
            return None
        
        collection = client.collections.get(collection_name)
        # Get object count - try to fetch with count metadata
        try:
            result = collection.query.fetch_objects(
                limit=1,
                return_metadata=weaviate.classes.query.MetadataQuery(count=True)
            )
            # Try different attributes for count
            if hasattr(result, 'total_count'):
                count = result.total_count
            elif hasattr(result, 'objects') and hasattr(result.objects, '__len__'):
                # Fallback: approximate count by fetching a larger sample
                count = "~" + str(len(collection.query.fetch_objects(limit=1000).objects))
            else:
                count = "unknown"
        except:
            count = "unknown"
        
        return {
            "name": collection_name,
            "exists": True,
            "object_count": count,
        }
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return None


def delete_collection(client: weaviate.WeaviateClient, collection_name: str, confirm: bool = False) -> bool:
    """
    Delete a collection from Weaviate.
    
    Args:
        client: Weaviate client
        collection_name: Name of collection to delete
        confirm: If True, skip confirmation prompt
        
    Returns:
        True if deleted successfully, False otherwise
    """
    if not client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return False
    
    # Get collection info
    info = get_collection_info(client, collection_name)
    if info:
        print(f"\nCollection: {info['name']}")
        print(f"  Objects: {info['object_count']}")
    
    # Confirmation
    if not confirm:
        response = input(f"\nAre you sure you want to delete collection '{collection_name}'? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Deletion cancelled.")
            return False
    
    try:
        client.collections.delete(collection_name)
        print(f"✓ Successfully deleted collection '{collection_name}'")
        return True
    except Exception as e:
        print(f"✗ Error deleting collection '{collection_name}': {e}")
        return False


def delete_all_collections(client: weaviate.WeaviateClient, confirm: bool = False) -> int:
    """
    Delete all collections from Weaviate.
    
    Args:
        client: Weaviate client
        confirm: If True, skip confirmation prompt
        
    Returns:
        Number of collections deleted
    """
    collections = list_collections(client)
    
    if not collections:
        print("No collections found.")
        return 0
    
    print(f"\nFound {len(collections)} collection(s):")
    for i, coll_name in enumerate(collections, 1):
        info = get_collection_info(client, coll_name)
        if info:
            print(f"  {i}. {coll_name} ({info['object_count']} objects)")
        else:
            print(f"  {i}. {coll_name}")
    
    # Confirmation
    if not confirm:
        response = input(f"\nAre you sure you want to delete ALL {len(collections)} collection(s)? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Deletion cancelled.")
            return 0
    
    deleted_count = 0
    for collection_name in collections:
        if delete_collection(client, collection_name, confirm=True):
            deleted_count += 1
    
    return deleted_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove collections from Weaviate vector database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all collections
  python3 remove_weaviate_collections.py --list

  # Delete a specific collection
  python3 remove_weaviate_collections.py --delete DocumentChunk

  # Delete multiple collections
  python3 remove_weaviate_collections.py --delete DocumentChunk EBENChunk

  # Delete all collections (with confirmation)
  python3 remove_weaviate_collections.py --delete-all

  # Delete without confirmation prompt
  python3 remove_weaviate_collections.py --delete DocumentChunk --yes
        """
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all collections in Weaviate.",
    )
    parser.add_argument(
        "--delete",
        nargs="+",
        metavar="COLLECTION",
        help="Delete one or more collections by name.",
    )
    parser.add_argument(
        "--delete-all",
        action="store_true",
        help="Delete all collections.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts (use with caution!).",
    )
    parser.add_argument(
        "--weaviate-url",
        default="http://localhost:8080",
        help="Weaviate server URL (default: http://localhost:8080).",
    )
    args = parser.parse_args()

    # Connect to Weaviate
    print(f"Connecting to Weaviate at {args.weaviate_url}...")
    try:
        if args.weaviate_url == "http://localhost:8080":
            client = weaviate.connect_to_local()
        else:
            client = weaviate.Client(url=args.weaviate_url)
        
        if not client.is_ready():
            print("✗ Error: Weaviate is not ready")
            sys.exit(1)
        
        print("✓ Connected to Weaviate successfully\n")
    except Exception as e:
        print(f"✗ Error connecting to Weaviate: {e}")
        print("\nMake sure Weaviate is running:")
        print("  docker run -d -p 8080:8080 semitechnologies/weaviate:latest")
        sys.exit(1)

    try:
        # List collections
        if args.list:
            collections = list_collections(client)
            if collections:
                print(f"Found {len(collections)} collection(s):\n")
                for i, coll_name in enumerate(collections, 1):
                    info = get_collection_info(client, coll_name)
                    if info:
                        print(f"  {i}. {coll_name}")
                        print(f"     Objects: {info['object_count']}")
                    else:
                        print(f"  {i}. {coll_name}")
                    print()
            else:
                print("No collections found.")
        
        # Delete specific collections
        elif args.delete:
            deleted_count = 0
            for collection_name in args.delete:
                if delete_collection(client, collection_name, confirm=args.yes):
                    deleted_count += 1
                print()
            
            print(f"Deleted {deleted_count}/{len(args.delete)} collection(s).")
        
        # Delete all collections
        elif args.delete_all:
            deleted_count = delete_all_collections(client, confirm=args.yes)
            print(f"\nDeleted {deleted_count} collection(s).")
        
        # No action specified
        else:
            parser.print_help()
            sys.exit(1)
    
    finally:
        # Close connection if it's a context manager
        if hasattr(client, '__exit__'):
            try:
                client.__exit__(None, None, None)
            except:
                pass


if __name__ == "__main__":
    main()
