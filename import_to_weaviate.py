#!/usr/bin/env python3
"""
Import vectors from npz files and metadata from JSON files into Weaviate vector database.
Uses Weaviate v4 client API.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType


def extract_country(source_pdf: str | None) -> str:
    """Extract country code from PDF filename."""
    if not source_pdf:
        return "unknown"
    match = re.search(r"NU_JWC_(?:ETI_)?([A-Z]+)", str(source_pdf).upper())
    if match:
        return match.group(1)
    return Path(source_pdf).stem if source_pdf else "unknown"


def load_vectors(npz_path: Path) -> np.ndarray:
    """Load vectors from npz file."""
    data = np.load(npz_path, allow_pickle=False)
    return data["vectors"]


def load_chunks(json_path: Path) -> List[Dict]:
    """Load chunks from JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("chunks", [])


def create_collection(client: weaviate.WeaviateClient, collection_name: str = "DocumentChunk") -> None:
    """Create or update Weaviate collection for document chunks."""
    # Check if collection exists
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Skipping schema creation.")
        return
    
    # Create collection with properties
    # Try Vectors (plural) first, fallback to Vector (singular) for older API
    try:
        vector_config = Configure.Vectors.none()
    except AttributeError:
        vector_config = Configure.Vector.none()
    
    client.collections.create(
        name=collection_name,
        description="Document chunks with embeddings for RAG system",
        vector_config=vector_config,  # We provide our own vectors
        properties=[
            Property(
                name="chunk_id",
                data_type=DataType.INT,
                description="Unique chunk identifier",
            ),
            Property(
                name="text",
                data_type=DataType.TEXT,
                description="Chunk text content",
            ),
            Property(
                name="source_pdf",
                data_type=DataType.TEXT,
                description="Source PDF filename",
            ),
            Property(
                name="country",
                data_type=DataType.TEXT,
                description="Country code extracted from PDF name",
            ),
            Property(
                name="page_start",
                data_type=DataType.INT,
                description="Starting page number",
            ),
            Property(
                name="page_end",
                data_type=DataType.INT,
                description="Ending page number",
            ),
            Property(
                name="token_count",
                data_type=DataType.INT,
                description="Number of tokens in chunk",
            ),
        ],
    )
    print(f"Created collection '{collection_name}'")


def import_vectors_to_weaviate(
    json_path: Path,
    npz_path: Path,
    client: weaviate.WeaviateClient,
    collection_name: str = "DocumentChunk",
    batch_size: int = 100,
    clear_existing: bool = False,
) -> None:
    """Import vectors and metadata from npz and JSON files to Weaviate."""
    print(f"Loading vectors from {npz_path}...")
    vectors = load_vectors(npz_path)
    print(f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
    
    print(f"Loading chunks from {json_path}...")
    chunks = load_chunks(json_path)
    print(f"Loaded {len(chunks)} chunks")
    
    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {vectors.shape[0]} vectors"
        )
    
    # Clear existing data if requested
    if clear_existing:
        print(f"Clearing existing data in collection '{collection_name}'...")
        try:
            if client.collections.exists(collection_name):
                client.collections.delete(collection_name)
            create_collection(client, collection_name)
        except Exception as e:
            print(f"Warning: Could not clear existing data: {e}")
    
    # Ensure collection exists
    if not client.collections.exists(collection_name):
        create_collection(client, collection_name)
    
    # Get collection
    collection = client.collections.get(collection_name)
    
    # Batch import
    total = len(chunks)
    imported = 0
    
    print(f"Importing {total} chunks in batches of {batch_size}...")
    
    # Import using batch context manager
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        # Prepare properties
        properties = {
            "chunk_id": chunk.get("chunk_id", i),
            "text": chunk.get("text", ""),
            "source_pdf": chunk.get("source_pdf", "unknown"),
            "country": extract_country(chunk.get("source_pdf")),
            "page_start": chunk.get("page_start", 0),
            "page_end": chunk.get("page_end", 0),
            "token_count": chunk.get("token_count", 0),
        }
        
        # Insert object with vector
        collection.data.insert(
            properties=properties,
            vector=vector.tolist(),  # Convert numpy array to list
        )
        
        imported += 1
        if (i + 1) % 100 == 0:
            print(f"Imported {i + 1}/{total} chunks ({100 * (i + 1) / total:.1f}%)")
    
    print(f"\nSuccessfully imported {imported} chunks to Weaviate collection '{collection_name}'")


def verify_import(client: weaviate.WeaviateClient, collection_name: str = "DocumentChunk", limit: int = 5) -> None:
    """Verify the import by querying a few objects."""
    print(f"\nVerifying import (showing {limit} sample objects)...")
    
    collection = client.collections.get(collection_name)
    result = collection.query.fetch_objects(limit=limit)
    
    if result.objects:
        print(f"Found {len(result.objects)} objects in collection '{collection_name}'")
        for obj in result.objects:
            props = obj.properties
            print(f"  - Chunk ID: {props.get('chunk_id')}, "
                  f"Country: {props.get('country')}, "
                  f"Source: {props.get('source_pdf')}, "
                  f"Pages: {props.get('page_start')}-{props.get('page_end')}")
            text_preview = props.get('text', '')[:100]
            print(f"    Text preview: {text_preview}...")
    else:
        print("No objects found or query failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import vectors and metadata from npz/JSON files to Weaviate."
    )
    parser.add_argument(
        "json_file",
        help="Path to JSON file with chunk metadata.",
    )
    parser.add_argument(
        "npz_file",
        help="Path to npz file with vectors.",
    )
    parser.add_argument(
        "--collection-name",
        default="DocumentChunk",
        help="Weaviate collection name (default: DocumentChunk).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for import (default: 100).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data in the collection before import.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify import by querying sample objects.",
    )
    args = parser.parse_args()

    json_path = Path(args.json_file)
    npz_path = Path(args.npz_file)
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return
    
    if not npz_path.exists():
        print(f"Error: NPZ file not found: {npz_path}")
        return

    # Connect to Weaviate (using local connection)
    print("Connecting to Weaviate...")
    try:
        with weaviate.connect_to_local() as client:
            if not client.is_ready():
                print("Error: Weaviate is not ready")
                return
            print("Connected to Weaviate successfully")
            
            # Import data
            try:
                import_vectors_to_weaviate(
                    json_path,
                    npz_path,
                    client,
                    args.collection_name,
                    args.batch_size,
                    args.clear,
                )
                
                if args.verify:
                    verify_import(client, args.collection_name)
                    
            except Exception as e:
                print(f"Error during import: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        print("Make sure Weaviate is running. You can start it with:")
        print("  docker run -d -p 8080:8080 semitechnologies/weaviate:latest")
        return


if __name__ == "__main__":
    main()
