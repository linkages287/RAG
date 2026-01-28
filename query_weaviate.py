#!/usr/bin/env python3
"""
Query Weaviate vector database to find the closest chunks to a given query.
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import weaviate
from transformers import AutoModel, AutoTokenizer


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings with attention mask to get a sentence embedding."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def embed_query(query: str, model_path: str) -> np.ndarray:
    """Embed a query string using the local transformer model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        outputs = model(**inputs)
        pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    return pooled.cpu().numpy()[0]


def query_weaviate(
    query: str,
    collection_name: str,
    model_path: str,
    top_k: int = 5,
    limit: int = 10,
) -> List[dict]:
    """
    Query Weaviate to find the closest chunks to a given query.
    
    Args:
        query: The search query string
        collection_name: Name of the Weaviate collection
        model_path: Path to the embedding model
        top_k: Number of top results to return
        limit: Maximum number of results to fetch from Weaviate
        
    Returns:
        List of dictionaries containing chunk information and similarity scores
    """
    # Embed the query
    print(f"Embedding query: '{query}'...")
    query_vector = embed_query(query, model_path)
    
    # Connect to Weaviate
    print("Connecting to Weaviate...")
    with weaviate.connect_to_local() as client:
        if not client.is_ready():
            raise ConnectionError("Weaviate is not ready")
        
        # Get collection
        if not client.collections.exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        collection = client.collections.get(collection_name)
        
        # Perform vector similarity search
        print(f"Searching for top {top_k} closest chunks...")
        result = collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
        )
        
        # Process results
        results = []
        for obj in result.objects:
            props = obj.properties
            metadata = obj.metadata
            
            results.append({
                "chunk_id": props.get("chunk_id"),
                "text": props.get("text", ""),
                "source_pdf": props.get("source_pdf", "unknown"),
                "country": props.get("country", "unknown"),
                "page_start": props.get("page_start", 0),
                "page_end": props.get("page_end", 0),
                "token_count": props.get("token_count", 0),
                "distance": metadata.distance if metadata else None,
                "similarity": 1 - metadata.distance if metadata and metadata.distance else None,
            })
        
        return results[:top_k]


def display_results(results: List[dict], query: str) -> None:
    """Display search results in a formatted way."""
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result #{i}")
        print("-" * 80)
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Country: {result['country']}")
        print(f"Source PDF: {result['source_pdf']}")
        print(f"Pages: {result['page_start']}-{result['page_end']}")
        print(f"Tokens: {result['token_count']}")
        if result['distance'] is not None:
            print(f"Distance: {result['distance']:.4f}")
        if result['similarity'] is not None:
            print(f"Similarity: {result['similarity']:.4f} ({result['similarity']*100:.2f}%)")
        print(f"\nText:\n{result['text'][:500]}{'...' if len(result['text']) > 500 else ''}")
        print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Weaviate vector database for closest chunks to a query."
    )
    parser.add_argument(
        "query",
        help="Search query string.",
    )
    parser.add_argument(
        "--collection-name",
        default="DocumentChunk",
        help="Weaviate collection name (default: DocumentChunk).",
    )
    parser.add_argument(
        "--model-path",
        default="/home/linkages/cursor/pdftext/models/mxbai-embed-large-v1",
        help="Path to embedding model (default: mxbai-embed-large-v1).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to fetch from Weaviate (default: 10).",
    )
    parser.add_argument(
        "--json-output",
        help="Optional JSON file to save results.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        return

    try:
        # Query Weaviate
        results = query_weaviate(
            args.query,
            args.collection_name,
            str(model_path),
            args.top_k,
            args.limit,
        )
        
        # Display results
        display_results(results, args.query)
        
        # Save to JSON if requested
        if args.json_output:
            import json
            output_path = Path(args.json_output)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "query": args.query,
                    "results": results,
                }, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
