#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def load_texts(json_path: Path) -> List[str]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    chunks = payload.get("chunks", [])
    return [chunk.get("text", "") for chunk in chunks]


def embed_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    cache_dir: Path | None,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()

    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings.append(pooled.cpu().numpy())
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vectorize text chunks from JSON using mxbai-embed-large-v1."
    )
    parser.add_argument("input_json", help="Path to input JSON (out.json).")
    parser.add_argument(
        "output_vectors",
        help="Path to output vectors (.npz recommended).",
    )
    parser.add_argument(
        "--model",
        default="mxbai-embed-large-v1",
        help="Model name or path (default: mxbai-embed-large-v1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding (default: 8).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache/download the model.",
    )
    args = parser.parse_args()

    json_path = Path(args.input_json)
    out_path = Path(args.output_vectors)
    texts = load_texts(json_path)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    local_model_dir = Path("/home/linkages/cursor/pdftext/models/mxbai-embed-large-v1")
    model_source = args.model
    if args.model == "mxbai-embed-large-v1" and local_model_dir.exists():
        model_source = str(local_model_dir)
    vectors = embed_texts(
        texts,
        model_name=model_source,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
    )

    metadata = {
        "source_json": str(json_path),
        "model": args.model,
        "vector_count": int(vectors.shape[0]),
        "vector_dim": int(vectors.shape[1]) if vectors.size else 0,
    }
    np.savez_compressed(out_path, vectors=vectors, metadata=json.dumps(metadata))
    print(f"Saved {metadata['vector_count']} vectors to {out_path}")


if __name__ == "__main__":
    main()
