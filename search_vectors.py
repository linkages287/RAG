#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def embed_query(query: str, model_path: str) -> np.ndarray:
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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return np.dot(a_norm, b_norm)


def load_vectors(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=False)
    return data["vectors"]


def load_chunks(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("chunks", [])


def top_k(
    vectors: np.ndarray, query_vec: np.ndarray, k: int
) -> List[Tuple[int, float]]:
    scores = cosine_sim(vectors, query_vec)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search chunk vectors with a text query."
    )
    parser.add_argument("vectors_npz", help="Path to vectors .npz file.")
    parser.add_argument("chunks_json", help="Path to chunks JSON (out.json).")
    parser.add_argument("query", help="Query text.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to show (default: 5).",
    )
    parser.add_argument(
        "--model-path",
        default="/home/linkages/cursor/pdftext/models/mxbai-embed-large-v1",
        help="Local model path (default: local mxbai-embed-large-v1).",
    )
    args = parser.parse_args()

    vectors = load_vectors(Path(args.vectors_npz))
    chunks = load_chunks(Path(args.chunks_json))
    if len(chunks) != vectors.shape[0]:
        raise SystemExit(
            "Vector count does not match chunk count. Regenerate vectors."
        )

    query_vec = embed_query(args.query, args.model_path)
    results = top_k(vectors, query_vec, args.top_k)

    for rank, (idx, score) in enumerate(results, start=1):
        chunk = chunks[idx]
        print(f"\n#{rank} score={score:.4f} chunk_id={chunk.get('chunk_id')}")
        print(f"pages: {chunk.get('page_start')}-{chunk.get('page_end')}")
        print(chunk.get("text", ""))


if __name__ == "__main__":
    main()
