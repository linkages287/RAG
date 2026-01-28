#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_vectors(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=False)
    return data["vectors"]


def load_chunks(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("chunks", [])


def group_by_document(chunks: List[dict]) -> Dict[str, List[Tuple[int, dict]]]:
    docs: Dict[str, List[Tuple[int, dict]]] = {}
    for idx, chunk in enumerate(chunks):
        doc = chunk.get("source_pdf", "unknown")
        docs.setdefault(doc, []).append((idx, chunk))
    return docs


def chunk_sections(
    doc_chunks: List[Tuple[int, dict]], section_size: int
) -> List[List[Tuple[int, dict]]]:
    return [
        doc_chunks[i : i + section_size]
        for i in range(0, len(doc_chunks), section_size)
        if doc_chunks[i : i + section_size]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a vector tree: chunk -> section -> document."
    )
    parser.add_argument("vectors_npz", help="Path to vectors .npz file.")
    parser.add_argument("chunks_json", help="Path to chunks JSON (cms.json).")
    parser.add_argument("output_json", help="Path to output tree JSON.")
    parser.add_argument(
        "--section-size",
        type=int,
        default=5,
        help="Chunks per section (default: 5).",
    )
    parser.add_argument(
        "--output-npz",
        default=None,
        help="Optional NPZ path for section/doc vectors.",
    )
    args = parser.parse_args()

    vectors = load_vectors(Path(args.vectors_npz))
    chunks = load_chunks(Path(args.chunks_json))
    if len(chunks) != vectors.shape[0]:
        raise SystemExit("Vector count does not match chunk count.")

    docs = group_by_document(chunks)

    tree = {"documents": []}
    section_vectors: List[np.ndarray] = []
    doc_vectors: List[np.ndarray] = []

    for doc_name, doc_chunks in docs.items():
        sections = chunk_sections(doc_chunks, args.section_size)
        doc_section_ids = []
        for section_idx, section in enumerate(sections, start=1):
            chunk_indices = [idx for idx, _chunk in section]
            section_vec = vectors[chunk_indices].mean(axis=0)
            section_id = len(section_vectors)
            section_vectors.append(section_vec)
            doc_section_ids.append(section_id)
            tree["documents"].append(
                {
                    "type": "section",
                    "section_id": section_id,
                    "document": doc_name,
                    "section_index": section_idx,
                    "chunk_indices": chunk_indices,
                }
            )

        if doc_section_ids:
            doc_vec = np.vstack([section_vectors[i] for i in doc_section_ids]).mean(axis=0)
        else:
            doc_vec = vectors[[idx for idx, _chunk in doc_chunks]].mean(axis=0)
        doc_id = len(doc_vectors)
        doc_vectors.append(doc_vec)
        tree["documents"].append(
            {
                "type": "document",
                "document_id": doc_id,
                "document": doc_name,
                "section_ids": doc_section_ids,
                "chunk_indices": [idx for idx, _chunk in doc_chunks],
            }
        )

    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    if args.output_npz:
        np.savez_compressed(
            args.output_npz,
            section_vectors=np.vstack(section_vectors) if section_vectors else np.zeros((0, vectors.shape[1])),
            doc_vectors=np.vstack(doc_vectors) if doc_vectors else np.zeros((0, vectors.shape[1])),
        )

    print(f"Saved tree to {args.output_json}")
    if args.output_npz:
        print(f"Saved section/doc vectors to {args.output_npz}")


if __name__ == "__main__":
    main()
