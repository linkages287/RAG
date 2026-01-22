#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ollama
import torch
from flask import Flask, abort, render_template, request
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


def call_ollama(prompt: str, model: str, base_url: str) -> str:
    client = ollama.Client(host=base_url)
    response = client.generate(model=model, prompt=prompt, stream=False)
    return response.get("response", "")


def extract_country(source_pdf: str | None) -> str:
    if not source_pdf:
        return "unknown"
    match = re.search(r"NU_JWC_(?:ETI_)?([A-Z]+)", source_pdf.upper())
    if match:
        return match.group(1)
    return Path(source_pdf).stem


def load_chunks(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("chunks", [])


def load_tree(tree_path: Path) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    with tree_path.open("r", encoding="utf-8") as f:
        tree = json.load(f)
    sections: Dict[int, dict] = {}
    documents: Dict[int, dict] = {}
    for entry in tree.get("documents", []):
        if entry.get("type") == "section":
            sections[int(entry["section_id"])] = entry
        elif entry.get("type") == "document":
            documents[int(entry["document_id"])] = entry
    return sections, documents


def create_app(
    default_json: Path,
    default_vectors: Path,
    default_tree_json: Path,
    default_tree_vectors: Path,
    model_path: str,
    ollama_model: str,
) -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        max_chunks = 10
        json_path = request.args.get("path") or str(default_json)
        vectors_path = request.args.get("vectors") or str(default_vectors)
        tree_path = request.args.get("tree") or str(default_tree_json)
        tree_vectors_path = request.args.get("tree_vectors") or str(default_tree_vectors)
        query = request.args.get("q", "").strip()
        top_k = int(request.args.get("k", "5"))
        doc_top_k = int(request.args.get("doc_k", "5"))
        section_top_k = int(request.args.get("section_k", "10"))
        use_llm = request.args.get("use_llm", "on") == "on"

        file_path = Path(json_path)
        if not file_path.exists() or not file_path.is_file():
            abort(404, description=f"JSON file not found: {file_path}")

        chunks = load_chunks(file_path)
        vectors = np.load(Path(vectors_path), allow_pickle=False)["vectors"]
        if len(chunks) != vectors.shape[0]:
            abort(400, description="Vector count does not match chunk count.")

        sections, documents = load_tree(Path(tree_path))
        tree_vectors = np.load(Path(tree_vectors_path), allow_pickle=False)
        section_vectors = tree_vectors.get("section_vectors")
        doc_vectors = tree_vectors.get("doc_vectors")

        search_results = []
        all_results = []
        llm_answer = ""

        if query:
            query_vec = embed_query(query, model_path)
            allowed_section_ids = None

            if doc_vectors is not None and documents:
                doc_scores = cosine_sim(doc_vectors, query_vec)
                doc_sorted = np.argsort(doc_scores)[::-1][:doc_top_k]
                allowed_section_ids = set()
                for doc_id in doc_sorted:
                    doc_entry = documents.get(int(doc_id))
                    if not doc_entry:
                        continue
                    for section_id in doc_entry.get("section_ids", []):
                        allowed_section_ids.add(int(section_id))

            section_scores = cosine_sim(section_vectors, query_vec)
            section_sorted = np.argsort(section_scores)[::-1]
            if allowed_section_ids is not None:
                section_sorted = np.array(
                    [i for i in section_sorted if int(i) in allowed_section_ids]
                )
            section_sorted = section_sorted[:section_top_k]

            candidate_chunk_indices = set()
            for section_id in section_sorted:
                section = sections.get(int(section_id))
                if not section:
                    continue
                candidate_chunk_indices.update(section.get("chunk_indices", []))

            candidate_chunk_indices = sorted(candidate_chunk_indices)
            if not candidate_chunk_indices:
                candidate_chunk_indices = list(range(len(chunks)))

            candidate_vectors = vectors[candidate_chunk_indices]
            candidate_scores = cosine_sim(candidate_vectors, query_vec)
            candidate_sorted = np.argsort(candidate_scores)[::-1]

            for rank, idx in enumerate(candidate_sorted[:top_k], start=1):
                chunk_idx = candidate_chunk_indices[int(idx)]
                chunk = chunks[chunk_idx]
                search_results.append(
                    {
                        "rank": rank,
                        "score": float(candidate_scores[idx]),
                        "chunk": chunk,
                        "country": extract_country(chunk.get("source_pdf")),
                    }
                )
            for idx in candidate_sorted:
                chunk_idx = candidate_chunk_indices[int(idx)]
                chunk = chunks[chunk_idx]
                all_results.append(
                    {
                        "score": float(candidate_scores[idx]),
                        "chunk": chunk,
                        "country": extract_country(chunk.get("source_pdf")),
                        "source_pdf": chunk.get("source_pdf", "unknown"),
                    }
                )

            if use_llm and candidate_sorted.size > 0:
                best_score = float(candidate_scores[candidate_sorted[0]])
                threshold = best_score * 0.85
                eligible_idx = [
                    i for i in candidate_sorted if float(candidate_scores[i]) >= threshold
                ]
                context_count = min(10, len(eligible_idx))
                context_idx = eligible_idx[:context_count]
                top_k = context_count
                top_context = "\n\n".join(
                    f"[Source {extract_country(chunks[candidate_chunk_indices[int(i)]].get('source_pdf'))}] "
                    f"{chunks[candidate_chunk_indices[int(i)]]['text']}"
                    for i in context_idx
                )
                prompt = (
                    "Answer the user question using the context below as the primary "
                    "source of truth. use only the context provided."
                    "If the context is insufficient, say so explicitly."
                    "Give a detailed answer as much as possible keeping in .mind the relationships.\n\n"
                    f"User question: {query}\n\n"
                    f"Context:\n{top_context}\n"
                )
                try:
                    print("\n--- Injected RAG Prompt ---\n")
                    print(prompt)
                    llm_answer = call_ollama(
                        prompt=prompt,
                        model=ollama_model,
                        base_url="http://localhost:11434",
                    )
                except Exception as exc:
                    llm_answer = f"LLM error: {exc}"

        return render_template(
            "index.html",
            source_pdf="multiple",
            max_tokens=None,
            chunk_count=len(chunks),
            chunks=chunks,
            json_path=str(file_path),
            vectors_path=vectors_path,
            query=query,
            top_k=top_k,
            search_results=search_results,
            all_results=all_results,
            llm_answer=llm_answer,
            use_llm=use_llm,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flask UI using vector tree for coarse-to-fine search."
    )
    parser.add_argument("--json", default="cms.json", help="Chunks JSON.")
    parser.add_argument("--vectors", default="cms.npz", help="Chunk vectors NPZ.")
    parser.add_argument("--tree-json", default="vector_tree.json", help="Tree JSON.")
    parser.add_argument(
        "--tree-vectors",
        default="vector_tree_vectors.npz",
        help="Tree vectors NPZ.",
    )
    parser.add_argument(
        "--model-path",
        default="/home/linkages/cursor/pdftext/models/mxbai-embed-large-v1",
        help="Local model path.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model name (default: llama3.2).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=5001, help="Port (default: 5001).")
    args = parser.parse_args()

    app = create_app(
        Path(args.json),
        Path(args.vectors),
        Path(args.tree_json),
        Path(args.tree_vectors),
        args.model_path,
        args.ollama_model,
    )
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
