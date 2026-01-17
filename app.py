#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import ollama
import torch
from flask import Flask, abort, render_template, request
from transformers import AutoModel, AutoTokenizer


# Mean-pool token embeddings with attention mask to get a sentence embedding.
def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


# Embed a query string using the local transformer model.
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


# Compute cosine similarity between a matrix of vectors and a single vector.
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return np.dot(a_norm, b_norm)


# Call the local Ollama server with a prompt and return the response text.
def call_ollama(prompt: str, model: str, base_url: str) -> str:
    client = ollama.Client(host=base_url)
    response = client.generate(model=model, prompt=prompt, stream=False)
    return response.get("response", "")


# Build the Flask app and wire up the RAG + LLM search flow.
def create_app(
    default_json: Path, default_vectors: Path, model_path: str, ollama_model: str
) -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        json_path = request.args.get("path") or str(default_json)
        vectors_path = request.args.get("vectors") or str(default_vectors)
        query = request.args.get("q", "").strip()
        top_k = int(request.args.get("k", "5"))
        use_llm = request.args.get("use_llm", "on") == "on"
        file_path = Path(json_path)
        if not file_path.exists() or not file_path.is_file():
            abort(404, description=f"JSON file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        chunks = payload.get("chunks", [])
        search_results = []
        llm_answer = ""
        if query:
            vectors_file = Path(vectors_path)
            if not vectors_file.exists() or not vectors_file.is_file():
                abort(404, description=f"Vectors file not found: {vectors_file}")
            vectors = np.load(vectors_file, allow_pickle=False)["vectors"]
            if len(chunks) != vectors.shape[0]:
                abort(400, description="Vector count does not match chunk count.")
            query_vec = embed_query(query, model_path)
            scores = cosine_sim(vectors, query_vec)
            top_idx = np.argsort(scores)[::-1][:top_k]
            for rank, idx in enumerate(top_idx, start=1):
                chunk = chunks[int(idx)]
                search_results.append(
                    {
                        "rank": rank,
                        "score": float(scores[idx]),
                        "chunk": chunk,
                    }
                )
            if use_llm:
                top_context = "\n\n".join(
                    f"[Chunk {r['chunk']['chunk_id']}] {r['chunk']['text']}"
                    for r in search_results[:5]
                )
                prompt = (
                    "Answer the user question using the context below as the primary "
                    "source of truth. You may use your general knowledge to add useful "
                    "background or clarification, but do not contradict the context. "
                    "If the context is insufficient, say so explicitly.\n\n"
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
            source_pdf=payload.get("source_pdf"),
            max_tokens=payload.get("max_tokens"),
            chunk_count=payload.get("chunk_count", len(chunks)),
            chunks=chunks,
            json_path=str(file_path),
            vectors_path=vectors_path,
            query=query,
            top_k=top_k,
            search_results=search_results,
            llm_answer=llm_answer,
            use_llm=use_llm,
        )

    return app


# CLI entry point to run the Flask server.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flask UI to view PDF text chunks and search vectors."
    )
    parser.add_argument(
        "--json",
        default="out.json",
        help="Default JSON file to load (default: out.json).",
    )
    parser.add_argument(
        "--vectors",
        default="out_vectors.npz",
        help="Default vectors file to load (default: out_vectors.npz).",
    )
    parser.add_argument(
        "--model-path",
        default="/home/linkages/cursor/pdftext/models/mxbai-embed-large-v1",
        help="Local model path (default: mxbai-embed-large-v1).",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model name (default: llama3.2).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1).",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000).")
    args = parser.parse_args()

    app = create_app(Path(args.json), Path(args.vectors), args.model_path, args.ollama_model)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
