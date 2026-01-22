#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import ollama
import torch
from flask import Flask, abort, render_template, request, session
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


# Agent to determine if query needs RAG context or can use general knowledge.
def classify_query_agent(query: str, ollama_model: str, base_url: str) -> str:
    """
    Classify whether a query requires RAG context or can be answered with general knowledge.
    Returns: 'rag' if context needed, 'general' if general knowledge sufficient.
    """
    classification_prompt = (
        "You are a query classification agent. Analyze the following question and determine "
        "if it requires specific document context (like country profiles, military data, infrastructure details) "
        "or can be answered with general knowledge.\n\n"
        "Respond with ONLY one word: 'rag' if the question needs specific document context, "
        "or 'general' if it can be answered with general knowledge.\n\n"
        f"Question: {query}\n\n"
        "Response:"
    )
    try:
        response = call_ollama(classification_prompt, ollama_model, base_url)
        result = response.strip().lower()
        if "rag" in result or "context" in result or "document" in result:
            return "rag"
        elif "general" in result:
            return "general"
        else:
            # Default to RAG if unclear
            return "rag"
    except Exception:
        # Default to RAG on error
        return "rag"


def extract_country(source_pdf: str | None) -> str:
    if not source_pdf:
        return "unknown"
    match = re.search(r"NU_JWC_(?:ETI_)?([A-Z]+)", source_pdf.upper())
    if match:
        return match.group(1)
    return Path(source_pdf).stem


# Build the Flask app with chatbot interface.
def create_app(
    default_json: Path, default_vectors: Path, model_path: str, ollama_model: str, chat_log_path: Optional[Path] = None
) -> Flask:
    app = Flask(__name__)
    app.secret_key = "chatbot-secret-key-change-in-production"

    @app.route("/", methods=["GET", "POST"])
    def index():
        json_path = request.args.get("path") or request.form.get("path") or str(default_json)
        vectors_path = request.args.get("vectors") or request.form.get("vectors") or str(default_vectors)
        query = request.form.get("query", "").strip() if request.method == "POST" else ""
        clear_history = request.form.get("clear", "") == "clear"
        
        # Initialize or clear chat history
        if "messages" not in session or clear_history:
            session["messages"] = []
        
        file_path = Path(json_path)
        if not file_path.exists() or not file_path.is_file():
            abort(404, description=f"JSON file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        chunks = payload.get("chunks", [])
        messages = session.get("messages", [])
        
        if query:
            # Classify query to determine if RAG context is needed
            query_type = classify_query_agent(query, ollama_model, "http://localhost:11434")
            print(f"\n[Query Classification] Type: {query_type}\n")
            
            vectors_file = Path(vectors_path)
            if not vectors_file.exists() or not vectors_file.is_file():
                abort(404, description=f"Vectors file not found: {vectors_file}")
            vectors = np.load(vectors_file, allow_pickle=False)["vectors"]
            if len(chunks) != vectors.shape[0]:
                abort(400, description="Vector count does not match chunk count.")
            
            query_vec = embed_query(query, model_path)
            scores = cosine_sim(vectors, query_vec)
            sorted_idx = np.argsort(scores)[::-1]
            best_score = float(scores[sorted_idx[0]])
            
            # Override to general knowledge if best score is too low
            if best_score < 0.6:
                query_type = "general"
                print(f"\n[Score Check] Best score {best_score:.3f} < 0.6, using general knowledge mode\n")
            
            # Get top results for display
            top_results = []
            for rank, idx in enumerate(sorted_idx[:5], start=1):
                chunk = chunks[int(idx)]
                top_results.append({
                    "rank": rank,
                    "score": float(scores[idx]),
                    "chunk": chunk,
                    "country": extract_country(chunk.get("source_pdf")),
                })
            
            # Generate LLM answer
            llm_answer = ""
            if query_type == "general":
                # For general knowledge queries, answer without RAG context
                prompt = (
                    "You are a NATO analyst. "
                    "Answer the following question using your general knowledge. "
                    "Provide a detailed and accurate response.\n\n"
                    f"Question: {query}\n"
                )
                print("\n[General Knowledge Mode] Answering without RAG context\n")
            else:
                # For RAG queries, use context from documents
                max_chunks = 12
                threshold = best_score * 0.91  # 7% less than best score
                eligible_idx = [i for i in sorted_idx if float(scores[i]) >= threshold]
                context_count = min(max_chunks, len(eligible_idx))
                context_idx = eligible_idx[:context_count]
                
                top_context = "\n\n".join(
                    f"[Source {extract_country(chunks[int(i)].get('source_pdf'))}] "
                    f"{chunks[int(i)]['text']}"
                    for i in context_idx
                )
                prompt = (
                    "You are a NATO analyst."
                    "Answer the user question analizing only the context below. "
                    "If the context is insufficient, say so explicitly.\n\n"
                    f"User question: {query}\n\n"
                    f"Context:\n{top_context}\n"
                )
                print("\n--- Injected RAG Prompt ---\n")
                print(prompt)
            
            try:
                llm_answer = call_ollama(
                    prompt=prompt,
                    model=ollama_model,
                    base_url="http://localhost:11434",
                )
            except Exception as exc:
                llm_answer = f"LLM error: {exc}"
            
            # Add messages to history
            messages.append({
                "role": "user",
                "content": query,
                "query_type": query_type,
            })
            messages.append({
                "role": "assistant",
                "content": llm_answer,
                "query_type": query_type,
                "top_results": top_results,
            })
            session["messages"] = messages
            
            # Log chat history to file if enabled
            if chat_log_path:
                try:
                    # Load existing log or create new
                    if chat_log_path.exists():
                        with chat_log_path.open("r", encoding="utf-8") as f:
                            all_messages = json.load(f)
                    else:
                        all_messages = []
                    
                    # Append new messages
                    all_messages.extend(messages[-2:])  # Only the last user+assistant pair
                    
                    # Save updated log
                    with chat_log_path.open("w", encoding="utf-8") as f:
                        json.dump(all_messages, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Warning: Could not save chat log: {e}")
        
        return render_template(
            "chat.html",
            source_pdf=payload.get("source_pdf"),
            max_tokens=220,
            chunk_count=payload.get("chunk_count", len(chunks)),
            json_path=str(file_path),
            vectors_path=vectors_path,
            messages=messages,
        )

    return app


# CLI entry point to run the Flask server.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flask chatbot UI with RAG capabilities."
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
    parser.add_argument("--port", type=int, default=5001, help="Port (default: 5001).")
    parser.add_argument(
        "--chat-log",
        help="Path to save chat history JSON for analysis (optional).",
    )
    args = parser.parse_args()

    chat_log_path = Path(args.chat_log) if args.chat_log else None
    app = create_app(Path(args.json), Path(args.vectors), args.model_path, args.ollama_model, chat_log_path)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
