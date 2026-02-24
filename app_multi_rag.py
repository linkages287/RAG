#!/usr/bin/env python3
"""
Multi-source RAG Chatbot with streaming responses.
Supports multiple npz/json pairs and combines best results from each source.
Optional Weaviate mode: select collections from the web interface.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import os
import torch
from flask import Flask, Response, jsonify, render_template, request, session, stream_with_context
from transformers import AutoModel, AutoTokenizer

from ollama_api import call_ollama_api, stream_ollama_api

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    weaviate = None
    WEAVIATE_AVAILABLE = False


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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a matrix of vectors and a single vector."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return np.dot(a_norm, b_norm)


def stream_ollama(prompt: str, model: str, base_url: str):
    """Stream LLM response from Ollama using API."""
    for chunk in stream_ollama_api(prompt=prompt, model=model, base_url=base_url):
        yield chunk


def classify_query_agent(query: str, ollama_model: str, base_url: str) -> str:
    """Classify whether query needs RAG context or general knowledge."""
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
        response = call_ollama_api(prompt=classification_prompt, model=ollama_model, base_url=base_url, stream=False)
        result = response.strip().lower()
        if "rag" in result or "context" in result or "document" in result:
            return "rag"
        elif "general" in result:
            return "general"
        else:
            return "rag"
    except Exception:
        return "rag"


def extract_country(source_pdf: str | None) -> str:
    """Extract country code from PDF filename."""
    if not source_pdf:
        return "unknown"
    match = re.search(r"NU_JWC_(?:ETI_)?([A-Z]+)", str(source_pdf).upper())
    if match:
        return match.group(1)
    return Path(source_pdf).stem if source_pdf else "unknown"


def list_weaviate_collections(weaviate_url: str) -> List[Dict]:
    """List Weaviate collection names and object counts. Returns [] if Weaviate unavailable or connection fails."""
    if not WEAVIATE_AVAILABLE or weaviate is None:
        return []
    try:
        if weaviate_url in ("http://localhost:8080", "http://127.0.0.1:8080", ""):
            client = weaviate.connect_to_local()
        else:
            url_clean = weaviate_url.replace("http://", "").replace("https://", "")
            parts = url_clean.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 8080
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=weaviate_url.startswith("https://"),
            )
        try:
            if not client.is_ready():
                return []
            raw = client.collections.list_all()
            names = list(raw.keys()) if isinstance(raw, dict) else list(raw) if hasattr(raw, "__iter__") else []
            result = []
            for name in names:
                try:
                    coll = client.collections.get(name)
                    agg = coll.aggregate.over_all(total_count=True)
                    count = getattr(agg, "total_count", "?")
                except Exception:
                    count = "?"
                result.append({"name": name, "count": count})
            return result
        finally:
            if hasattr(client, "close"):
                client.close()
    except Exception:
        return []


def get_message_size(message: Dict) -> int:
    """Calculate the size of a message in bytes (approximate)."""
    content = message.get("content", "")
    # Approximate: each character is roughly 1 byte, plus overhead for structure
    return len(content.encode("utf-8")) + 100  # 100 bytes overhead for metadata


def trim_messages_to_context_window(
    messages: List[Dict],
    max_size_kb: int = 50,
    new_message_size: int = 0,
) -> List[Dict]:
    """
    Trim messages to fit within context window, keeping the most recent messages.
    
    Args:
        messages: List of message dictionaries
        max_size_kb: Maximum context window size in KB
        new_message_size: Size of the new message being added (in bytes)
        
    Returns:
        Trimmed list of messages that fits within the context window
    """
    max_size_bytes = max_size_kb * 1024
    current_size = new_message_size
    
    # Start from the end and work backwards, keeping messages that fit
    trimmed = []
    for message in reversed(messages):
        msg_size = get_message_size(message)
        if current_size + msg_size <= max_size_bytes:
            trimmed.insert(0, message)  # Insert at beginning to maintain order
            current_size += msg_size
        else:
            # Can't fit this message, stop
            break
    
    return trimmed


def format_chat_history(messages: List[Dict]) -> str:
    """Format chat history for inclusion in prompt."""
    if not messages:
        return ""
    
    history_lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            history_lines.append(f"User: {content}")
        elif role == "assistant":
            history_lines.append(f"Assistant: {content}")
    
    return "\n".join(history_lines)


class MultiSourceRAG:
    """Manages multiple vector sources and performs multi-source search."""
    
    def __init__(self, sources: List[Dict], model_path: str):
        """
        Initialize with multiple sources.
        
        Args:
            sources: List of dicts with 'json_path', 'npz_path', and 'name' keys
            model_path: Path to embedding model
        """
        self.sources = []
        self.model_path = model_path
        
        for source in sources:
            json_path = Path(source["json_path"])
            npz_path = Path(source["npz_path"])
            
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            if not npz_path.exists():
                raise FileNotFoundError(f"NPZ file not found: {npz_path}")
            
            # Load chunks
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            chunks = payload.get("chunks", [])
            
            # Load vectors
            vectors = np.load(npz_path, allow_pickle=False)["vectors"]
            
            if len(chunks) != vectors.shape[0]:
                raise ValueError(
                    f"Mismatch in {source['name']}: {len(chunks)} chunks but {vectors.shape[0]} vectors"
                )
            
            self.sources.append({
                "name": source["name"],
                "chunks": chunks,
                "vectors": vectors,
                "json_path": json_path,
                "npz_path": npz_path,
            })
        
        print(f"Loaded {len(self.sources)} sources:")
        for source in self.sources:
            print(f"  - {source['name']}: {len(source['chunks'])} chunks")
    
    def search_all_sources(
        self,
        query: str,
        top_k_per_source: int = 5,
        max_total: int = 15,
        score_threshold: float = 0.55,
    ) -> List[Dict]:
        """
        Search across all sources and return best results.
        
        Returns:
            List of results with source name, chunk, score, and metadata
        """
        query_vec = embed_query(query, self.model_path)
        all_results = []
        
        # Search each source
        for source in self.sources:
            scores = cosine_sim(source["vectors"], query_vec)
            sorted_idx = np.argsort(scores)[::-1]
            
            # Get top results from this source
            best_score = float(scores[sorted_idx[0]])
            threshold = max(score_threshold, best_score * 0.91)
            eligible_idx = [i for i in sorted_idx if float(scores[i]) >= threshold]
            context_idx = eligible_idx[:top_k_per_source]
            
            for idx in context_idx:
                chunk = source["chunks"][int(idx)]
                score = float(scores[idx])
                
                all_results.append({
                    "source": source["name"],
                    "score": score,
                    "chunk": chunk,
                    "country": extract_country(chunk.get("source_pdf")),
                    "chunk_id": chunk.get("chunk_id"),
                })
        
        # Sort all results by score (descending)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        return all_results[:max_total]
    
    def get_best_from_each_source(
        self,
        query: str,
        top_k_per_source: int = 3,
    ) -> Dict[str, List[Dict]]:
        """Get best results from each source separately."""
        query_vec = embed_query(query, self.model_path)
        results_by_source = {}
        
        for source in self.sources:
            scores = cosine_sim(source["vectors"], query_vec)
            sorted_idx = np.argsort(scores)[::-1]
            
            source_results = []
            for idx in sorted_idx[:top_k_per_source]:
                chunk = source["chunks"][int(idx)]
                source_results.append({
                    "score": float(scores[idx]),
                    "chunk": chunk,
                    "country": extract_country(chunk.get("source_pdf")),
                    "chunk_id": chunk.get("chunk_id"),
                })
            
            results_by_source[source["name"]] = source_results
        
        return results_by_source


def create_app(
    sources_config: List[Dict],
    model_path: str,
    ollama_model: str,
    chat_log_path: Optional[Path] = None,
    weaviate_url: Optional[str] = None,
    max_chunks: int = 12,
    max_score_diff_pct: float = 10.0,
) -> Flask:
    """Create Flask app with multi-source RAG and streaming. Optional Weaviate mode when weaviate_url is set."""
    app = Flask(__name__)
    app.secret_key = "multi-rag-chatbot-secret-key"
    app.config["RAG_MAX_CHUNKS"] = max_chunks
    app.config["RAG_MAX_SCORE_DIFF_PCT"] = max_score_diff_pct

    # Initialize multi-source RAG from npz/json when sources are provided
    rag = MultiSourceRAG(sources_config, model_path) if sources_config else None

    if weaviate_url and WEAVIATE_AVAILABLE:
        from app_weaviate_rag import WeaviateMultiSourceRAG as _WeaviateMultiSourceRAG
    else:
        _WeaviateMultiSourceRAG = None

    @app.route("/", methods=["GET", "POST"])
    def index():
        query = request.form.get("query", "").strip() if request.method == "POST" else ""
        clear_history = request.form.get("clear", "") == "clear"
        
        # Initialize or clear chat history
        if "messages" not in session or clear_history:
            session["messages"] = []
        
        messages = session.get("messages", [])
        sources = [s["name"] for s in rag.sources] if rag else []
        available_collections = list_weaviate_collections(weaviate_url) if weaviate_url else []
        selected_collections = session.get("weaviate_collections", [])
        
        return render_template(
            "multi_chat.html",
            messages=messages,
            sources=sources,
            weaviate_mode=bool(weaviate_url and WEAVIATE_AVAILABLE),
            weaviate_url=weaviate_url or "",
            available_collections=available_collections,
            selected_collections=selected_collections,
        )
    
    @app.route("/api/chat", methods=["POST"])
    def chat():
        """API endpoint for chat with streaming."""
        data = request.json
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Initialize session messages if needed
        if "messages" not in session:
            session["messages"] = []
        
        messages = session.get("messages", [])
        
        # Classify query
        ollama_url = os.getenv("OLLAMA_URL", "http://0.0.0.0:11434")
        query_type = classify_query_agent(query, ollama_model, ollama_url)
        print(f"\n[Query Classification] Type: {query_type}\n")
        
        # Prepare new user message
        new_user_message = {
            "role": "user",
            "content": query,
            "query_type": query_type,
        }
        new_message_size = get_message_size(new_user_message)
        
        # Trim messages to fit within context window (50KB)
        # Reserve space for RAG context and prompt overhead (~10KB)
        original_count = len(messages)
        messages = trim_messages_to_context_window(
            messages,
            max_size_kb=40,  # Reserve 10KB for RAG context and prompt
            new_message_size=new_message_size,
        )
        if len(messages) < original_count:
            print(f"[Context Window] Trimmed {original_count - len(messages)} old messages to fit context window")
        
        # Add user message
        messages.append(new_user_message)
        
        # Get chat history (excluding the current user message)
        chat_history = messages[:-1] if messages else []
        history_text = format_chat_history(chat_history)
        
        # Get search results
        search_results = []
        llm_answer = ""
        
        if query_type == "general":
            # General knowledge - no RAG context, but include chat history
            if history_text:
                prompt = (
                    "You are a NATO analyst in a conversation. "
                    "Answer the following question using your general knowledge and the conversation history below. "
                    "Provide a detailed and accurate response that is consistent with the ongoing discussion.\n\n"
                    f"Conversation History:\n{history_text}\n\n"
                    f"Current Question: {query}\n\n"
                    "Your response:"
                )
            else:
                prompt = (
                    "You are a NATO analyst. "
                    "Answer the following question using your general knowledge. "
                    "Provide a detailed and accurate response.\n\n"
                    f"Question: {query}\n"
                )
            print("\n[General Knowledge Mode] Answering without RAG context\n")
        else:
            # RAG mode - get best results from all sources (npz/json or Weaviate)
            no_collections_prompt = None
            if weaviate_url and _WeaviateMultiSourceRAG and not session.get("weaviate_collections"):
                no_collections_prompt = (
                    "You are a NATO analyst. No Weaviate collections are selected. "
                    "Please select at least one collection in the interface above, then ask your question again. "
                    "For now, answer briefly using general knowledge.\n\n"
                    f"Question: {query}\n\nYour response:"
                )
                search_results = []
                print("\n[Weaviate] No collections selected; prompting user to select collections\n")
            elif weaviate_url and _WeaviateMultiSourceRAG and session.get("weaviate_collections"):
                max_chunks_cfg = app.config.get("RAG_MAX_CHUNKS", 12)
                weaviate_rag = _WeaviateMultiSourceRAG(
                    session["weaviate_collections"],
                    model_path,
                    weaviate_url,
                )
                search_results = weaviate_rag.search_all_sources(
                    query,
                    top_k_per_source=10,
                    max_total=max(50, max_chunks_cfg * 3),
                    score_threshold=0.3,
                )
            elif rag:
                max_chunks_cfg = app.config.get("RAG_MAX_CHUNKS", 12)
                search_results = rag.search_all_sources(
                    query,
                    top_k_per_source=10,
                    max_total=max(50, max_chunks_cfg * 3),
                    score_threshold=0.3,
                )
            else:
                search_results = []

            # Apply max chunks and max score-diff percentage (only keep chunks within X% of best score)
            if search_results and not no_collections_prompt:
                max_chunks_cfg = app.config.get("RAG_MAX_CHUNKS", 12)
                max_score_diff_pct = app.config.get("RAG_MAX_SCORE_DIFF_PCT", 10.0)
                best_score_val = search_results[0]["score"]
                min_score = best_score_val * (1.0 - max_score_diff_pct / 100.0)
                search_results = [r for r in search_results if r["score"] >= min_score][:max_chunks_cfg]

            if no_collections_prompt:
                prompt = no_collections_prompt
                query_type = "general"
            else:
                best_score = search_results[0]["score"] if search_results else 0.0
                if best_score < 0.6:
                    query_type = "general"
                    if history_text:
                        prompt = (
                            "You are a NATO analyst in a conversation. "
                            "Answer the following question using your general knowledge and the conversation history below. "
                            "Provide a detailed and accurate response that is consistent with the ongoing discussion.\n\n"
                            f"Conversation History:\n{history_text}\n\n"
                            f"Current Question: {query}\n\n"
                            "Your response:"
                        )
                    else:
                        prompt = (
                            "You are a NATO analyst. "
                            "Answer the following question using your general knowledge. "
                            "Provide a detailed and accurate response.\n\n"
                            f"Question: {query}\n"
                        )
                    print(f"\n[Score Check] Best score {best_score:.3f} < 0.6, using general knowledge mode\n")
                else:
                    # Build context from search results (without country prefix in LLM prompt)
                    top_context = "\n\n".join(
                        f"{r['chunk']['text']}"
                        for r in search_results
                    )
                    
                    if history_text:
                        prompt = (
                            "You are a NATO analyst in a conversation. "
                            "Answer the user question analyzing only the context below, and consider the conversation history. "
                            "If the context is insufficient, say so explicitly. "
                            "Be consistent with the ongoing discussion.\n\n"
                            f"Conversation History:\n{history_text}\n\n"
                            f"Current User Question: {query}\n\n"
                            f"Relevant Context from Documents:\n{top_context}\n\n"
                            "Your response:"
                        )
                    else:
                        prompt = (
                            "You are a NATO analyst. "
                            "Answer the user question analyzing only the context below. "
                            "If the context is insufficient, say so explicitly.\n\n"
                            f"User question: {query}\n\n"
                            f"Context:\n{top_context}\n"
                        )
                    print("\n--- Injected RAG Prompt ---\n")
                    print(prompt)
        
        # Stream LLM response
        def generate():
            nonlocal messages  # Allow modification of outer scope variable
            full_response = ""
            try:
                ollama_url = os.getenv("OLLAMA_URL", "http://0.0.0.0:11434")
                for chunk in stream_ollama(prompt, ollama_model, ollama_url):
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                
                # Save assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": full_response,
                    "query_type": query_type,
                    "sources": [r["source"] for r in search_results],
                    "top_results": [
                        {
                            "source": r["source"],
                            "country": r["country"],
                            "score": r["score"],
                        }
                        for r in search_results[:5]
                    ],
                }
                messages.append(assistant_msg)
                
                # Trim messages again after adding assistant response to stay within context window
                original_count = len(messages)
                messages = trim_messages_to_context_window(
                    messages,
                    max_size_kb=50,  # Full 50KB now since we're done with this query
                    new_message_size=0,  # No new message being added
                )
                if len(messages) < original_count:
                    print(f"[Context Window] Trimmed {original_count - len(messages)} old messages after assistant response")
                
                session["messages"] = messages
                
                # Final message with metadata
                yield f"data: {json.dumps({
                    'chunk': '',
                    'done': True,
                    'metadata': {
                        'query_type': query_type,
                        'sources': assistant_msg['sources'],
                        'top_results': assistant_msg['top_results'],
                    }
                })}\n\n"
                
                # Log to file if enabled
                if chat_log_path:
                    try:
                        if chat_log_path.exists():
                            with chat_log_path.open("r", encoding="utf-8") as f:
                                all_messages = json.load(f)
                        else:
                            all_messages = []
                        all_messages.extend(messages[-2:])
                        with chat_log_path.open("w", encoding="utf-8") as f:
                            json.dump(all_messages, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"Warning: Could not save chat log: {e}")
                        
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    
    @app.route("/api/sources", methods=["GET"])
    def get_sources():
        """Get information about available sources (npz/json or selected Weaviate collections)."""
        if weaviate_url and session.get("weaviate_collections"):
            return jsonify({
                "sources": [
                    {"name": c["source_name"], "chunk_count": "—"}
                    for c in session["weaviate_collections"]
                ]
            })
        if rag:
            return jsonify({
                "sources": [
                    {"name": s["name"], "chunk_count": len(s["chunks"])}
                    for s in rag.sources
                ]
            })
        return jsonify({"sources": []})

    @app.route("/api/weaviate/collections", methods=["GET"])
    def api_weaviate_collections():
        """List available Weaviate collections (only in Weaviate mode)."""
        if not weaviate_url or not WEAVIATE_AVAILABLE:
            return jsonify({"collections": []})
        collections = list_weaviate_collections(weaviate_url)
        return jsonify({"collections": collections})

    @app.route("/api/weaviate/select", methods=["POST"])
    def api_weaviate_select():
        """Store selected Weaviate collection names in session. Expects JSON: { \"collections\": [\"Name1\", \"Name2\"] }."""
        if not weaviate_url or not WEAVIATE_AVAILABLE:
            return jsonify({"success": False, "message": "Weaviate mode not enabled"}), 400
        data = request.get_json(silent=True) or {}
        names = data.get("collections")
        if not isinstance(names, list):
            return jsonify({"success": False, "message": "Expected JSON body with 'collections' array"}), 400
        session["weaviate_collections"] = [
            {"source_name": n, "collection_name": n}
            for n in names
            if isinstance(n, str) and n.strip()
        ]
        return jsonify({"success": True, "collections": session["weaviate_collections"]})
    
    @app.route("/api/clear", methods=["POST"])
    def clear_chat():
        """Clear all chat history and start a new conversation."""
        session["messages"] = []
        return jsonify({"success": True, "message": "Chat history cleared"})
    
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-source RAG chatbot with streaming responses. Use --weaviate-url to select Weaviate collections from the web interface."
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[],
        help="Source configurations in format 'name:json_path:npz_path' (optional when --weaviate-url is set).",
    )
    parser.add_argument(
        "--weaviate-url",
        default=None,
        help="Weaviate server URL (e.g. http://localhost:8080). When set, collections can be selected in the web UI.",
    )
    parser.add_argument(
        "--model-path",
        default="/home/linkages/cursor/pdftext/models/mxbai-embed-large-v1",
        help="Path to embedding model.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model name (default: llama3.2).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=12,
        help="Max number of chunks to pass to the LLM (default: 12).",
    )
    parser.add_argument(
        "--max-score-diff-pct",
        type=float,
        default=10.0,
        help="Max percentage difference from the best score; only chunks within this %% of the best score are used (default: 10.0).",
    )
    parser.add_argument(
        "--chat-log",
        help="Path to save chat history JSON for analysis.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5002,
        help="Port (default: 5002).",
    )
    args = parser.parse_args()

    weaviate_url = args.weaviate_url or os.environ.get("WEAVIATE_URL")
    if not args.sources and not weaviate_url:
        parser.error("Provide either --sources (npz/json) or --weaviate-url to use Weaviate collections from the UI.")

    # Parse sources (used only when not relying on Weaviate UI)
    sources_config = []
    for source_str in args.sources:
        parts = source_str.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid source format: {source_str}. Expected 'name:json_path:npz_path'"
            )
        name, json_path, npz_path = parts
        sources_config.append({
            "name": name,
            "json_path": json_path,
            "npz_path": npz_path,
        })

    chat_log_path = Path(args.chat_log) if args.chat_log else None

    app = create_app(
        sources_config,
        args.model_path,
        args.ollama_model,
        chat_log_path,
        weaviate_url=weaviate_url,
        max_chunks=args.max_chunks,
        max_score_diff_pct=args.max_score_diff_pct,
    )
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
