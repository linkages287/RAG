#!/usr/bin/env python3
"""
Multi-source RAG Chatbot with Weaviate and **graph expansion** (post-retrieval).

Same interface as app_weaviate_rag.py, but after vector search we expand context
using the knowledge graph: for each top result we fetch related chunks via the graph
and merge them into the context sent to the LLM.

Requires a pre-built knowledge graph JSON (from knowledge_graph_builder.py).
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import weaviate
except ImportError:
    weaviate = None

from app_weaviate_rag import (
    WeaviateMultiSourceRAG,
    stream_ollama,
    classify_query_agent,
    extract_country,
    get_message_size,
    trim_messages_to_context_window,
    format_chat_history,
)
from knowledge_graph_builder import load_graph_from_json, get_related_chunks

# Re-export for template/routes
from flask import Flask, Response, jsonify, render_template, request, session, stream_with_context


def connect_weaviate_client(weaviate_url: str):
    """Return a Weaviate client (same logic as connect_weaviate.py). Caller must close if needed."""
    if weaviate_url in ["http://localhost:8080", "http://127.0.0.1:8080"]:
        return weaviate.connect_to_local()
    url_clean = weaviate_url.replace("http://", "").replace("https://", "")
    parts = url_clean.split(":", 1)
    host = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 8080
    http_secure = weaviate_url.startswith("https://")
    return weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=http_secure,
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=http_secure,
    )


def fetch_chunk_by_node_id(
    client: weaviate.WeaviateClient,
    node_id: str,
    collection_names: List[str],
) -> Optional[Dict]:
    """
    Fetch chunk text and metadata from Weaviate by node_id (format 'collection_name:uuid').
    Returns dict with keys: text, source_pdf, country, source, collection_name, uuid, chunk_id, etc.
    """
    if ":" not in node_id:
        return None
    collection_name, uuid_str = node_id.split(":", 1)
    if collection_name not in collection_names:
        return None
    try:
        collection = client.collections.get(collection_name)
        obj = collection.data.get_by_id(uuid_str)
        if obj is None or not obj.properties:
            return None
        props = dict(obj.properties)
        source_name = next(
            (c["source_name"] for c in [] if c.get("collection_name") == collection_name),
            collection_name,
        )
        return {
            "text": props.get("text", ""),
            "source_pdf": props.get("source_pdf", "unknown"),
            "country": props.get("country", extract_country(props.get("source_pdf"))),
            "source": source_name,
            "collection_name": collection_name,
            "uuid": uuid_str,
            "chunk_id": props.get("chunk_id"),
            "page_start": props.get("page_start", 0),
            "page_end": props.get("page_end", 0),
        }
    except Exception:
        return None


def expand_results_with_graph(
    search_results: List[Dict],
    G,
    weaviate_url: str,
    collections_config: List[Dict],
    top_n: int = 5,
    expand_k: int = 3,
) -> List[Dict]:
    """
    Expand search results with graph-related chunks. Returns merged list of result dicts
    (each with 'chunk' dict containing 'text', and 'source', 'country', etc.).
    """
    if not search_results or G is None:
        return search_results
    collection_names = [c["collection_name"] for c in collections_config]
    source_by_collection = {c["collection_name"]: c["source_name"] for c in collections_config}
    seen_node_id: Set[str] = set()
    for r in search_results:
        nid = f"{r.get('collection_name', '')}:{r.get('uuid', '')}"
        if nid:
            seen_node_id.add(nid)

    merged: List[Dict] = list(search_results)
    expanded_chunks: List[Dict] = []

    try:
        client = connect_weaviate_client(weaviate_url)
        if not client.is_ready():
            return search_results
        for r in search_results[:top_n]:
            node_id = f"{r.get('collection_name', '')}:{r.get('uuid', '')}"
            if not node_id or node_id.endswith(":"):
                continue
            related = get_related_chunks(G, node_id, k=expand_k, same_collection=False)
            for (nid, _weight) in related:
                if nid in seen_node_id:
                    continue
                chunk_dict = fetch_chunk_by_node_id(client, nid, collection_names)
                if chunk_dict is None or not chunk_dict.get("text"):
                    continue
                seen_node_id.add(nid)
                chunk_dict["source"] = source_by_collection.get(chunk_dict["collection_name"], chunk_dict["collection_name"])
                expanded_chunks.append({
                    "source": chunk_dict["source"],
                    "collection_name": chunk_dict["collection_name"],
                    "uuid": uid,
                    "score": 0.0,
                    "chunk": {
                        "text": chunk_dict["text"],
                        "source_pdf": chunk_dict["source_pdf"],
                        "page_start": chunk_dict.get("page_start", 0),
                        "page_end": chunk_dict.get("page_end", 0),
                        "chunk_id": chunk_dict.get("chunk_id"),
                        "token_count": 0,
                    },
                    "country": chunk_dict["country"],
                    "chunk_id": chunk_dict.get("chunk_id"),
                    "distance": 0.0,
                    "from_graph": True,
                })
        if hasattr(client, "__exit__"):
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass
    except Exception as e:
        print(f"[Graph expansion] Error: {e}")
        return search_results

    merged = search_results + expanded_chunks
    return merged


def create_app(
    collections_config: List[Dict],
    model_path: str,
    ollama_model: str,
    weaviate_url: str = "http://localhost:8080",
    chat_log_path: Optional[Path] = None,
    graph_json_path: Optional[Path] = None,
    graph_expand_top_n: int = 5,
    graph_expand_k: int = 3,
) -> Flask:
    """Create Flask app with Weaviate RAG + graph expansion."""
    app = Flask(__name__)
    app.secret_key = "weaviate-rag-graph-chatbot-secret-key"

    rag = WeaviateMultiSourceRAG(collections_config, model_path, weaviate_url)
    knowledge_graph = None
    if graph_json_path and graph_json_path.exists():
        try:
            knowledge_graph = load_graph_from_json(graph_json_path)
            print(f"Loaded knowledge graph: {knowledge_graph.number_of_nodes()} nodes, {knowledge_graph.number_of_edges()} edges")
        except Exception as e:
            print(f"Warning: Could not load graph from {graph_json_path}: {e}")

    @app.route("/", methods=["GET", "POST"])
    def index():
        query = request.form.get("query", "").strip() if request.method == "POST" else ""
        clear_history = request.form.get("clear", "") == "clear"
        if "messages" not in session or clear_history:
            session["messages"] = []
        messages = session.get("messages", [])
        return render_template(
            "multi_chat.html",
            messages=messages,
            sources=[c["source_name"] for c in rag.collections],
        )

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.json
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400
        if "messages" not in session:
            session["messages"] = []
        messages = session.get("messages", [])
        ollama_url = os.getenv("OLLAMA_URL", "http://0.0.0.0:11434")
        query_type = classify_query_agent(query, ollama_model, ollama_url)
        new_user_message = {"role": "user", "content": query, "query_type": query_type}
        new_message_size = get_message_size(new_user_message)
        messages = trim_messages_to_context_window(messages, max_size_kb=40, new_message_size=new_message_size)
        messages.append(new_user_message)
        chat_history = messages[:-1] if messages else []
        history_text = format_chat_history(chat_history)
        search_results = []

        if query_type == "general":
            if history_text:
                prompt = (
                    "You are a NATO analyst in a conversation. "
                    "Answer the following question using your general knowledge and the conversation history below. "
                    "Provide a detailed and accurate response that is consistent with the ongoing discussion.\n\n"
                    "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
                    "1. ALWAYS format lists using markdown syntax (- for unordered, 1. for ordered)\n"
                    "2. ALWAYS use tables (| col1 | col2 |) when presenting structured/comparative data\n"
                    "3. ALWAYS use **bold** for key terms, important points, and emphasis\n"
                    "4. ALWAYS use code blocks (```code```) for technical terms, codes, or identifiers\n"
                    "5. Use headers (# ## ###) to organize sections when appropriate\n"
                    "6. Use blockquotes (>) for citations or important notes\n\n"
                    "Your response MUST be formatted in Markdown. Do not use plain text formatting.\n\n"
                    f"Conversation History:\n{history_text}\n\nCurrent Question: {query}\n\nYour response (MUST be in Markdown format):"
                )
            else:
                prompt = (
                    "You are a NATO analyst. Answer the following question using your general knowledge. "
                    "Provide a detailed and accurate response.\n\n"
                    "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
                    "1. ALWAYS format lists using markdown syntax\n"
                    "2. ALWAYS use tables when presenting structured data\n"
                    "3. ALWAYS use **bold** for key terms\n\n"
                    f"Question: {query}\n\nYour response (MUST be in Markdown format):"
                )
        else:
            search_results = rag.search_all_sources(
                query,
                top_k_per_source=5,
                max_total=12,
                score_threshold=0.6,
            )
            # Graph expansion: add related chunks from knowledge graph
            search_results = expand_results_with_graph(
                search_results,
                knowledge_graph,
                weaviate_url,
                collections_config,
                top_n=graph_expand_top_n,
                expand_k=graph_expand_k,
            )
            best_score = search_results[0]["score"] if search_results else 0.0
            if best_score < 0.6:
                query_type = "general"
                prompt = (
                    "You are a NATO analyst. Answer using your general knowledge.\n\n"
                    f"Conversation History:\n{history_text}\n\n" if history_text else ""
                    f"Question: {query}\n\nYour response:"
                )
            else:
                top_context = "\n\n".join(r["chunk"]["text"] for r in search_results)
                top_3_summary = ""
                if len(search_results) >= 3:
                    top_3_summary = (
                        "\nTop 3 Best Matching Results (by relevance score):\n"
                        + "\n".join(
                            f"  {i+1}. Source: {r['source']}, Country: {r['country']}, Score: {r['score']:.3f}"
                            + (" (graph-expanded)" if r.get("from_graph") else "")
                            for i, r in enumerate(search_results[:3])
                        )
                        + "\n"
                    )
                elif search_results:
                    top_3_summary = (
                        "\nBest Matching Results:\n"
                        + "\n".join(
                            f"  {i+1}. Source: {r['source']}, Country: {r['country']}, Score: {r['score']:.3f}"
                            + (" (graph-expanded)" if r.get("from_graph") else "")
                            for i, r in enumerate(search_results)
                        )
                        + "\n"
                    )
                if history_text:
                    prompt = (
                        "You are a NATO analyst in a conversation. "
                        "Answer the user question analyzing only the context below, and consider the conversation history. "
                        "If the context is insufficient, say so explicitly. Be consistent with the ongoing discussion.\n\n"
                        "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
                        "1. ALWAYS format lists using markdown syntax\n"
                        "2. ALWAYS use tables when presenting structured data\n"
                        "3. ALWAYS use **bold** for key terms\n\n"
                        f"Conversation History:\n{history_text}\n\nCurrent User Question: {query}\n\n"
                        f"{top_3_summary}Relevant Context from Documents (including graph-related chunks):\n{top_context}\n\n"
                        "Your response (MUST be in Markdown format):"
                    )
                else:
                    prompt = (
                        "You are a NATO analyst. Answer the user question analyzing only the context below. "
                        "If the context is insufficient, say so explicitly.\n\n"
                        "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
                        "1. ALWAYS format lists using markdown syntax\n"
                        "2. ALWAYS use tables when presenting structured data\n"
                        "3. ALWAYS use **bold** for key terms\n\n"
                        f"User question: {query}\n\n{top_3_summary}Context:\n{top_context}\n\n"
                        "Your response (MUST be in Markdown format):"
                    )

        def generate():
            nonlocal messages
            full_response = ""
            try:
                for chunk in stream_ollama(prompt, ollama_model, ollama_url):
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                top_3_scores = [
                    {"source": r["source"], "country": r["country"], "score": r["score"], "chunk_id": r.get("chunk_id")}
                    for r in search_results[:3]
                ] if search_results else []
                assistant_msg = {
                    "role": "assistant",
                    "content": full_response,
                    "query_type": query_type,
                    "sources": [r["source"] for r in search_results],
                    "top_3_scores": top_3_scores,
                    "top_results": top_3_scores,
                }
                messages.append(assistant_msg)
                messages = trim_messages_to_context_window(messages, max_size_kb=50, new_message_size=0)
                session["messages"] = messages
                yield f"data: {json.dumps({
                    'chunk': '', 'done': True,
                    'metadata': {
                        'query_type': query_type,
                        'sources': assistant_msg['sources'],
                        'top_3_scores': assistant_msg['top_3_scores'],
                        'top_results': assistant_msg['top_results'],
                    }
                })}\n\n"
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
                    except Exception:
                        pass
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/sources", methods=["GET"])
    def get_sources():
        return jsonify({
            "sources": [{"name": c["source_name"], "collection": c["collection_name"]} for c in rag.collections]
        })

    @app.route("/api/clear", methods=["POST"])
    def clear_chat():
        session["messages"] = []
        return jsonify({"success": True, "message": "Chat history cleared"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-source RAG chatbot with Weaviate and knowledge-graph expansion (post-retrieval)."
    )
    parser.add_argument("--collections", required=True, nargs="+", help="Collection configs 'source_name:collection_name'")
    parser.add_argument("--model-path", default="/home/linkages/cursor/pdftext/models/mxbai-embed-large-v1", help="Path to embedding model")
    parser.add_argument("--ollama-model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate server URL")
    parser.add_argument("--graph-json", type=Path, help="Path to knowledge graph JSON (from knowledge_graph_builder.py)")
    parser.add_argument("--graph-expand-top-n", type=int, default=5, help="Number of top results to expand with graph (default 5)")
    parser.add_argument("--graph-expand-k", type=int, default=3, help="Number of related chunks per result from graph (default 3)")
    parser.add_argument("--chat-log", type=Path, help="Path to save chat history JSON")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=5006, help="Port (default 5006)")
    args = parser.parse_args()

    collections_config = []
    for s in args.collections:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid collection format: {s}. Expected 'source_name:collection_name'")
        collections_config.append({"source_name": parts[0], "collection_name": parts[1]})

    app = create_app(
        collections_config,
        args.model_path,
        args.ollama_model,
        args.weaviate_url,
        Path(args.chat_log) if args.chat_log else None,
        graph_json_path=args.graph_json,
        graph_expand_top_n=args.graph_expand_top_n,
        graph_expand_k=args.graph_expand_k,
    )
    print(f"RAG + Graph expansion app on http://{args.host}:{args.port}")
    if args.graph_json:
        print(f"  Graph: {args.graph_json} (expand top {args.graph_expand_top_n} x {args.graph_expand_k} related)")
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
