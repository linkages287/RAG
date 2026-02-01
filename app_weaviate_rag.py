#!/usr/bin/env python3
"""
Multi-source RAG Chatbot with Weaviate vector database and streaming responses.
Retrieves vectors from Weaviate server instead of npz files.
"""
import argparse
import json
import re
import socket
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import os
import torch
import weaviate
from flask import Flask, Response, jsonify, render_template, request, session, stream_with_context
from transformers import AutoModel, AutoTokenizer

from ollama_api import call_ollama_api, stream_ollama_api


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


def get_message_size(message: Dict) -> int:
    """Calculate the size of a message in bytes (approximate)."""
    content = message.get("content", "")
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


class WeaviateMultiSourceRAG:
    """Manages multiple Weaviate collections and performs multi-source search."""
    
    def __init__(self, collections: List[Dict], model_path: str, weaviate_url: str = "http://localhost:8080"):
        """
        Initialize with multiple Weaviate collections.
        
        Args:
            collections: List of dicts with 'collection_name' and 'source_name' keys
            model_path: Path to embedding model
            weaviate_url: Weaviate server URL
        """
        self.collections = collections
        self.model_path = model_path
        self.weaviate_url = weaviate_url
        
        # Test connection
        try:
            # Use local connection for localhost/127.0.0.1
            if weaviate_url in ["http://localhost:8080", "http://127.0.0.1:8080"]:
                client = weaviate.connect_to_local()
            else:
                # For remote connections, parse URL and use connect_to_custom()
                # Parse URL
                url_clean = weaviate_url.replace("http://", "").replace("https://", "")
                if ":" in url_clean:
                    host, port_str = url_clean.split(":", 1)
                    port = int(port_str)
                else:
                    host = url_clean
                    port = 8080
                
                http_secure = weaviate_url.startswith("https://")
                
                # Default GRPC port is 50051
                grpc_port = 50051
                
                client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=port,
                    http_secure=http_secure,
                    grpc_host=host,
                    grpc_port=grpc_port,
                    grpc_secure=http_secure,
                )
            
            if not client.is_ready():
                raise ConnectionError("Weaviate is not ready")
            
            # Verify collections exist
            for collection_config in collections:
                collection_name = collection_config["collection_name"]
                if not client.collections.exists(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist in Weaviate")
            
            print(f"✓ Connected to Weaviate at {weaviate_url}")
            print(f"✓ Loaded {len(collections)} collections:")
            for collection_config in collections:
                collection_name = collection_config["collection_name"]
                print(f"  - {collection_config['source_name']}: Collection '{collection_name}'")
            
            # Close connection if it's a context manager
            if hasattr(client, '__exit__'):
                client.__exit__(None, None, None)
        except Exception as e:
            # Provide more helpful error message
            error_msg = str(e)
            if "host.docker.internal" in weaviate_url:
                print(f"\n⚠ Warning: Could not verify Weaviate connection: {error_msg}")
                print(f"   Attempted to connect to: {weaviate_url}")
                print(f"   Make sure:")
                print(f"   1. Weaviate is running on the host at http://127.0.0.1:8080")
                print(f"   2. Docker command includes: --add-host=host.docker.internal:host-gateway")
                print(f"   3. Weaviate is accessible from the host network")
                print(f"   The app will continue but may not be able to connect to Weaviate.\n")
            else:
                print(f"\n⚠ Warning: Could not verify Weaviate connection: {error_msg}")
                print(f"   Attempted to connect to: {weaviate_url}")
                print(f"   The app will continue but may not be able to connect to Weaviate.\n")
            # Don't raise - allow app to start and try connection later
    
    
    # Search across all Weaviate collections and return the best matching results for a given query.
    # Embeds the query, performs a similarity search in each collection, aggregates, sorts,
    # and thresholds results (based on highest score and provided score_threshold) to return at most max_total items.
    def search_all_sources(
        self,
        query: str,
        top_k_per_source: int = 5,
        max_total: int = 15,
        score_threshold: float = 0.6,
    ) -> List[Dict]:
        """
        Search across all Weaviate collections and return best results.
        
        Returns:
            List of results with source name, chunk, score, and metadata
        """
        query_vec = embed_query(query, self.model_path)
        all_results = []
        
        # Connect to Weaviate and search all collections
        # Use local connection for localhost/127.0.0.1
        if self.weaviate_url in ["http://localhost:8080", "http://127.0.0.1:8080"]:
            with weaviate.connect_to_local() as client:
                all_results = self._search_with_client(client, query_vec, top_k_per_source, score_threshold)
        else:
            # For remote connections, parse URL and use connect_to_custom()
            # Parse URL
            url_clean = self.weaviate_url.replace("http://", "").replace("https://", "")
            if ":" in url_clean:
                host, port_str = url_clean.split(":", 1)
                port = int(port_str)
            else:
                host = url_clean
                port = 8080
            
            http_secure = self.weaviate_url.startswith("https://")
            
            # Default GRPC port is 50051
            grpc_port = 50051
            
            with weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=http_secure,
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=http_secure,
            ) as client:
                all_results = self._search_with_client(client, query_vec, top_k_per_source, score_threshold)
        
        # Sort all results by score (descending)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply threshold based on best score
        if all_results:
            best_score = all_results[0]["score"]
            threshold = max(score_threshold, best_score * 0.91)
            all_results = [r for r in all_results if r["score"] >= threshold]
        
        # Return top results
        return all_results[:max_total]
    
    def _search_with_client(self, client, query_vec, top_k_per_source, score_threshold):
        """Helper method to search collections with an existing client."""
        all_results = []
        
        # Search each collection
        for collection_config in self.collections:
            collection_name = collection_config["collection_name"]
            source_name = collection_config["source_name"]
            
            collection = client.collections.get(collection_name)
            
            # Perform vector similarity search
            result = collection.query.near_vector(
                near_vector=query_vec.tolist(),
                limit=top_k_per_source * 2,  # Get more to filter by threshold
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
            )
            
            # Process results
            for obj in result.objects:
                props = obj.properties
                metadata = obj.metadata
                
                # Convert distance to similarity score (distance is typically 0-2, similarity is 1-distance)
                distance = metadata.distance if metadata else 2.0
                similarity = max(0.0, 1.0 - distance)  # Ensure non-negative
                
                # Filter by threshold
                if similarity >= score_threshold:
                    chunk_data = {
                        "chunk_id": props.get("chunk_id"),
                        "text": props.get("text", ""),
                        "source_pdf": props.get("source_pdf", "unknown"),
                        "page_start": props.get("page_start", 0),
                        "page_end": props.get("page_end", 0),
                        "token_count": props.get("token_count", 0),
                    }
                    
                    all_results.append({
                        "source": source_name,
                        "collection_name": collection_name,
                        "uuid": str(obj.uuid),
                        "score": similarity,
                        "chunk": chunk_data,
                        "country": props.get("country", extract_country(props.get("source_pdf"))),
                        "chunk_id": props.get("chunk_id"),
                        "distance": distance,
                    })
        
        return all_results


def create_app(
    collections_config: List[Dict],
    model_path: str,
    ollama_model: str,
    weaviate_url: str = "http://localhost:8080",
    chat_log_path: Optional[Path] = None,
) -> Flask:
    """Create Flask app with Weaviate-based multi-source RAG and streaming."""
    app = Flask(__name__)
    app.secret_key = "weaviate-rag-chatbot-secret-key"
    
    # Initialize Weaviate multi-source RAG
    rag = WeaviateMultiSourceRAG(collections_config, model_path, weaviate_url)
    
    @app.route("/", methods=["GET", "POST"])
    def index():
        query = request.form.get("query", "").strip() if request.method == "POST" else ""
        clear_history = request.form.get("clear", "") == "clear"
        
        # Initialize or clear chat history
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
        
        if query_type == "general":
            # General knowledge - no RAG context, but include chat history
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
                    f"Conversation History:\n{history_text}\n\n"
                    f"Current Question: {query}\n\n"
                    "Your response (MUST be in Markdown format):"
                )
            else:
                prompt = (
                    "You are a NATO analyst. "
                    "Answer the following question using your general knowledge. "
                    "Provide a detailed and accurate response.\n\n"
                    "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
                    "1. ALWAYS format lists using markdown syntax (- for unordered, 1. for ordered)\n"
                    "2. ALWAYS use tables (| col1 | col2 |) when presenting structured/comparative data\n"
                    "3. ALWAYS use **bold** for key terms, important points, and emphasis\n"
                    "4. ALWAYS use code blocks (```code```) for technical terms, codes, or identifiers\n"
                    "5. Use headers (# ## ###) to organize sections when appropriate\n"
                    "6. Use blockquotes (>) for citations or important notes\n\n"
                    "Your response MUST be formatted in Markdown. Do not use plain text formatting.\n\n"
                    f"Question: {query}\n\n"
                    "Your response (MUST be in Markdown format):"
                )
            print("\n[General Knowledge Mode] Answering without RAG context\n")
        else:
            # RAG mode - get best results from all Weaviate collections
            search_results = rag.search_all_sources(
                query,
                top_k_per_source=5,
                max_total=12,
                score_threshold=0.6,
            )
            
            best_score = search_results[0]["score"] if search_results else 0.0
            
            if best_score < 0.6:
                query_type = "general"
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
                        f"Conversation History:\n{history_text}\n\n"
                        f"Current Question: {query}\n\n"
                        "Your response (MUST be in Markdown format):"
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
                
                # Create summary of top 3 best scores
                top_3_summary = ""
                if len(search_results) >= 3:
                    top_3_summary = (
                        "\nTop 3 Best Matching Results (by relevance score):\n"
                        + "\n".join(
                            f"  {i+1}. Source: {r['source']}, Country: {r['country']}, Score: {r['score']:.3f} ({r['score']*100:.1f}% match)"
                            for i, r in enumerate(search_results[:3])
                        )
                        + "\n"
                    )
                elif search_results:
                    top_3_summary = (
                        "\nBest Matching Results (by relevance score):\n"
                        + "\n".join(
                            f"  {i+1}. Source: {r['source']}, Country: {r['country']}, Score: {r['score']:.3f} ({r['score']*100:.1f}% match)"
                            for i, r in enumerate(search_results)
                        )
                        + "\n"
                    )
                
                if history_text:
                    prompt = (
                        "You are a NATO analyst in a conversation. "
                        "Answer the user question analyzing only the context below, and consider the conversation history. "
                        "If the context is insufficient, say so explicitly. "
                        "Be consistent with the ongoing discussion.\n\n"
                        "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
                        "1. ALWAYS format lists using markdown syntax (- for unordered, 1. for ordered)\n"
                        "2. ALWAYS use tables (| col1 | col2 |) when presenting structured/comparative data\n"
                        "3. ALWAYS use **bold** for key terms, important points, and emphasis\n"
                        "4. ALWAYS use code blocks (```code```) for technical terms, codes, or identifiers\n"
                        "5. Use headers (# ## ###) to organize sections when appropriate\n"
                        "6. Use blockquotes (>) for citations or important notes\n\n"
                        "Your response MUST be formatted in Markdown. Do not use plain text formatting.\n\n"
                        f"Conversation History:\n{history_text}\n\n"
                        f"Current User Question: {query}\n\n"
                        f"{top_3_summary}"
                        f"Relevant Context from Documents:\n{top_context}\n\n"
                        "Your response (MUST be in Markdown format):"
                    )
                else:
                    prompt = (
                        "You are a NATO analyst. "
                        "Answer the user question analyzing only the context below. "
                        "If the context is insufficient, say so explicitly.\n\n"
                        "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
                        "1. ALWAYS format lists using markdown syntax (- for unordered, 1. for ordered)\n"
                        "2. ALWAYS use tables (| col1 | col2 |) when presenting structured/comparative data\n"
                        "3. ALWAYS use **bold** for key terms, important points, and emphasis\n"
                        "4. ALWAYS use code blocks (```code```) for technical terms, codes, or identifiers\n"
                        "5. Use headers (# ## ###) to organize sections when appropriate\n"
                        "6. Use blockquotes (>) for citations or important notes\n\n"
                        "Your response MUST be formatted in Markdown. Do not use plain text formatting.\n\n"
                        f"User question: {query}\n\n"
                        f"{top_3_summary}"
                        f"Context:\n{top_context}\n\n"
                        "Your response (MUST be in Markdown format):"
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
                
                # Get top 3 best scores for tracking
                top_3_scores = [
                    {
                        "source": r["source"],
                        "country": r["country"],
                        "score": r["score"],
                        "chunk_id": r.get("chunk_id"),
                    }
                    for r in search_results[:3]
                ] if search_results else []
                
                # Log top 3 scores
                if top_3_scores:
                    print("\n[Top 3 Best Scores from RAG]")
                    for i, score_info in enumerate(top_3_scores, 1):
                        print(f"  {i}. {score_info['source']} - {score_info['country']}: "
                              f"Score {score_info['score']:.3f} ({score_info['score']*100:.1f}%)")
                    print()
                
                # Save assistant message with top 3 scores
                assistant_msg = {
                    "role": "assistant",
                    "content": full_response,
                    "query_type": query_type,
                    "sources": [r["source"] for r in search_results],
                    "top_3_scores": top_3_scores,
                    "top_results": top_3_scores,  # Keep for backward compatibility
                }
                messages.append(assistant_msg)
                
                # Trim messages again after adding assistant response
                original_count = len(messages)
                messages = trim_messages_to_context_window(
                    messages,
                    max_size_kb=50,
                    new_message_size=0,
                )
                if len(messages) < original_count:
                    print(f"[Context Window] Trimmed {original_count - len(messages)} old messages after assistant response")
                
                session["messages"] = messages
                
                # Final message with metadata including top 3 scores
                yield f"data: {json.dumps({
                    'chunk': '',
                    'done': True,
                    'metadata': {
                        'query_type': query_type,
                        'sources': assistant_msg['sources'],
                        'top_3_scores': assistant_msg['top_3_scores'],
                        'top_results': assistant_msg['top_results'],  # Backward compatibility
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
        """Get information about available sources."""
        return jsonify({
            "sources": [
                {
                    "name": c["source_name"],
                    "collection": c["collection_name"],
                }
                for c in rag.collections
            ]
        })
    
    @app.route("/api/clear", methods=["POST"])
    def clear_chat():
        """Clear all chat history and start a new conversation."""
        session["messages"] = []
        return jsonify({"success": True, "message": "Chat history cleared"})
    
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-source RAG chatbot with Weaviate vector database and streaming responses."
    )
    parser.add_argument(
        "--collections",
        required=True,
        nargs="+",
        help="Collection configurations in format 'source_name:collection_name' (can specify multiple).",
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
        "--weaviate-url",
        default="http://localhost:8080",
        help="Weaviate server URL (default: http://localhost:8080).",
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
        default=5003,
        help="Port (default: 5003).",
    )
    args = parser.parse_args()
    
    # Parse collections
    collections_config = []
    for collection_str in args.collections:
        parts = collection_str.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid collection format: {collection_str}. Expected 'source_name:collection_name'"
            )
        source_name, collection_name = parts
        collections_config.append({
            "source_name": source_name,
            "collection_name": collection_name,
        })
    
    chat_log_path = Path(args.chat_log) if args.chat_log else None
    
    app = create_app(
        collections_config,
        args.model_path,
        args.ollama_model,
        args.weaviate_url,
        chat_log_path,
    )
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
