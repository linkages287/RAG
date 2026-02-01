#!/usr/bin/env python3
"""
LangChain-based RAG Chatbot using Weaviate vector store.
Streams responses and maintains conversation history.
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional

try:
    from langchain_community.vectorstores import Weaviate
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_community.llms import Ollama
    from langchain.prompts import PromptTemplate
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError:
        try:
            from langchain.embeddings.base import Embeddings
        except ImportError:
            from langchain.schema.embeddings import Embeddings
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: LangChain packages not found. Install with:")
    print("  pip install langchain langchain-community langchain-text-splitters")

from flask import Flask, Response, jsonify, render_template, request, session, stream_with_context


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Wrapper for SentenceTransformer to work with LangChain for local model paths."""
    
    def __init__(self, model_path: str, normalize_embeddings: bool = True):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for local model paths. Install with: pip install sentence-transformers")
        self.model = SentenceTransformer(model_path)
        self.normalize_embeddings = normalize_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.tolist()


def create_app(
    collection_names: List[str],
    weaviate_url: str = "http://localhost:8080",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ollama_model: str = "llama3.2",
    chat_log_path: Optional[Path] = None,
) -> Flask:
    """Create Flask app with LangChain RAG chatbot."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain packages are required")
    
    app = Flask(__name__)
    app.secret_key = "langchain-rag-chatbot-secret-key"
    
    # Initialize embeddings
    # Check if using local model path
    model_path = Path(embedding_model)
    if model_path.exists() and model_path.is_dir():
        # Local model path - use SentenceTransformer directly
        print(f"Using local model at: {embedding_model}")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for local model paths. Install with: pip install sentence-transformers")
        embeddings = LocalSentenceTransformerEmbeddings(
            model_path=str(model_path),
            normalize_embeddings=True,
        )
    else:
        # Hugging Face repo ID or model name - use HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    
    # Initialize vector stores for each collection
    vector_stores = []
    for collection_name in collection_names:
        try:
            vectorstore = Weaviate(
                client=None,  # Will be created by Weaviate class
                index_name=collection_name,
                text_key="text",
                embedding=embeddings,
                weaviate_url=weaviate_url,
                by_text=False,
            )
            vector_stores.append(vectorstore)
            print(f"Connected to Weaviate collection: {collection_name}")
        except Exception as e:
            print(f"Warning: Could not connect to collection {collection_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not vector_stores:
        raise ValueError("No vector stores could be initialized")
    
    # Use first vector store as retriever (or combine if multiple)
    if len(vector_stores) == 1:
        retriever = vector_stores[0].as_retriever(
            search_kwargs={"k": 12}
        )
    else:
        # For multiple collections, use the first one
        # TODO: Could implement multi-retriever ensemble
        retriever = vector_stores[0].as_retriever(
            search_kwargs={"k": 12}
        )
    
    # Initialize LLM
    llm = Ollama(model=ollama_model, base_url="http://localhost:11434")
    
    # Custom prompt template
    template = """You are a NATO analyst. Use the following pieces of context to answer the question.
If you don't know the answer based on the context, say so explicitly.

Context: {context}

Question: {question}

Answer:"""
    
    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create conversation chain with memory
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,  # Keep last 10 exchanges
        output_key="answer"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )
    
    @app.route("/", methods=["GET", "POST"])
    def index():
        clear_history = request.form.get("clear", "") == "clear"
        
        if "messages" not in session or clear_history:
            session["messages"] = []
            # Clear LangChain memory
            memory.clear()
        
        messages = session.get("messages", [])
        
        return render_template(
            "multi_chat.html",
            messages=messages,
            sources=collection_names,
        )
    
    @app.route("/api/chat", methods=["POST"])
    def chat():
        """API endpoint for chat with streaming."""
        data = request.json
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        if "messages" not in session:
            session["messages"] = []
        
        messages = session.get("messages", [])
        messages.append({"role": "user", "content": query})
        
        # Stream response
        def generate():
            nonlocal messages
            full_response = ""
            sources_used = []
            
            try:
                # Get response from LangChain
                result = qa_chain.invoke({"question": query})
                answer = result.get("answer", "")
                source_docs = result.get("source_documents", [])
                
                # Extract sources
                for doc in source_docs[:5]:
                    source = doc.metadata.get("source", "unknown")
                    if source not in sources_used:
                        sources_used.append(source)
                
                # Stream the answer word by word (simulate streaming)
                words = answer.split()
                for word in words:
                    full_response += word + " "
                    yield f"data: {json.dumps({'chunk': word + ' ', 'done': False})}\n\n"
                
                # Final message
                yield f"data: {json.dumps({
                    'chunk': '',
                    'done': True,
                    'metadata': {
                        'query_type': 'rag',
                        'sources': sources_used,
                        'top_results': [
                            {
                                'source': doc.metadata.get('source', 'unknown'),
                                'country': doc.metadata.get('country', 'unknown'),
                                'score': 0.0,  # LangChain doesn't provide scores directly
                            }
                            for doc in source_docs[:3]
                        ],
                    }
                })}\n\n"
                
                # Save assistant message
                messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "query_type": "rag",
                    "sources": sources_used,
                })
                session["messages"] = messages
                
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
                import traceback
                traceback.print_exc()
        
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    
    @app.route("/api/clear", methods=["POST"])
    def clear_chat():
        """Clear all chat history."""
        session["messages"] = []
        memory.clear()
        return jsonify({"success": True, "message": "Chat history cleared"})
    
    return app


def main():
    parser = argparse.ArgumentParser(
        description="LangChain RAG Chatbot with Weaviate and streaming responses."
    )
    parser.add_argument(
        "--collections",
        required=True,
        nargs="+",
        help="Weaviate collection names to use for RAG.",
    )
    parser.add_argument(
        "--weaviate-url",
        default="http://localhost:8080",
        help="Weaviate server URL (default: http://localhost:8080).",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model (default: sentence-transformers/all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model name (default: llama3.2).",
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
        default=5004,
        help="Port (default: 5004).",
    )
    args = parser.parse_args()
    
    if not LANGCHAIN_AVAILABLE:
        print("Error: LangChain packages are required.")
        print("Install with: pip install langchain langchain-community langchain-text-splitters")
        return
    
    chat_log_path = Path(args.chat_log) if args.chat_log else None
    
    app = create_app(
        args.collections,
        args.weaviate_url,
        args.embedding_model,
        args.ollama_model,
        chat_log_path,
    )
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
