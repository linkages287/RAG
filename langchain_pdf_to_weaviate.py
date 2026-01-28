#!/usr/bin/env python3
"""
Process PDF directory using LangChain, vectorize, and store in Weaviate.
Uses LangChain's document loaders, text splitters, and Weaviate integration.
"""
import argparse
import re
from pathlib import Path
from typing import List

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
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
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"Warning: LangChain packages not found. Error: {e}")
    print("Install with: pip install langchain langchain-community langchain-text-splitters")

import weaviate
from weaviate.classes.config import Configure, Property, DataType


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


# Text cleaning functions (from pdf_to_text_chunker.py)
def is_page_number_line(line: str) -> bool:
    """Detect lines that are likely page numbers (arabic or roman numerals)."""
    stripped = re.sub(r"[^\w]", "", line).lower()
    if not stripped:
        return False
    if stripped.isdigit():
        return True
    return stripped in {
        "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
        "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
    }


def extract_header_footer_candidates(pages: List[str]) -> tuple[set, set]:
    """Find repeated first/last lines across pages as header/footer candidates."""
    header_counts = {}
    footer_counts = {}
    for page_text in pages:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0]
        footer = lines[-1]
        header_counts[header] = header_counts.get(header, 0) + 1
        footer_counts[footer] = footer_counts.get(footer, 0) + 1

    headers = {line for line, count in header_counts.items() if count >= 2}
    footers = {line for line, count in footer_counts.items() if count >= 2}
    return headers, footers


SPLIT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "for",
    "from", "has", "have", "he", "her", "his", "how", "i", "if", "in", "is",
    "it", "its", "may", "me", "my", "no", "not", "of", "on", "or", "our",
    "she", "so", "that", "the", "their", "there", "they", "this", "to", "us",
    "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your",
}

SPLIT_SUFFIXES = {
    "al", "ed", "e", "er", "ers", "es", "ing", "ins", "ion", "ive", "ly",
    "nal", "ness", "ss",
}


def merge_ocr_splits(text: str) -> str:
    """Merge common OCR split tokens while avoiding stopword joins."""
    def repl(match: re.Match) -> str:
        left = match.group(1)
        right = match.group(2)
        if left.lower() in SPLIT_STOPWORDS or right.lower() in SPLIT_STOPWORDS:
            return f"{left} {right}"
        return f"{left}{right}"

    def repl_suffix(match: re.Match) -> str:
        left = match.group(1)
        right = match.group(2)
        if left.lower() in SPLIT_STOPWORDS:
            return f"{left} {right}"
        if right.lower() not in SPLIT_SUFFIXES:
            return f"{left} {right}"
        return f"{left}{right}"

    text = re.sub(r"\b([A-Za-z]{1,2})\s+([a-z]{2,})\b", repl, text)
    text = re.sub(r"\b([A-Za-z]{3,})\s+([a-z]{1,3})\b", repl_suffix, text)
    text = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", text)
    text = re.sub(r"\b([A-Z]{1,3})\s+([A-Z]{2,})\b", repl, text)
    return text


# Regex patterns for coordinate detection
DECIMAL_PAIR_RE = re.compile(
    r"(?P<lat>[+-]?\d{1,2}\.\d+)\s*[,;/]\s*(?P<lon>[+-]?\d{1,3}\.\d+)"
)

DMS_RE = re.compile(
    r"(?P<deg>\d{1,3})\s*[°]\s*"
    r"(?P<min>\d{1,2})\s*[′'\u2019]\s*"
    r"(?P<sec>\d{1,2}(?:\.\d+)?)?\s*[″\"\u201d]?\s*"
    r"(?P<hem>[NSEW])",
    re.IGNORECASE,
)


def normalize_decimal_pair(lat: float, lon: float) -> tuple[float, float] | None:
    """Validate and normalize lat/lon pair."""
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return lat, lon
    if -90.0 <= lon <= 90.0 and -180.0 <= lat <= 180.0:
        return lon, lat
    return None


def truncate_decimal(num: float) -> str:
    """Truncate a decimal number to 3 decimal places, removing trailing zeros."""
    truncated = f"{num:.3f}"
    if "." in truncated:
        truncated = truncated.rstrip("0").rstrip(".")
    return truncated


def truncate_coordinates(text: str) -> str:
    """Find and truncate coordinates to 3 decimal places."""
    def truncate_decimal_pair(match: re.Match) -> str:
        lat_str = match.group("lat")
        lon_str = match.group("lon")
        full_match = match.group(0)
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            normalized = normalize_decimal_pair(lat, lon)
            if normalized:
                lat_trunc = truncate_decimal(normalized[0])
                lon_trunc = truncate_decimal(normalized[1])
                sep_match = re.search(r"\s*([,;/])\s*", full_match)
                sep = sep_match.group(1) if sep_match else ","
                return f"{lat_trunc}{sep} {lon_trunc}"
        except (ValueError, AttributeError):
            pass
        return full_match
    
    def truncate_dms_sec(match: re.Match) -> str:
        sec_str = match.group("sec")
        if sec_str and "." in sec_str:
            try:
                sec = float(sec_str)
                sec_trunc = truncate_decimal(sec)
                deg = match.group("deg")
                min_val = match.group("min")
                hem = match.group("hem")
                original = match.group(0)
                quote_char = "\"" if "\"" in original else "″" if "″" in original else "\""
                sec_part = f"{sec_trunc}{quote_char}"
                return f"{deg}°{min_val}'{sec_part}{hem}"
            except (ValueError, AttributeError):
                pass
        return match.group(0)
    
    text = DECIMAL_PAIR_RE.sub(truncate_decimal_pair, text)
    text = DMS_RE.sub(truncate_dms_sec, text)
    return text


def clean_page_text(text: str, headers: set, footers: set) -> str:
    """Clean page text for RAG: remove headers/footers and normalize spacing."""
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: List[str] = []
    for line in lines:
        if not line:
            continue
        if line in headers or line in footers or is_page_number_line(line):
            continue
        if re.search(r"\.{2,}\s*\d+\b", line):  # Table of contents lines
            continue
        if re.search(r"\bTABLE OF CONTENTS?\b", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    # Merge hyphenated words across newlines
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    # Normalize newlines to spaces
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned)
    # Remove NATO UNCLASSIFIED
    cleaned = re.sub(r"\bNATO\s+UNCLASSIFIED\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bNATO\s+SECRET\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bNATO\s+CONFIDENTIAL\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bNATO\s+RESTRICTED\b", " ", cleaned, flags=re.IGNORECASE)
    # Fix hyphen spacing
    cleaned = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", cleaned)
    # Remove bullet points
    cleaned = re.sub(r"[•\u2022]", " ", cleaned)
    # Fix punctuation spacing
    cleaned = re.sub(r"\s+([,;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])(?=\w)", r"\1 ", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s*\.\s*(\.)+\s*", ". ", cleaned)
    # Fix common OCR errors
    cleaned = re.sub(r"\b([Aa])\s+nd\b", "and", cleaned)
    cleaned = re.sub(r"\b([Aa])\s+nnually\b", "annually", cleaned)
    cleaned = merge_ocr_splits(cleaned)
    cleaned = re.sub(r"\bt\s+o([a-z]{2,})\b", r"to \1", cleaned, flags=re.IGNORECASE)
    # Fix all caps word splits
    cleaned = re.sub(r"\b([A-Z]{2,})([a-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\b([A-Z]{2,})\s+s\b", r"\1s", cleaned)
    cleaned = re.sub(r"([a-z])(?=etc\.)", r"\1 ", cleaned, flags=re.IGNORECASE)
    # Truncate coordinates
    cleaned = truncate_coordinates(cleaned)
    # Final whitespace normalization
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_country_from_filename(filename: str) -> str:
    """Extract country code from PDF filename."""
    match = re.search(r"NU_JWC_(?:ETI_)?([A-Z]+)", filename.upper())
    if match:
        return match.group(1)
    return Path(filename).stem


def create_weaviate_collection(client: weaviate.WeaviateClient, collection_name: str):
    """Create Weaviate collection schema for documents."""
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return
    
    try:
        vector_config = Configure.Vectors.none()
    except AttributeError:
        try:
            vector_config = Configure.Vector.none()
        except AttributeError:
            vector_config = None
    
    create_kwargs = {
        "name": collection_name,
        "description": "PDF documents processed with LangChain",
        "properties": [
            Property(name="text", data_type=DataType.TEXT, description="Document chunk text"),
            Property(name="source", data_type=DataType.TEXT, description="Source PDF filename"),
            Property(name="page", data_type=DataType.INT, description="Page number"),
            Property(name="chunk_index", data_type=DataType.INT, description="Chunk index within document"),
        ],
    }
    
    if vector_config is not None:
        create_kwargs["vector_config"] = vector_config
    
    client.collections.create(**create_kwargs)
    print(f"Created collection '{collection_name}'")


def process_pdf_directory(
    pdf_directory: Path,
    collection_name: str,
    weaviate_url: str = "http://localhost:8080",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """
    Process all PDFs in a directory and store in Weaviate using LangChain.
    
    Args:
        pdf_directory: Directory containing PDF files
        collection_name: Weaviate collection name
        weaviate_url: Weaviate server URL
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_model: HuggingFace embedding model name
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain packages are required. Install with: pip install langchain langchain-community langchain-text-splitters")
    
    # Find all PDF files
    pdf_files = list(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_directory}")
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Initialize embeddings
    print(f"Loading embedding model: {embedding_model}...")
    
    # Check if using local model path
    model_path = Path(embedding_model)
    if model_path.exists() and model_path.is_dir():
        # Local model path - use SentenceTransformer directly
        print(f"  Using local model at: {embedding_model}")
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
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    
    # Connect to Weaviate (for collection creation - optional)
    print(f"Connecting to Weaviate at {weaviate_url}...")
    try:
        if weaviate_url == "http://localhost:8080":
            client = weaviate.connect_to_local()
        else:
            url_parts = weaviate_url.replace("http://", "").replace("https://", "").split(":")
            host = url_parts[0]
            port = int(url_parts[1]) if len(url_parts) > 1 else 8080
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=weaviate_url.startswith("https://"),
            )
        
        if not client.is_ready():
            raise ConnectionError("Weaviate is not ready")
        
        # Create collection if needed (optional - LangChain can create it)
        try:
            create_weaviate_collection(client, collection_name)
        except Exception as e:
            print(f"Note: Collection creation skipped (LangChain will handle it): {e}")
        
        if hasattr(client, '__exit__'):
            client.__exit__(None, None, None)
    except Exception as e:
        print(f"Warning: Could not pre-create collection: {e}")
        print("LangChain will attempt to create it automatically.")
    
    # Process each PDF
    all_documents = []
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            # Load PDF with LangChain
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            
            # Extract raw page texts for header/footer detection
            page_texts = [page.page_content for page in pages]
            headers, footers = extract_header_footer_candidates(page_texts)
            
            # Clean each page and update content
            total_pages = len(pages)
            for i, page in enumerate(pages):
                # Clean the page content
                cleaned_text = clean_page_text(page.page_content, headers, footers)
                page.page_content = cleaned_text
                
                # Add metadata
                page.metadata.update({
                    "source": pdf_file.name,
                    "page": page.metadata.get("page", i + 1),
                })
                
                # Progress indicator
                if (i + 1) % 10 == 0 or (i + 1) == total_pages:
                    percent = 100 * (i + 1) / total_pages
                    print(f"  Cleaned {i + 1}/{total_pages} pages ({percent:.0f}%)...", end="\r")
            
            print()  # Newline after progress
            
            # Split into chunks (after cleaning)
            chunks = text_splitter.split_documents(pages)
            
            # Add chunk index to metadata
            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = idx
            
            all_documents.extend(chunks)
            print(f"  Created {len(chunks)} chunks from {len(pages)} pages")
            
        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_documents:
        print("No documents to store.")
        return
    
    print(f"\nTotal chunks to store: {len(all_documents)}")
    
    # Store in Weaviate using Weaviate v4 API directly
    # (LangChain's Weaviate integration doesn't support v4 yet)
    print(f"Storing documents in Weaviate collection '{collection_name}'...")
    
    # Connect to Weaviate v4
    if weaviate_url == "http://localhost:8080":
        client = weaviate.connect_to_local()
    else:
        url_parts = weaviate_url.replace("http://", "").replace("https://", "").split(":")
        host = url_parts[0]
        port = int(url_parts[1]) if len(url_parts) > 1 else 8080
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=weaviate_url.startswith("https://"),
        )
    
    try:
        if not client.is_ready():
            raise ConnectionError("Weaviate is not ready")
        
        # Ensure collection exists
        if not client.collections.exists(collection_name):
            create_weaviate_collection(client, collection_name)
        
        # Get collection
        collection = client.collections.get(collection_name)
        
        # Embed and store documents in batches
        batch_size = 50
        total = len(all_documents)
        stored = 0
        
        print(f"Embedding and storing {total} chunks...")
        for i, doc in enumerate(all_documents):
            try:
                # Generate embedding using LangChain
                doc_embedding = embeddings.embed_query(doc.page_content)
                
                # Prepare properties
                properties = {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0),
                    "chunk_index": doc.metadata.get("chunk_index", i),
                }
                
                # Insert with vector
                collection.data.insert(
                    properties=properties,
                    vector=doc_embedding,
                )
                stored += 1
                
                if (i + 1) % batch_size == 0 or (i + 1) == total:
                    percent = 100 * (i + 1) / total
                    print(f"  Stored {i + 1}/{total} chunks ({percent:.1f}%)...")
                    
            except Exception as e:
                print(f"  Error storing chunk {i + 1}: {e}")
                continue
        
        print(f"\nSuccessfully stored {stored}/{total} chunks in Weaviate collection '{collection_name}'!")
        
    finally:
        if hasattr(client, '__exit__'):
            client.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Process PDF directory with LangChain and store in Weaviate."
    )
    parser.add_argument(
        "pdf_directory",
        help="Directory containing PDF files to process.",
    )
    parser.add_argument(
        "--collection-name",
        default="PDFDocuments",
        help="Weaviate collection name (default: PDFDocuments).",
    )
    parser.add_argument(
        "--weaviate-url",
        default="http://localhost:8080",
        help="Weaviate server URL (default: http://localhost:8080).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200).",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model (default: sentence-transformers/all-MiniLM-L6-v2).",
    )
    args = parser.parse_args()
    
    pdf_dir = Path(args.pdf_directory)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise ValueError(f"PDF directory not found: {pdf_dir}")
    
    process_pdf_directory(
        pdf_dir,
        args.collection_name,
        args.weaviate_url,
        args.chunk_size,
        args.chunk_overlap,
        args.embedding_model,
    )


if __name__ == "__main__":
    main()
