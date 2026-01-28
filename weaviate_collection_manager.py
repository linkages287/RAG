#!/usr/bin/env python3
"""
Graphical Text Interface (TUI) for managing Weaviate collections.
Interactive menu-driven interface to list, import, and remove collections.
Supports importing collections from JSON and NPZ files.
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType

# Try to import embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


def extract_country(source_pdf: str | None) -> str:
    """Extract country code from PDF filename."""
    if not source_pdf:
        return "unknown"
    match = re.search(r"NU_JWC_(?:ETI_)?([A-Z]+)", str(source_pdf).upper())
    if match:
        return match.group(1)
    return Path(source_pdf).stem if source_pdf else "unknown"


def load_vectors(npz_path: Path) -> np.ndarray:
    """Load vectors from npz file."""
    data = np.load(npz_path, allow_pickle=False)
    return data["vectors"]


def load_chunks(json_path: Path) -> List[Dict]:
    """Load chunks from JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("chunks", [])


def create_collection_schema(client: weaviate.WeaviateClient, collection_name: str = "DocumentChunk") -> None:
    """Create or update Weaviate collection for document chunks."""
    if client.collections.exists(collection_name):
        return
    
    # Try different API versions for vector configuration
    try:
        # Try Vectors (plural) - this is the correct API
        vector_config = Configure.Vectors.none()
    except AttributeError:
        try:
            # Fallback: Try Vector (singular) - older API
            vector_config = Configure.Vector.none()
        except AttributeError:
            # If neither works, create without vector config
            vector_config = None
    
    create_kwargs = {
        "name": collection_name,
        "description": "Document chunks with embeddings for RAG system",
        "properties": [
            Property(name="chunk_id", data_type=DataType.INT, description="Unique chunk identifier"),
            Property(name="text", data_type=DataType.TEXT, description="Chunk text content"),
            Property(name="source_pdf", data_type=DataType.TEXT, description="Source PDF filename"),
            Property(name="country", data_type=DataType.TEXT, description="Country code extracted from PDF name"),
            Property(name="page_start", data_type=DataType.INT, description="Starting page number"),
            Property(name="page_end", data_type=DataType.INT, description="Ending page number"),
            Property(name="token_count", data_type=DataType.INT, description="Number of tokens in chunk"),
        ],
    }
    
    if vector_config is not None:
        create_kwargs["vector_config"] = vector_config
    
    client.collections.create(**create_kwargs)


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}{Colors.RESET}\n")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")


def print_table(headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None):
    """Print a formatted table with headers and rows."""
    if not rows:
        return
    
    # Auto-calculate column widths if not provided
    if col_widths is None:
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(str(header))
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)
    
    # Print header
    header_row = "│".join([f" {str(headers[i]):<{col_widths[i]-1}}" for i in range(len(headers))])
    print(f"{Colors.BOLD}│{header_row}│{Colors.RESET}")
    
    # Print separator
    separator = "─" * (sum(col_widths) + len(headers) - 1)
    print(f"{Colors.BOLD}├{separator}┤{Colors.RESET}")
    
    # Print rows
    for row in rows:
        row_str = "│".join([f" {str(row[i] if i < len(row) else ''):<{col_widths[i]-1}}" for i in range(len(headers))])
        print(f"│{row_str}│")
    
    # Print footer
    footer_separator = "─" * (sum(col_widths) + len(headers) - 1)
    print(f"{Colors.BOLD}└{footer_separator}┘{Colors.RESET}")


def print_list(items: List[str], numbered: bool = False, marker: str = "•"):
    """Print a formatted list."""
    for i, item in enumerate(items, 1):
        if numbered:
            print(f"  {Colors.WHITE}{i}.{Colors.RESET} {item}")
        else:
            print(f"  {Colors.CYAN}{marker}{Colors.RESET} {item}")


def print_key_value(key: str, value: str, indent: int = 2):
    """Print a key-value pair in a formatted way."""
    spaces = " " * indent
    print(f"{spaces}{Colors.BOLD}{key}:{Colors.RESET} {value}")


def get_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default."""
    if default:
        full_prompt = f"{Colors.CYAN}{prompt} [{default}]: {Colors.RESET}"
    else:
        full_prompt = f"{Colors.CYAN}{prompt}: {Colors.RESET}"
    result = input(full_prompt).strip()
    return result if result else (default if default else "")


def confirm(prompt: str, default: bool = False) -> bool:
    """Get yes/no confirmation from user."""
    default_text = "Y/n" if default else "y/N"
    response = get_input(f"{prompt} ({default_text})", "Y" if default else "N")
    return response.lower() in ['y', 'yes']


def list_collections(client: weaviate.WeaviateClient) -> List[str]:
    """List all collections in Weaviate."""
    try:
        collections = client.collections.list_all()
        # list_all() returns a dict in v4, convert to list of names
        if isinstance(collections, dict):
            return list(collections.keys())
        elif isinstance(collections, list):
            return collections
        else:
            # Try to convert if it's iterable
            return list(collections) if hasattr(collections, '__iter__') else []
    except AttributeError:
        try:
            schema = client.schema.get()
            if schema and 'classes' in schema:
                return [cls['class'] for cls in schema['classes']]
            return []
        except:
            return []
    except Exception as e:
        print_error(f"Error listing collections: {e}")
        return []


def get_collection_info(client: weaviate.WeaviateClient, collection_name: str) -> Optional[Dict]:
    """Get information about a collection."""
    try:
        if not client.collections.exists(collection_name):
            return None
        
        collection = client.collections.get(collection_name)
        # Try to get object count
        count = "unknown"
        try:
            result = collection.query.fetch_objects(
                limit=1,
                return_metadata=weaviate.classes.query.MetadataQuery(count=True)
            )
            if hasattr(result, 'total_count'):
                count = result.total_count
            elif hasattr(result, 'objects'):
                # Try fetching more to estimate
                large_result = collection.query.fetch_objects(limit=1000)
                if large_result.objects:
                    count = f"~{len(large_result.objects)}+"
        except:
            pass
        
        return {
            "name": collection_name,
            "exists": True,
            "object_count": count,
        }
    except Exception as e:
        print_error(f"Error getting collection info: {e}")
        return None


def display_collections(client: weaviate.WeaviateClient):
    """Display all collections in a formatted table."""
    clear_screen()
    print_header("Weaviate Collections")
    
    collections = list_collections(client)
    
    if not collections:
        print_warning("No collections found in Weaviate.")
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
        return
    
    # Prepare table data
    headers = ["#", "Collection Name", "Objects"]
    rows = []
    for i, coll_name in enumerate(collections, 1):
        info = get_collection_info(client, coll_name)
        count = info['object_count'] if info else "unknown"
        rows.append([
            f"{Colors.WHITE}{i}{Colors.RESET}",
            coll_name,
            f"{Colors.YELLOW}{str(count)}{Colors.RESET}"
        ])
    
    print_table(headers, rows, col_widths=[6, 35, 18])
    print(f"\n{Colors.GREEN}Total: {len(collections)} collection(s){Colors.RESET}")
    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")


def delete_collection_interactive(client: weaviate.WeaviateClient):
    """Interactive collection deletion."""
    clear_screen()
    print_header("Delete Collection")
    
    collections = list_collections(client)
    
    if not collections:
        print_warning("No collections found in Weaviate.")
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
        return
    
    # Display collections with numbers
    print(f"{Colors.BOLD}Available Collections:{Colors.RESET}\n")
    collection_items = []
    for coll_name in collections:
        info = get_collection_info(client, coll_name)
        count = info['object_count'] if info else "unknown"
        collection_items.append(f"{coll_name:<30} {Colors.YELLOW}({count} objects){Colors.RESET}")
    
    print_list(collection_items, numbered=True)
    print(f"\n  {Colors.WHITE}0.{Colors.RESET} Cancel")
    
    try:
        choice = get_input("\nSelect collection number to delete", "0")
        choice_num = int(choice)
        
        if choice_num == 0:
            print_info("Deletion cancelled.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        if 1 <= choice_num <= len(collections):
            collection_name = collections[choice_num - 1]
            
            # Get collection info
            info = get_collection_info(client, collection_name)
            if info:
                print(f"\n{Colors.BOLD}Collection Details:{Colors.RESET}")
                print_key_value("Name", collection_name)
                print_key_value("Objects", str(info['object_count']))
            
            # Confirm deletion
            if confirm(f"\n{Colors.RED}Are you sure you want to delete '{collection_name}'?{Colors.RESET}", False):
                try:
                    # Verify collection exists before deletion
                    if not client.collections.exists(collection_name):
                        print_error(f"Collection '{collection_name}' does not exist.")
                    else:
                        # Check if delete method exists
                        if not hasattr(client.collections, 'delete'):
                            print_error("Delete method not available on client.collections")
                            print_info("Available methods: " + ", ".join(dir(client.collections)))
                        else:
                            client.collections.delete(collection_name)
                            # Verify deletion
                            if not client.collections.exists(collection_name):
                                print_success(f"Collection '{collection_name}' deleted successfully!")
                            else:
                                print_warning(f"Collection '{collection_name}' may not have been deleted. Please verify.")
                except AttributeError as e:
                    print_error(f"API error: {e}")
                    print_info(f"Client type: {type(client)}")
                    print_info(f"Collections type: {type(client.collections)}")
                    print_info("Available methods: " + ", ".join([m for m in dir(client.collections) if not m.startswith('_')]))
                except Exception as e:
                    print_error(f"Error deleting collection: {e}")
                    print_info(f"Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
            else:
                print_info("Deletion cancelled.")
        else:
            print_error("Invalid selection.")
    except ValueError:
        print_error("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Operation cancelled.{Colors.RESET}")
    except Exception as e:
        print_error(f"Error: {e}")
    
    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")


def delete_multiple_collections(client: weaviate.WeaviateClient):
    """Delete multiple collections interactively."""
    clear_screen()
    print_header("Delete Multiple Collections")
    
    collections = list_collections(client)
    
    if not collections:
        print_warning("No collections found in Weaviate.")
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
        return
    
    # Display collections
    print(f"{Colors.BOLD}Available Collections:{Colors.RESET}\n")
    selected = []
    
    while True:
        print(f"\n{Colors.BOLD}Selected: {len(selected)} collection(s){Colors.RESET}\n")
        collection_items = []
        for i, coll_name in enumerate(collections, 1):
            marker = f"{Colors.GREEN}✓{Colors.RESET}" if coll_name in selected else " "
            info = get_collection_info(client, coll_name)
            count = info['object_count'] if info else "unknown"
            collection_items.append(f"{marker} {coll_name:<30} {Colors.YELLOW}({count} objects){Colors.RESET}")
        
        print_list(collection_items, numbered=True)
        
        print(f"\n  {Colors.WHITE}0.{Colors.RESET} Done selecting")
        print(f"  {Colors.WHITE}d.{Colors.RESET} Delete selected")
        print(f"  {Colors.WHITE}c.{Colors.RESET} Cancel")
        
        choice = get_input("\nSelect collection number (or 0/d/c)").strip().lower()
        
        if choice == '0':
            break
        elif choice == 'd':
            if not selected:
                print_warning("No collections selected.")
                continue
            
            print(f"\n{Colors.BOLD}Collections to delete:{Colors.RESET}")
            delete_items = []
            for coll_name in selected:
                info = get_collection_info(client, coll_name)
                count = info['object_count'] if info else "unknown"
                delete_items.append(f"{coll_name} {Colors.YELLOW}({count} objects){Colors.RESET}")
            
            print_list(delete_items)
            
            if confirm(f"\n{Colors.RED}Delete {len(selected)} collection(s)?{Colors.RESET}", False):
                deleted = 0
                for coll_name in selected:
                    try:
                        if not client.collections.exists(coll_name):
                            print_warning(f"Collection '{coll_name}' does not exist, skipping.")
                            continue
                        client.collections.delete(coll_name)
                        if not client.collections.exists(coll_name):
                            print_success(f"Deleted '{coll_name}'")
                            deleted += 1
                        else:
                            print_warning(f"Collection '{coll_name}' may not have been deleted.")
                    except Exception as e:
                        print_error(f"Error deleting '{coll_name}': {e}")
                        import traceback
                        traceback.print_exc()
                
                print_success(f"\nDeleted {deleted}/{len(selected)} collection(s).")
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
                return
            else:
                print_info("Deletion cancelled.")
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
                return
        elif choice == 'c':
            print_info("Operation cancelled.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        else:
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(collections):
                    coll_name = collections[choice_num - 1]
                    if coll_name in selected:
                        selected.remove(coll_name)
                        print_info(f"Removed '{coll_name}' from selection.")
                    else:
                        selected.append(coll_name)
                        print_info(f"Added '{coll_name}' to selection.")
                else:
                    print_error("Invalid selection.")
            except ValueError:
                print_error("Invalid input.")


def delete_all_collections(client: weaviate.WeaviateClient):
    """Delete all collections with confirmation."""
    clear_screen()
    print_header("Delete All Collections")
    
    collections = list_collections(client)
    
    if not collections:
        print_warning("No collections found in Weaviate.")
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
        return
    
    print(f"{Colors.RED}{Colors.BOLD}WARNING: This will delete ALL {len(collections)} collection(s)!{Colors.RESET}\n")
    
    print(f"{Colors.BOLD}Collections to be deleted:{Colors.RESET}\n")
    delete_items = []
    for coll_name in collections:
        info = get_collection_info(client, coll_name)
        count = info['object_count'] if info else "unknown"
        delete_items.append(f"{coll_name:<30} {Colors.YELLOW}({count} objects){Colors.RESET}")
    
    print_list(delete_items, numbered=True)
    
    if confirm(f"\n{Colors.RED}{Colors.BOLD}Are you absolutely sure?{Colors.RESET}", False):
        if confirm(f"{Colors.RED}This action cannot be undone. Continue?{Colors.RESET}", False):
            deleted = 0
            for coll_name in collections:
                try:
                    if not client.collections.exists(coll_name):
                        print_warning(f"Collection '{coll_name}' does not exist, skipping.")
                        continue
                    client.collections.delete(coll_name)
                    if not client.collections.exists(coll_name):
                        print_success(f"Deleted '{coll_name}'")
                        deleted += 1
                    else:
                        print_warning(f"Collection '{coll_name}' may not have been deleted.")
                except Exception as e:
                    print_error(f"Error deleting '{coll_name}': {e}")
                    import traceback
                    traceback.print_exc()
            
            print_success(f"\nDeleted {deleted}/{len(collections)} collection(s).")
        else:
            print_info("Deletion cancelled.")
    else:
        print_info("Deletion cancelled.")
    
    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")


def save_collection_interactive(client: weaviate.WeaviateClient):
    """Interactive collection export to JSON and NPZ files."""
    clear_screen()
    print_header("Save Collection to JSON/NPZ Files")
    
    collections = list_collections(client)
    
    if not collections:
        print_warning("No collections found in Weaviate.")
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
        return
    
    # Display collections with numbers
    print(f"{Colors.BOLD}Available Collections:{Colors.RESET}\n")
    collection_items = []
    for coll_name in collections:
        info = get_collection_info(client, coll_name)
        count = info['object_count'] if info else "unknown"
        collection_items.append(f"{coll_name:<30} {Colors.YELLOW}({count} objects){Colors.RESET}")
    
    print_list(collection_items, numbered=True)
    print(f"\n  {Colors.WHITE}0.{Colors.RESET} Cancel")
    
    try:
        choice = get_input("\nSelect collection number to save", "0")
        choice_num = int(choice)
        
        if choice_num == 0:
            print_info("Export cancelled.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        if not (1 <= choice_num <= len(collections)):
            print_error("Invalid selection.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        collection_name = collections[choice_num - 1]
        
        # Check if collection exists
        if not client.collections.exists(collection_name):
            print_error(f"Collection '{collection_name}' does not exist.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        # Get output file paths
        print(f"\n{Colors.BOLD}Step 1: JSON output file{Colors.RESET}\n")
        json_path_str = get_input("Enter path for JSON file", f"{collection_name}.json")
        if not json_path_str:
            print_warning("JSON file path is required.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        json_path = Path(json_path_str)
        if json_path.exists():
            if not confirm(f"File '{json_path}' already exists. Overwrite?", False):
                print_info("Export cancelled.")
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
                return
        
        print(f"\n{Colors.BOLD}Step 2: NPZ output file{Colors.RESET}\n")
        npz_path_str = get_input("Enter path for NPZ file", f"{collection_name}.npz")
        if not npz_path_str:
            print_warning("NPZ file path is required.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        npz_path = Path(npz_path_str)
        if npz_path.exists():
            if not confirm(f"File '{npz_path}' already exists. Overwrite?", False):
                print_info("Export cancelled.")
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
                return
        
        # Fetch all objects from collection with vectors via pagination
        print(f"\n{Colors.BLUE}Fetching objects and vectors from collection '{collection_name}'...{Colors.RESET}")
        print_info("Attempting to fetch vectors programmatically via pagination...")
        collection = client.collections.get(collection_name)
        
        # Fetch all objects with vectors (using pagination)
        all_objects = []
        all_vectors = []
        limit = 1000
        offset = 0
        
        while True:
            try:
                # Fetch objects with pagination
                result = collection.query.fetch_objects(
                    limit=limit,
                    offset=offset
                )
                
                if not result.objects:
                    break
                
                # Process each object and fetch its vector by UUID
                for obj in result.objects:
                    all_objects.append(obj)
                    
                    # Fetch vector for this object by UUID using data.get_by_id
                    vector = None
                    try:
                        # Use data.get_by_id with include_vector=True to get the vector
                        obj_with_vector = collection.data.get_by_id(
                            obj.uuid,
                            include_vector=True
                        )
                        if obj_with_vector:
                            # Extract vector from the object
                            if hasattr(obj_with_vector, 'vector'):
                                vec_data = obj_with_vector.vector
                                if isinstance(vec_data, dict):
                                    # Multiple vectors (named), get default or first
                                    vector = vec_data.get('default') or list(vec_data.values())[0]
                                elif vec_data is not None:
                                    vector = vec_data
                            elif hasattr(obj_with_vector, 'vectors'):
                                # Try vectors (plural)
                                vec_data = obj_with_vector.vectors
                                if isinstance(vec_data, dict):
                                    vector = vec_data.get('default') or list(vec_data.values())[0]
                                elif vec_data is not None:
                                    vector = vec_data
                    except Exception as e:
                        # Vector fetch failed for this object
                        vector = None
                    
                    all_vectors.append(vector)
                
                valid_vectors = sum(1 for v in all_vectors if v is not None)
                print(f"  Fetched {len(all_objects)} objects (vectors: {valid_vectors}/{len(all_vectors)})...", end="\r")
                
                if len(result.objects) < limit:
                    break
                
                offset += limit
                
            except Exception as e:
                print_error(f"Error fetching objects: {e}")
                import traceback
                traceback.print_exc()
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
                return
        
        print()  # Newline after progress
        
        # Check if we got vectors
        valid_vectors_count = sum(1 for v in all_vectors if v is not None)
        
        # Ask about re-embedding if vectors are missing
        re_embed = False
        embedding_model_path = None
        
        if valid_vectors_count == 0:
            print(f"\n{Colors.YELLOW}No vectors found via pagination.{Colors.RESET}")
            print(f"\n{Colors.BOLD}Step 3: Vector generation{Colors.RESET}\n")
            re_embed = confirm("Re-embed text chunks to generate vectors for NPZ file?", True)
            
            if re_embed:
                embedding_model_path_str = get_input(
                    "Enter path to embedding model (or HuggingFace model name)",
                    "/home/linkages/cursor/pdftext/models/mxbai-embed-large-v1"
                )
                if embedding_model_path_str:
                    embedding_model_path = Path(embedding_model_path_str) if Path(embedding_model_path_str).exists() else embedding_model_path_str
        elif valid_vectors_count < len(all_vectors):
            print_warning(f"Only {valid_vectors_count}/{len(all_vectors)} vectors found. Some objects may be missing vectors.")
        
        print()  # Newline after progress
        
        if not all_objects:
            print_warning(f"No objects found in collection '{collection_name}'.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        print_success(f"Fetched {len(all_objects)} objects")
        
        # Prepare chunks for JSON and extract vectors
        print(f"\n{Colors.BLUE}Preparing data for export...{Colors.RESET}")
        chunks = []
        vectors_list = []
        
        for i, obj in enumerate(all_objects):
            props = obj.properties
            
            # Build chunk dict
            chunk = {
                "chunk_id": props.get("chunk_id", i + 1),
                "text": props.get("text", ""),
                "source_pdf": props.get("source_pdf", props.get("source", "unknown")),
                "page_start": props.get("page_start", props.get("page", 0)),
                "page_end": props.get("page_end", props.get("page", 0)),
                "token_count": props.get("token_count", 0),
            }
            
            # Add country if it exists
            if "country" in props:
                chunk["country"] = props["country"]
            
            chunks.append(chunk)
            
            # Vector should already be in all_vectors list from pagination fetch
            if i < len(all_vectors):
                vector = all_vectors[i]
            else:
                vector = None
            
            vectors_list.append(vector)
        
        # Save JSON file
        print(f"\n{Colors.BLUE}Saving JSON file to {json_path}...{Colors.RESET}")
        payload = {
            "sources": list(set(chunk.get("source_pdf", "unknown") for chunk in chunks)),
            "chunks": chunks,
            "chunk_count": len(chunks),
        }
        
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        print_success(f"Saved {len(chunks)} chunks to {json_path}")
        
        # Check if we have vectors from Weaviate
        valid_vectors = [v for v in vectors_list if v is not None]
        vectors_from_weaviate = len(valid_vectors) > 0
        
        if vectors_from_weaviate:
            print(f"\n{Colors.GREEN}Found {len(valid_vectors)}/{len(vectors_list)} vectors from Weaviate!{Colors.RESET}")
            print_info("Using vectors directly from Weaviate (no re-embedding needed).")
            
            # Save NPZ file with vectors from Weaviate
            print(f"\n{Colors.BLUE}Saving vectors to NPZ file...{Colors.RESET}")
            vectors_array = np.array(valid_vectors)
            np.savez_compressed(npz_path, vectors=vectors_array)
            print_success(f"Saved {vectors_array.shape[0]} vectors ({vectors_array.shape[1]} dimensions) to {npz_path}")
            
            if len(valid_vectors) < len(vectors_list):
                print_warning(f"Note: {len(vectors_list) - len(valid_vectors)} objects had no vectors (skipped)")
        elif re_embed and embedding_model_path:
            print(f"\n{Colors.YELLOW}No vectors found in Weaviate. Re-embedding text chunks...{Colors.RESET}")
            print(f"\n{Colors.BLUE}Re-embedding {len(chunks)} text chunks...{Colors.RESET}")
            print_info("This may take a few minutes depending on model size and number of chunks.")
            
            try:
                vectors_list = []
                
                # Try to use sentence-transformers first (simpler API)
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    print(f"  Loading embedding model: {embedding_model_path}...")
                    model = SentenceTransformer(str(embedding_model_path))
                    
                    # Extract texts
                    texts = [chunk["text"] for chunk in chunks]
                    
                    # Embed in batches
                    batch_size = 32
                    total = len(texts)
                    
                    for i in range(0, total, batch_size):
                        batch_texts = texts[i:i + batch_size]
                        batch_embeddings = model.encode(
                            batch_texts,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )
                        vectors_list.extend(batch_embeddings)
                        
                        percent = 100 * min(i + batch_size, total) / total
                        print(f"  Embedded {min(i + batch_size, total)}/{total} chunks ({percent:.1f}%)...", end="\r")
                    
                    print()  # Newline after progress
                    
                elif TRANSFORMERS_AVAILABLE:
                    print(f"  Loading embedding model: {embedding_model_path}...")
                    tokenizer = AutoTokenizer.from_pretrained(str(embedding_model_path))
                    model = AutoModel.from_pretrained(str(embedding_model_path))
                    model.eval()
                    
                    # Extract texts
                    texts = [chunk["text"] for chunk in chunks]
                    
                    # Embed in batches
                    batch_size = 16
                    total = len(texts)
                    
                    with torch.no_grad():
                        for i in range(0, total, batch_size):
                            batch_texts = texts[i:i + batch_size]
                            encoded = tokenizer(
                                batch_texts,
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                                max_length=512
                            )
                            
                            outputs = model(**encoded)
                            # Use mean pooling
                            embeddings = outputs.last_hidden_state.mean(dim=1)
                            # Normalize
                            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                            
                            vectors_list.extend(embeddings.cpu().numpy())
                            
                            percent = 100 * min(i + batch_size, total) / total
                            print(f"  Embedded {min(i + batch_size, total)}/{total} chunks ({percent:.1f}%)...", end="\r")
                    
                    print()  # Newline after progress
                    
                else:
                    print_error("No embedding library available (sentence-transformers or transformers)")
                    print_info("Install with: pip install sentence-transformers")
                    re_embed = False
                
                if vectors_list:
                    # Save NPZ file
                    print(f"\n{Colors.BLUE}Saving vectors to NPZ file...{Colors.RESET}")
                    vectors_array = np.array(vectors_list)
                    np.savez_compressed(npz_path, vectors=vectors_array)
                    print_success(f"Saved {vectors_array.shape[0]} vectors ({vectors_array.shape[1]} dimensions) to {npz_path}")
                
            except Exception as e:
                print_error(f"Error during re-embedding: {e}")
                import traceback
                traceback.print_exc()
                print_warning("Skipping NPZ file creation due to error.")
                npz_path = None
        else:
            print_info("Skipping vector generation.")
            npz_path = None
        
        # Summary
        print(f"\n{Colors.BOLD}{Colors.GREEN}Export Summary:{Colors.RESET}\n")
        print_key_value("Collection", collection_name)
        print_key_value("Objects exported", str(len(chunks)))
        print_key_value("JSON file", str(json_path))
        
        if npz_path and Path(npz_path).exists():
            print_key_value("NPZ file", f"{npz_path} {'(placeholder)' if not vectors_from_weaviate and re_embed else ''}")
        else:
            print_key_value("NPZ file", "Not created (vectors not available)")
        
        # Verify files
        if json_path.exists():
            json_size = json_path.stat().st_size / 1024  # KB
            print_key_value("JSON size", f"{json_size:.1f} KB")
        if npz_path and Path(npz_path).exists():
            npz_size = Path(npz_path).stat().st_size / 1024  # KB
            print_key_value("NPZ size", f"{npz_size:.1f} KB")
        
    except ValueError:
        print_error("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Export cancelled by user.{Colors.RESET}")
    except Exception as e:
        print_error(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
    
    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")


def import_collection_interactive(client: weaviate.WeaviateClient):
    """Interactive collection import from JSON and NPZ files."""
    clear_screen()
    print_header("Import Collection from JSON/NPZ Files")
    
    try:
        # Get JSON file path
        print(f"{Colors.BOLD}Step 1: Select JSON file{Colors.RESET}\n")
        json_path_str = get_input("Enter path to JSON file with chunks")
        if not json_path_str:
            print_warning("JSON file path is required.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        json_path = Path(json_path_str)
        if not json_path.exists():
            print_error(f"JSON file not found: {json_path}")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        # Get NPZ file path
        print(f"\n{Colors.BOLD}Step 2: Select NPZ file{Colors.RESET}\n")
        npz_path_str = get_input("Enter path to NPZ file with vectors")
        if not npz_path_str:
            print_warning("NPZ file path is required.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        npz_path = Path(npz_path_str)
        if not npz_path.exists():
            print_error(f"NPZ file not found: {npz_path}")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        # Get collection name
        print(f"\n{Colors.BOLD}Step 3: Collection name{Colors.RESET}\n")
        collection_name = get_input("Enter collection name", "DocumentChunk")
        if not collection_name:
            print_warning("Collection name is required.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        # Check if collection exists
        if client.collections.exists(collection_name):
            print_warning(f"Collection '{collection_name}' already exists.")
            if not confirm("Do you want to overwrite it? (existing data will be deleted)", False):
                print_info("Import cancelled.")
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
                return
            # Delete existing collection
            try:
                client.collections.delete(collection_name)
                print_success(f"Deleted existing collection '{collection_name}'")
            except Exception as e:
                print_error(f"Error deleting existing collection: {e}")
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
                return
        
        # Load files
        print(f"\n{Colors.BLUE}Loading files...{Colors.RESET}")
        print(f"  Loading vectors from {npz_path}...")
        vectors = load_vectors(npz_path)
        print_success(f"  Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
        
        print(f"  Loading chunks from {json_path}...")
        chunks = load_chunks(json_path)
        print_success(f"  Loaded {len(chunks)} chunks")
        
        if len(chunks) != vectors.shape[0]:
            print_error(f"Mismatch: {len(chunks)} chunks but {vectors.shape[0]} vectors")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
            return
        
        # Create collection schema
        print(f"\n{Colors.BLUE}Creating collection schema...{Colors.RESET}")
        create_collection_schema(client, collection_name)
        print_success(f"Collection '{collection_name}' created")
        
        # Import data
        print(f"\n{Colors.BLUE}Importing {len(chunks)} chunks...{Colors.RESET}")
        collection = client.collections.get(collection_name)
        batch_size = 100
        imported = 0
        
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            properties = {
                "chunk_id": chunk.get("chunk_id", i),
                "text": chunk.get("text", ""),
                "source_pdf": chunk.get("source_pdf", "unknown"),
                "country": extract_country(chunk.get("source_pdf")),
                "page_start": chunk.get("page_start", 0),
                "page_end": chunk.get("page_end", 0),
                "token_count": chunk.get("token_count", 0),
            }
            
            collection.data.insert(
                properties=properties,
                vector=vector.tolist(),
            )
            
            imported += 1
            if (i + 1) % batch_size == 0:
                percent = 100 * (i + 1) / len(chunks)
                print(f"  {Colors.CYAN}Progress: {i + 1}/{len(chunks)} ({percent:.1f}%){Colors.RESET}")
        
        print_success(f"\nSuccessfully imported {imported} chunks to collection '{collection_name}'")
        
        # Verify import
        if confirm("\nVerify import by showing sample objects?", True):
            result = collection.query.fetch_objects(limit=3)
            if result.objects:
                print(f"\n{Colors.BOLD}Sample objects:{Colors.RESET}\n")
                sample_items = []
                for obj in result.objects:
                    props = obj.properties
                    sample_text = f"Chunk ID: {props.get('chunk_id')}, "
                    sample_text += f"Country: {props.get('country')}, "
                    sample_text += f"Source: {props.get('source_pdf')}"
                    sample_items.append(sample_text)
                    text_preview = props.get('text', '')[:80]
                    sample_items.append(f"  {Colors.CYAN}Text preview:{Colors.RESET} {text_preview}...")
                
                print_list(sample_items)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Import cancelled by user.{Colors.RESET}")
    except Exception as e:
        print_error(f"Error during import: {e}")
        import traceback
        traceback.print_exc()
    
    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")


def show_main_menu(client: weaviate.WeaviateClient):
    """Display and handle main menu."""
    while True:
        clear_screen()
        print_header("Weaviate Collection Manager")
        
        collections = list_collections(client)
        collection_count = len(collections) if collections else 0
        
        print(f"{Colors.GREEN}Connected to Weaviate{Colors.RESET}")
        print(f"{Colors.BLUE}Collections: {collection_count}{Colors.RESET}\n")
        
        print(f"{Colors.BOLD}Main Menu:{Colors.RESET}\n")
        print(f"  {Colors.WHITE}1.{Colors.RESET} List all collections")
        print(f"  {Colors.WHITE}2.{Colors.RESET} Import collection from JSON/NPZ files")
        print(f"  {Colors.WHITE}3.{Colors.RESET} Save collection to JSON/NPZ files")
        print(f"  {Colors.WHITE}4.{Colors.RESET} Delete a collection")
        print(f"  {Colors.WHITE}5.{Colors.RESET} Delete multiple collections")
        print(f"  {Colors.WHITE}6.{Colors.RESET} Delete all collections")
        print(f"  {Colors.WHITE}0.{Colors.RESET} Exit\n")
        
        choice = get_input("Select an option", "0").strip()
        
        if choice == "1":
            display_collections(client)
        elif choice == "2":
            import_collection_interactive(client)
        elif choice == "3":
            save_collection_interactive(client)
        elif choice == "4":
            delete_collection_interactive(client)
        elif choice == "5":
            delete_multiple_collections(client)
        elif choice == "6":
            delete_all_collections(client)
        elif choice == "0":
            clear_screen()
            print(f"\n{Colors.GREEN}Thank you for using Weaviate Collection Manager!{Colors.RESET}\n")
            break
        else:
            print_error("Invalid option. Please try again.")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Graphical Text Interface for managing Weaviate collections."
    )
    parser.add_argument(
        "--weaviate-url",
        default="http://localhost:8080",
        help="Weaviate server URL (default: http://localhost:8080).",
    )
    args = parser.parse_args()
    
    # Connect to Weaviate using v4 API with context manager
    print(f"{Colors.CYAN}Connecting to Weaviate at {args.weaviate_url}...{Colors.RESET}")
    try:
        # Use Weaviate v4 API - use context manager properly
        if args.weaviate_url == "http://localhost:8080":
            # For local connection
            with weaviate.connect_to_local() as client:
                if not client.is_ready():
                    print_error("Weaviate is not ready")
                    sys.exit(1)
                
                print_success("Connected to Weaviate successfully!")
                
                # Show main menu - client stays open within context
                try:
                    show_main_menu(client)
                except KeyboardInterrupt:
                    clear_screen()
                    print(f"\n{Colors.YELLOW}Interrupted by user.{Colors.RESET}\n")
                except Exception as e:
                    print_error(f"Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # For remote connection, parse URL
            url_parts = args.weaviate_url.replace("http://", "").replace("https://", "").split(":")
            host = url_parts[0]
            port = int(url_parts[1]) if len(url_parts) > 1 else 8080
            is_secure = args.weaviate_url.startswith("https://")
            
            with weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=is_secure,
            ) as client:
                if not client.is_ready():
                    print_error("Weaviate is not ready")
                    sys.exit(1)
                
                print_success("Connected to Weaviate successfully!")
                
                # Show main menu - client stays open within context
                try:
                    show_main_menu(client)
                except KeyboardInterrupt:
                    clear_screen()
                    print(f"\n{Colors.YELLOW}Interrupted by user.{Colors.RESET}\n")
                except Exception as e:
                    print_error(f"Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()
                    
    except Exception as e:
        print_error(f"Error connecting to Weaviate: {e}")
        print(f"\n{Colors.YELLOW}Make sure Weaviate is running:{Colors.RESET}")
        print("  docker run -d -p 8080:8080 semitechnologies/weaviate:latest")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
