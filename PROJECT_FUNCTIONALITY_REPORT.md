# PDFText RAG System - Project Functionality Report

**Date**: 2025-01-31  
**Environment**: Python 3.x + Weaviate + Ollama  
**Project**: RAG (Retrieval-Augmented Generation) System for PDF Document Analysis

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Main Applications](#main-applications)
3. [Utilities and Support Scripts](#utilities-and-support-scripts)
4. [Weaviate Management Programs](#weaviate-management-programs)
5. [Advanced Programs](#advanced-programs)
6. [Dependencies and Requirements](#dependencies-and-requirements)
7. [Docker Deployment](#docker-deployment)

---

## Project Overview

This project implements a complete RAG (Retrieval-Augmented Generation) system for analyzing and querying PDF documents through:
- **Text extraction and chunking** from PDFs
- **Vectorization** with embedding models (mxbai-embed-large-v1)
- **Weaviate vector database** for storage and search
- **Ollama LLM** for intelligent response generation
- **Knowledge Graph** for logical connections between chunks
- **Flask web interfaces** for user interaction

---

## Main Applications

### 1. `app_weaviate_rag.py`
**Description**: Main web application for multi-source RAG chatbot with Weaviate

**Features**:
- Flask web interface for interactive chat
- Multi-collection vector search on Weaviate
- Streaming responses from Ollama API
- Support for multiple collections (e.g., countrymodels, copd)
- Automatic filtering of `[country:]` prefix in results
- Chat history saved to JSON

**Usage**:
```bash
python3 app_weaviate_rag.py \
  --collections "countrymodels:cms" "copd:structdoc" \
  --model-path models/mxbai-embed-large-v1 \
  --ollama-model llama3.2 \
  --weaviate-url http://localhost:8080 \
  --chat-log weaviate_chat_history.json \
  --host 0.0.0.0 \
  --port 5003
```

**Parameters**:
- `--collections`: Weaviate collections in "name:alias" format
- `--model-path`: Path to local embedding model
- `--ollama-model`: Ollama model name (default: llama3.2)
- `--weaviate-url`: Weaviate server URL
- `--chat-log`: JSON file for chat history
- `--host/--port`: Flask server host and port

**Notes**:
- Requires Weaviate running on `localhost:8080` (or specified URL)
- Uses Ollama API at `http://0.0.0.0:11434/v1/chat/completions`
- Supports Docker with `--network host` for local service access

---

### 2. `app_weaviate_rag_graph.py`
**Description**: Advanced RAG chatbot with Knowledge Graph expansion

**Features**:
- Extends `app_weaviate_rag.py` with graph expansion
- Post-retrieval: expands results following graph connections
- Finds related chunks through semantic relationships
- Improves LLM context with connected information

**Usage**:
```bash
python3 app_weaviate_rag_graph.py \
  --collections "countrymodels:cms" "copd:structdoc" \
  --model-path models/mxbai-embed-large-v1 \
  --ollama-model llama3.2 \
  --weaviate-url http://localhost:8080 \
  --knowledge-graph knowledge_graph.json \
  --host 0.0.0.0 \
  --port 5004
```

**Additional Parameters**:
- `--knowledge-graph`: Path to knowledge graph JSON file
- `--expansion-hops`: Number of hops in the graph (default: 1-2)

**Advantages**:
- Richer context compared to standard RAG
- Discovers non-obvious connections between documents
- Useful for complex multi-document queries

---

### 3. `app.py`
**Description**: Flask web interface for viewing PDF chunks and vector search

**Features**:
- Structured visualization of text chunks extracted from PDFs
- Vector search by semantic similarity
- LLM queries for context-based answers
- Uses local JSON and NPZ files (not Weaviate)

**Usage**:
```bash
python app.py \
  --json out.json \
  --vectors out_vectors.npz \
  --model-path mxbai-embed-large-v1 \
  --ollama-model llama3.2 \
  --host 127.0.0.1 \
  --port 5000
```

**Parameters**:
- `--json`: JSON file with chunks (default: out.json)
- `--vectors`: NPZ file with vectors (default: out_vectors.npz)
- `--model-path`: Path to embedding model
- `--ollama-model`: Ollama model (default: llama3.2)
- `--host/--port`: Server host and port

**Notes**:
- Basic version without Weaviate
- Useful for local development and testing

---

### 4. `app_chat.py`
**Description**: Flask chatbot with UI interface and RAG capabilities

**Features**:
- Interactive chat with conversational memory
- RAG on local JSON/NPZ files
- Save chat history for analysis
- Responsive web UI

**Usage**:
```bash
python app_chat.py \
  --json out.json \
  --vectors out_vectors.npz \
  --model-path mxbai-embed-large-v1 \
  --ollama-model llama3.2 \
  --host 127.0.0.1 \
  --port 5001 \
  --chat-log chat_history.json
```

**Parameters**:
- `--json`: JSON with chunks (default: out.json)
- `--vectors`: NPZ vectors (default: out_vectors.npz)
- `--model-path`: Local model path (default: mxbai-embed-large-v1)
- `--ollama-model`: Ollama model (default: llama3.2)
- `--chat-log`: JSON file to save history (optional)

---

### 5. `app_multi_rag.py`
**Description**: Multi-source RAG chatbot with response streaming

**Features**:
- Support for multiple document sources
- Parallel search across multiple JSON/NPZ collections
- Real-time LLM response streaming
- Aggregation of results from different sources

**Usage**:
```bash
python app_multi_rag.py \
  --sources "cms:cms.json:cms.npz" "docs:docs.json:docs.npz" \
  --model-path mxbai-embed-large-v1 \
  --ollama-model llama3.2 \
  --chat-log multi_chat_history.json \
  --host 127.0.0.1 \
  --port 5002
```

**Parameters**:
- `--sources`: Source configurations in `name:json_path:npz_path` format (multiple)
- `--model-path`: Path to embedding model
- `--ollama-model`: Ollama model (default: llama3.2)
- `--chat-log`: Path to save chat history

**Notes**:
- Specify multiple sources for multi-document RAG
- Useful for querying heterogeneous datasets

---

### 6. `app_tree.py`
**Description**: Flask UI with coarse-to-fine hierarchical search via vector tree

**Features**:
- Vector search on tree structure (chunk → section → document)
- Fast coarse-level search (documents/sections)
- Fine-grained refinement on relevant chunks
- Performance optimization for large datasets

**Usage**:
```bash
python app_tree.py \
  --json out.json \
  --vectors out_vectors.npz \
  --tree-json tree.json \
  --tree-vectors tree_vectors.npz \
  --model-path mxbai-embed-large-v1 \
  --ollama-model llama3.2 \
  --host 127.0.0.1 \
  --port 5001
```

**Parameters**:
- `--json`: JSON with chunks
- `--vectors`: NPZ with chunk vectors
- `--tree-json`: JSON with tree structure
- `--tree-vectors`: NPZ with section/document vectors
- `--model-path`: Local model path
- `--ollama-model`: Ollama model (default: llama3.2)

**Notes**:
- Requires `build_vector_tree.py` to generate tree
- Efficient for large data volumes

---

## Utilities and Support Scripts

### 7. `pdf_to_text_chunker.py`
**Description**: Text extraction from PDFs and token-based chunking

**Features**:
- Extracts text from multiple PDF files or directories
- Visualizes page layouts
- Splits text into chunks with configurable token limit
- Exports structured JSON with metadata

**Usage**:
```bash
python pdf_to_text_chunker.py output.json path/to/pdfs/ --max-tokens 250
```

**Parameters**:
- `output_json`: Path to output JSON file
- `pdf_paths`: One or more PDF files and/or directories (positional)
- `--max-tokens`: Maximum tokens per chunk (default: 250)

**Output JSON**:
```json
[
  {
    "chunk_id": 0,
    "source_pdf": "document.pdf",
    "page": 1,
    "text": "chunk text...",
    "token_count": 245
  }
]
```

---

### 8. `embed_json.py`
**Description**: Vectorize text chunks from JSON using mxbai-embed-large-v1

**Features**:
- Load chunks from JSON
- Generate embeddings with local transformer model
- Save vectors in compressed NPZ format
- Batch processing for efficiency

**Usage**:
```bash
python embed_json.py input.json output_vectors.npz \
  --model mxbai-embed-large-v1 \
  --batch-size 8 \
  --cache-dir ./models
```

**Parameters**:
- `input_json`: JSON with chunks (positional)
- `output_vectors`: Output NPZ file (positional)
- `--model`: Model name or path (default: mxbai-embed-large-v1)
- `--batch-size`: Batch size for embedding (default: 8)
- `--cache-dir`: Directory for model cache/download

**Notes**:
- Generates embeddings of dimension 1024 (mxbai-embed-large-v1)
- Output: NPZ file with numpy array of vectors

---

### 9. `search_vectors.py`
**Description**: Search chunk vectors with text query

**Features**:
- Embed search query
- Calculate cosine similarity with chunk vectors
- Return top-k most relevant results
- Display chunks with scores

**Usage**:
```bash
python search_vectors.py vectors.npz chunks.json "search query" \
  --top-k 5 \
  --model-path mxbai-embed-large-v1
```

**Parameters**:
- `vectors_npz`: NPZ file with vectors (positional)
- `chunks_json`: JSON with chunks (positional)
- `query`: Query text (positional)
- `--top-k`: Number of top results (default: 5)
- `--model-path`: Local model path (default: mxbai-embed-large-v1)

**Output**:
```
Top 5 results:
1. [Score: 0.87] chunk_id=42, source=doc.pdf, page=3
   Text: "..."
```

---

### 10. `build_vector_tree.py`
**Description**: Build vector tree (chunk → section → document)

**Features**:
- Group chunks into sections (configurable)
- Create aggregated vectors for sections/documents
- Export JSON with hierarchical structure
- Save section/document vectors in NPZ

**Usage**:
```bash
python build_vector_tree.py vectors.npz chunks.json tree_output.json \
  --section-size 5 \
  --output-npz tree_vectors.npz
```

**Parameters**:
- `vectors_npz`: NPZ file with chunk vectors (positional)
- `chunks_json`: JSON with chunks (positional)
- `output_json`: Path to output tree JSON (positional)
- `--section-size`: Chunks per section (default: 5)
- `--output-npz`: Optional NPZ path for section/doc vectors

**Output**:
- JSON with structure: `{documents: [{sections: [{chunks: [...]}]}]}`
- NPZ with aggregated vectors (chunk averages)

---

### 11. `coords_finder.py`
**Description**: Find geographic coordinates in JSON (decimal or DMS formats)

**Features**:
- Scan JSON text for coordinate patterns
- Support formats: decimal (lat/lon) and DMS (degrees-minutes-seconds)
- Extract and validate coordinates
- Export JSON with found matches

**Usage**:
```bash
python coords_finder.py input.json output_coords.json
```

**Parameters**:
- `input_json`: JSON to analyze (positional)
- `output_json`: JSON with extracted coordinates (positional)

**Output JSON**:
```json
[
  {
    "chunk_id": 15,
    "source_pdf": "map.pdf",
    "coordinates": {
      "lat": 45.123,
      "lon": 12.456,
      "format": "decimal"
    }
  }
]
```

---

### 12. `visualize_vectors_3d.py`
**Description**: 3D visualization of vector embeddings with PCA

**Features**:
- Dimensional reduction with PCA (1024 → 3 dimensions)
- K-means clustering for coloring
- Interactive 3D plot (matplotlib or plotly)
- Sampling for large datasets

**Usage**:
```bash
python visualize_vectors_3d.py vectors.npz \
  --sample 2000 \
  --clusters 8
```

**Parameters**:
- `vectors_npz`: NPZ file with vectors (positional)
- `--sample`: Maximum points to plot (default: 2000)
- `--clusters`: Number of clusters for coloring (default: 8)

**Output**:
- Interactive 3D visualization
- Useful for exploratory embedding analysis

---

### 13. `downloadmodel.py`
**Description**: Download transformer models from HuggingFace

**Features**:
- Download pre-trained models (e.g., mxbai-embed-large-v1)
- Local cache in `./models/`
- Automatic management of tokenizer and model files

**Usage**:
```bash
python downloadmodel.py
```

**Notes**:
- Automatic download of mxbai-embed-large-v1 (default)
- Output directory: `./models/mxbai-embed-large-v1`
- Requires internet connection

---

### 14. `folder_archive.py`
**Description**: Compress/decompress folders via ZIP

**Features**:
- Compress folders to ZIP archives
- Decompress ZIP archives
- Support for backup and distribution

**Usage**:
```bash
# Compress folder
python folder_archive.py compress /path/to/folder output.zip

# Decompress archive
python folder_archive.py decompress archive.zip /path/to/output
```

**Commands**:
- `compress`: Compress a folder
- `decompress`: Decompress a ZIP file

---

## Weaviate Management Programs

### 15. `weaviate_collection_manager.py`
**Description**: Interactive TUI (Text User Interface) for Weaviate collection management

**Features**:
- Interactive menu with ANSI colors
- View existing collections with statistics
- Import collections from JSON+NPZ
- Remove collections
- Batch object management
- Support for embedding models

**Usage**:
```bash
python weaviate_collection_manager.py
```

**Menu Options**:
1. **List Collections**: List all collections with object count
2. **Import Collection**: Import from JSON+NPZ with auto-vectorization
3. **Remove Collection**: Delete collection (confirmation required)
4. **Exit**: Exit the program

**Internal Functions**:
- `extract_country()`: Extract country code from PDF name
- `batch_import()`: Optimized import with batches of 100 objects
- `connect_to_weaviate()`: Connection management (local/remote)

**Notes**:
- Requires Weaviate running
- Auto-detection of embedding models (sentence-transformers or transformers)

---

### 16. `import_to_weaviate.py`
**Description**: Batch import script from JSON+NPZ to Weaviate

**Features**:
- Massive import of chunks with vectors
- Automatic collection schema creation
- Optimized batch processing
- Automatic metadata extraction (country, source)

**Usage**:
```bash
python import_to_weaviate.py \
  --collection MyCollection \
  --json data.json \
  --vectors data.npz \
  --weaviate-url http://localhost:8080
```

**Parameters**:
- `--collection`: Weaviate collection name
- `--json`: JSON file with chunks
- `--vectors`: NPZ file with vectors
- `--weaviate-url`: Weaviate server URL

**Automatic Schema**:
- Properties: `text`, `source_pdf`, `page`, `chunk_id`, `country`, `token_count`
- Vectorizer: none (pre-generated vectors)
- Index: HNSW for efficient search

---

### 17. `query_weaviate.py`
**Description**: Interactive query for Weaviate collections

**Features**:
- Text query with vector search
- Metadata filters (country, source_pdf)
- Display results with scores
- Support for complex queries

**Usage**:
```bash
python query_weaviate.py \
  --collection MyCollection \
  --query "search this text" \
  --top-k 10 \
  --model-path models/mxbai-embed-large-v1
```

**Parameters**:
- `--collection`: Collection name to query
- `--query`: Query text
- `--top-k`: Number of results (default: 5)
- `--model-path`: Path to embedding model

---

### 18. `remove_weaviate_collections.py`
**Description**: Batch removal of Weaviate collections

**Features**:
- Delete specified collections
- Interactive confirmation
- Operation logging

**Usage**:
```bash
python remove_weaviate_collections.py Collection1 Collection2 Collection3
```

**Notes**:
- Irreversible operation (requires confirmation)
- Useful for development environment cleanup

---

### 19. `connect_weaviate.py`
**Description**: Weaviate connection test with connection method management

**Features**:
- Test connectivity to Weaviate server
- Auto-detection localhost vs remote
- Support for `connect_to_local()` and `connect_to_custom()`
- Diagnostics and troubleshooting

**Usage**:
```bash
python connect_weaviate.py
```

**Output**:
```
✓ Connected to Weaviate at http://localhost:8080
Available collections: ['Cms', 'DocumentChunk', 'Copd']
```

**Notes**:
- Requires Weaviate running
- Use to verify configuration before complex apps

---

## Advanced Programs

### 20. `knowledge_graph_builder.py`
**Description**: Build knowledge graphs between chunks from different collections

**Features**:
- Automatic entity extraction (acronyms, country codes, key terms)
- Link chunks through:
  - Common metadata (country, source_pdf)
  - Shared entities (NATO, COPD, JWC, etc.)
  - Vector semantic similarity (optional)
- Export graph in JSON and GraphML (for Gephi)
- Statistics on nodes/edges/communities

**Usage**:
```bash
# Basic: metadata and entities only
python knowledge_graph_builder.py \
  --collections Cms DocumentChunk \
  --export knowledge_graph.json

# Advanced: with vector similarity
python knowledge_graph_builder.py \
  --collections Cms DocumentChunk \
  --with-vectors \
  --similarity-threshold 0.75 \
  --sample 1000 \
  --export knowledge_graph.json \
  --graphml graph.graphml
```

**Parameters**:
- `--collections`: Weaviate collection names (multiple)
- `--weaviate-url`: Weaviate URL (default: http://localhost:8080)
- `--sample`: Limit to N objects per collection (0 = all)
- `--export`: JSON export path for graph
- `--graphml`: GraphML export for Gephi/Cytoscape
- `--with-vectors`: Enable vector similarity connections
- `--similarity-threshold`: Similarity threshold (0.0-1.0, default: 0.75)

**Extracted Entities**:
- Acronyms: NATO, COPD, JWC, CM, SHAPE, ACT, ACO
- Country codes: DUSHMAN, MURINUS, etc.
- Key terms: crisis, operation, planning, directive

**Output JSON**:
```json
{
  "nodes": [
    {
      "id": "Cms_uuid123",
      "collection": "Cms",
      "text": "...",
      "country": "DUSHMAN",
      "entities": ["NATO", "COPD"],
      "properties": {...}
    }
  ],
  "edges": [
    {
      "source": "Cms_uuid1",
      "target": "DocumentChunk_uuid2",
      "relationship": "same_country",
      "weight": 1.0
    }
  ],
  "statistics": {
    "total_nodes": 1500,
    "total_edges": 4200,
    "avg_degree": 2.8
  }
}
```

---

### 21. `visualize_knowledge_graph.py`
**Description**: Interactive/static visualization of Knowledge Graph

**Features**:
- 2D static visualization (matplotlib PNG)
- 2D interactive visualization (pyvis HTML)
- 3D static visualization (matplotlib PNG)
- 3D interactive visualization (plotly HTML)
- 3D window visualization (matplotlib interactive, no HTML)
- GraphML export for Gephi
- Sampling for large graphs (500+ nodes)
- Spring-based layout for optimal distribution

**Usage**:
```bash
# 2D static PNG
python visualize_knowledge_graph.py knowledge_graph.json \
  --output graph_2d.png \
  --sample 300 \
  --figsize 20,20

# 2D interactive HTML (pyvis)
python visualize_knowledge_graph.py knowledge_graph.json \
  --interactive \
  --output graph_interactive.html

# 3D interactive HTML (plotly)
python visualize_knowledge_graph.py knowledge_graph.json \
  --3d-html \
  --output graph_3d.html \
  --sample 200

# 3D interactive window (matplotlib)
python visualize_knowledge_graph.py knowledge_graph.json \
  --3d-window \
  --sample 150

# Export GraphML for Gephi
python visualize_knowledge_graph.py knowledge_graph.json \
  --graphml graph_for_gephi.graphml
```

**Parameters**:
- `graph_json`: Graph JSON file (positional)
- `--sample`: Node limit (0 = all, recommended 150-300 for large graphs)
- `--output`: Output file (.png or .html)
- `--interactive`: Generate interactive HTML (pyvis)
- `--3d`: 3D static plot (matplotlib PNG)
- `--3d-window`: 3D interactive window (matplotlib, no HTML)
- `--3d-html`: 3D interactive HTML (plotly)
- `--graphml`: GraphML export for Gephi/Cytoscape
- `--figsize`: Figure dimensions for static plots (width,height)

**Node Colors**:
- By collection: different colors for each collection
- By country: colors for country codes
- By community: community detection with Louvain algorithm

**Notes**:
- For graphs > 500 nodes, use `--sample` for performance
- Pyvis and Plotly generate navigable interactive HTML
- Gephi offers advanced analysis (centrality, modularity)

---

### 22. `rag_integrity_check.py`
**Description**: Verify integrity and logical consistency of RAG data from Weaviate

**Features**:
- Check duplicates (identical texts)
- Validate metadata (country, source_pdf, page)
- Verify vector consistency (dimension, anomalous values)
- Analyze text length distribution
- Vector search testing (example queries)
- Detailed report with statistics

**Usage**:
```bash
python rag_integrity_check.py \
  --collections Cms DocumentChunk \
  --weaviate-url http://localhost:8080 \
  --report integrity_report.txt
```

**Parameters**:
- `--collections`: Collections to verify
- `--weaviate-url`: Weaviate URL
- `--report`: Output file for report

**Checks Performed**:
1. **Duplicates**: Identify identical or very similar texts
2. **Metadata**: Verify required fields and formats
3. **Vectors**: Uniform dimension, NaN/Inf values
4. **Text Length**: Min/max/avg characters, outliers
5. **Search**: Test vector queries to verify functionality

**Output Report**:
```
=== RAG Integrity Report ===
Collection: Cms
Total Objects: 1500
Duplicates Found: 3 (0.2%)
Missing Metadata: 0
Vector Issues: 0
Text Length Stats:
  Min: 50 chars
  Max: 1200 chars
  Avg: 450 chars
  Outliers: 5 (very short/long)
Search Test: PASSED
```

---

### 23. `rag_analyzer.py`
**Description**: Analyze RAG system performance from chat history

**Features**:
- Parse chat history JSON
- Calculate metrics:
  - Average/median response time
  - Retrieval precision (if ground truth available)
  - Token usage per query
  - Source coverage
- Identify problematic queries
- Export text and JSON reports

**Usage**:
```bash
python rag_analyzer.py chat_history.json \
  --output rag_analysis_report.txt \
  --json-output rag_analysis.json
```

**Parameters**:
- `chat_log`: Chat history JSON file (positional)
- `--output`: Text report file (default: rag_analysis_report.txt)
- `--json-output`: JSON file with structured data (optional)

**Calculated Metrics**:
- **Latency**: Response time per query
- **Retrieval Quality**: Average score of retrieved chunks
- **Source Coverage**: Distribution of sources used
- **Query Complexity**: Query length and complexity
- **Error Rate**: Frequency of errors/failures

**Output**:
```
=== RAG Performance Analysis ===
Total Queries: 150
Avg Response Time: 2.3s
Median Response Time: 1.8s
Avg Chunks Retrieved: 5.2
Avg Retrieval Score: 0.78
Top Sources:
  - Cms: 65%
  - DocumentChunk: 35%
Problematic Queries: 3 (2%)
  - "complex query..." (timeout)
```

---

### 24. `rag_research_agent.py`
**Description**: Autonomous research agent to generate PDF reports from RAG

**Features**:
- Autonomous RAG query for research topics
- Multi-step retrieval with score filtering
- Structured report generation with LLM
- PDF export with professional formatting
- Automatic source citations

**Usage**:
```bash
python rag_research_agent.py "Analyze crisis response in NATO COPD" \
  --json cms.json \
  --vectors cms.npz \
  --model-path models/mxbai-embed-large-v1 \
  --ollama-model llama3.2 \
  --output research_report.pdf \
  --max-chunks 20 \
  --score-threshold 0.7
```

**Parameters**:
- `research_query`: Research question/topic (positional)
- `--json`: JSON file with chunks (default: cms.json)
- `--vectors`: NPZ file with vectors (default: cms.npz)
- `--model-path`: Path to embedding model
- `--ollama-model`: Ollama model (default: llama3.2)
- `--output`: Output PDF file (default: research_report_<timestamp>.pdf)
- `--max-chunks`: Maximum chunks to use (default: 20)
- `--score-threshold`: Minimum score threshold (default: 0.7)

**Pipeline**:
1. **Query Expansion**: Generate search query variants
2. **Retrieval**: Search relevant chunks with score filtering
3. **Ranking**: Order by relevance and diversity
4. **Generation**: LLM generates structured report
5. **PDF Export**: Professional formatting with citations

**Output PDF**:
```
Research Report: [Topic]
Generated: [Date]

Executive Summary
... summary ...

Findings
1. [Finding 1]
   Sources: [pdf1.pdf, p.5], [pdf2.pdf, p.12]
   
2. [Finding 2]
   Sources: ...

Recommendations
...

References
[1] pdf1.pdf, page 5: "quote..."
```

---

### 25. `langgraph_task_agent.py`
**Description**: LangGraph agent for task definition and execution with RAG

**Features**:
- Define complex tasks with LangGraph
- Integration with Weaviate for retrieval
- Multi-step execution with state management
- Export results to JSON and Markdown
- Support for iterative and conditional tasks

**Usage**:
```bash
python langgraph_task_agent.py \
  --task "Analyze crisis relationship between MURINUS and DUSHMAN" \
  --collections Cms DocumentChunk \
  --ollama-model llama3.2 \
  --output tasks.json \
  --markdown-report report.md
```

**Parameters**:
- `--task`: Task description to execute
- `--collections`: Weaviate collections for retrieval
- `--ollama-model`: Ollama model
- `--output`: JSON results file
- `--markdown-report`: Markdown report (optional)

**Notes**:
- Deprecation warning: Update to `langchain-ollama`
- JSON parsing error: Escape special characters in prompts
- `markdown_report` variable undefined: Fix in development

---

### 26. `langchain_pdf_to_weaviate.py`
**Description**: Import PDFs to Weaviate using LangChain

**Features**:
- PDF loading with LangChain loaders
- Text splitting with LangChain splitters
- Automatic embedding
- Batch import to Weaviate
- LangChain-Weaviate integration

**Usage**:
```bash
python langchain_pdf_to_weaviate.py \
  --pdf-dir /path/to/pdfs \
  --collection MyDocs \
  --chunk-size 500 \
  --chunk-overlap 50
```

**Notes**:
- Requires: `pip install langchain langchain-community langchain-text-splitters`
- Alternative to `pdf_to_text_chunker.py` with LangChain

---

### 27. `langchain_rag_chatbot.py`
**Description**: LangChain-based RAG chatbot

**Features**:
- RAG pipeline with LangChain
- Retrieval from Weaviate via LangChain
- Conversational memory
- LLM integration (Ollama/OpenAI)

**Usage**:
```bash
python langchain_rag_chatbot.py \
  --collection MyDocs \
  --ollama-model llama3.2
```

**Notes**:
- `Embeddings` undefined error: Fix import `from langchain.embeddings import Embeddings`

---

### 28. `ollama_api.py`
**Description**: API module for interfacing with Ollama server

**Features**:
- `call_ollama_api()`: Synchronous call to Ollama
- `stream_ollama_api()`: Response streaming
- Endpoint management `/v1/chat/completions`
- Error handling and retry logic

**Usage**:
```python
from ollama_api import call_ollama_api, stream_ollama_api

# Synchronous call
response = call_ollama_api(
    prompt="Hello!",
    model="llama3.2",
    base_url="http://0.0.0.0:11434"
)

# Streaming
for chunk in stream_ollama_api(prompt, model, base_url):
    print(chunk, end='', flush=True)
```

**Notes**:
- Used by `app_weaviate_rag.py` and other chatbots
- Endpoint: `http://0.0.0.0:11434/v1/chat/completions`

---

## Dependencies and Requirements

### Main Python Packages
```
# Core
python>=3.8
numpy>=1.20.0
flask>=2.0.0

# Vector DB
weaviate-client>=4.0.0

# Embeddings
transformers>=4.30.0
sentence-transformers>=2.2.0
torch>=2.0.0

# Graphs
networkx>=3.0
matplotlib>=3.5.0
pyvis>=0.3.0
plotly>=5.0.0

# LangChain (optional)
langchain>=0.1.0
langchain-community>=0.1.0
langchain-text-splitters>=0.1.0
langchain-ollama>=0.1.0
langchain-huggingface>=0.0.1

# PDF Processing
PyPDF2>=3.0.0
pdfplumber>=0.9.0

# Others
scikit-learn>=1.0.0
tiktoken>=0.5.0
```

### External Services
1. **Weaviate**: Vector database
   - URL: `http://localhost:8080`
   - Installation: Docker or standalone
   
2. **Ollama**: LLM server
   - URL: `http://0.0.0.0:11434`
   - Default model: `llama3.2`
   - Installation: `curl https://ollama.ai/install.sh | sh`

### Embedding Models
- **mxbai-embed-large-v1**: 1024 dimensions
  - Path: `./models/mxbai-embed-large-v1`
  - Download: `python downloadmodel.py`

---

## Docker Deployment

### Main Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 5003 5004 5005

CMD ["python3", "app_weaviate_rag.py"]
```

### Build and Run
```bash
# Build image
docker build -t pdftext-rag:latest .

# Run with host network (for access to local Weaviate/Ollama)
docker run --rm -it \
  --network host \
  -v "$(pwd)":/app \
  pdftext-rag:latest \
  python3 app_weaviate_rag.py \
    --collections "countrymodels:cms" "copd:structdoc" \
    --model-path models/mxbai-embed-large-v1 \
    --ollama-model llama3.2 \
    --weaviate-url http://127.0.0.1:8080 \
    --host 0.0.0.0 \
    --port 5003

# Run with port mapping (if Weaviate is in another container)
docker run --rm -it \
  -p 5003:5003 \
  -v "$(pwd)":/app \
  pdftext-rag:latest \
  python3 app_weaviate_rag.py \
    --weaviate-url http://host.docker.internal:8080 \
    ...
```

### Docker Compose (optional)
```yaml
version: '3.8'

services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 20
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  pdftext-rag:
    build: .
    ports:
      - "5003:5003"
    volumes:
      - ./:/app
    depends_on:
      - weaviate
      - ollama
    environment:
      WEAVIATE_URL: http://weaviate:8080
      OLLAMA_URL: http://ollama:11434

volumes:
  ollama_data:
```

### Docker Notes
- **Host network**: Required for access to services on localhost (Weaviate, Ollama) when external to container
- **Port mapping**: Use `-p 5003:5003` to expose Flask app
- **Volume mount**: `-v "$(pwd)":/app` for access to local models and data
- **host.docker.internal**: Use to refer to host machine's localhost from inside container

---

## Common Troubleshooting

### Weaviate Errors

1. **`TypeError: Client.__init__() got an unexpected keyword argument 'url'`**
   - Cause: Weaviate client 4.x changes API
   - Solution: Use `weaviate.connect_to_local()` or `weaviate.connect_to_custom()`
   
2. **`WeaviateQueryError: vector lengths don't match: 384 vs 1024`**
   - Cause: Vector dimension mismatch between query and index
   - Solution: Verify embedding model (mxbai = 1024 dim)

3. **`Connection refused to http://0.0.0.0:8080`**
   - Cause: Weaviate not running
   - Solution: Start Weaviate: `docker run -p 8080:8080 semitechnologies/weaviate`

### LLM Errors

1. **`Error calling Ollama API: Expecting value: line 1 column 1 (char 0)`**
   - Cause: Ollama server not responding or wrong response format
   - Solution: Verify `curl http://0.0.0.0:11434/v1/models`

2. **`LangChainDeprecationWarning: The class Ollama was deprecated`**
   - Cause: Old langchain version
   - Solution: `pip install -U langchain-ollama` and change imports

### Embedding Errors

1. **`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`**
   - Cause: Local path not recognized as repo name
   - Solution: Use HF model name or verify path exists

2. **`ModuleNotFoundError: No module named 'sentence_transformers'`**
   - Cause: Library not installed
   - Solution: `pip install sentence-transformers`

### Docker Errors

1. **Container cannot connect to localhost**
   - Cause: Docker networking isolation
   - Solution: Use `--network host` or `http://host.docker.internal:PORT`

2. **`FileNotFoundError: models/mxbai-embed-large-v1`**
   - Cause: Model not mounted in volume
   - Solution: Add `-v "$(pwd)/models:/app/models"` or download in container

---

## Best Practices

### Performance
1. **Batch Processing**: Use batches of 50-100 objects for Weaviate imports
2. **Sampling**: Limit to 200-500 nodes for large graph visualizations
3. **Caching**: Reuse pre-calculated embeddings (NPZ files)
4. **Vector Tree**: Use for datasets > 10K chunks

### Security
1. **API Keys**: Don't commit credentials (use `.env`)
2. **Network**: Limit Weaviate/Ollama access with firewall
3. **Input Validation**: Sanitize user input before LLM queries

### Maintenance
1. **Logging**: Enable chat logs for performance analysis
2. **Monitoring**: Check latency and error rate with `rag_analyzer.py`
3. **Integrity Checks**: Run `rag_integrity_check.py` periodically
4. **Backup**: Archive Weaviate collections regularly

---

## Roadmap and Future Development

### In Development
- [ ] Fix LangChain deprecation warnings
- [ ] Complete migration to weaviate-client 4.x
- [ ] Multi-language embedding support
- [ ] Improved web UI with Markdown rendering

### Planned
- [ ] Graph-RAG with reranking
- [ ] Custom embedding model fine-tuning
- [ ] Support for multimodal PDFs (images, tables)
- [ ] RESTful API for external integration
- [ ] Real-time analytics dashboard

---

## Conclusions

This project provides a complete ecosystem for RAG on PDF documents:
- **Complete Pipeline**: From PDF to intelligent chatbot
- **Flexibility**: Support for local files (JSON/NPZ) and Weaviate
- **Scalability**: Optimizations for large datasets
- **Visualization**: Interactive 2D/3D knowledge graphs
- **Production-ready**: Docker, logging, integrity checks

For questions or contributions, consult individual script documentation with `--help`.

---

**Report generated**: 2025-01-31  
**Project version**: 1.0  
**Author**: AI Assistant
