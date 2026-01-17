# Project Usage Report

## Overview
This report summarizes how to use the Python scripts in this workspace and what
each major function does. The focus is on the end-to-end flow:

1) Extract and clean PDF text, then chunk to JSON.
2) Embed chunks into vectors (saved as `.npz`).
3) Search vectors from CLI or from the Flask web UI.
4) Visualize chunk content and query results in the web UI.

## `pdf_to_text_chunker.py`

### Usage
Extract text from a PDF, clean it, chunk into token windows (default 300 tokens
with 15% overlap), and save JSON:

```
python3 pdf_to_text_chunker.py /path/to/input.pdf /path/to/out.json
```

Optional flags:
```
--max-tokens 300
```

### Key functions
- `iter_pdf_pages_text(pdf_path)`  
  Reads each PDF page with `PyPDF2` and yields `(page_index, page_text)`.

- `tokenize(text)`  
  Tokenizes by whitespace for deterministic, dependency-free counting.

- `is_page_number_line(line)`  
  Heuristic to detect page numbers (arabic or roman numerals) to remove.

- `extract_header_footer_candidates(pages)`  
  Builds sets of repeated first/last lines across pages to remove as
  headers/footers.

- `clean_page_text(text, headers, footers)`  
  Cleans a page by removing headers/footers, merging hyphenated words across
  line breaks, and normalizing whitespace.

- `chunk_tokens(tokens_with_pages, max_tokens, overlap_ratio)`  
  Chunks token streams into overlapping windows and tracks page ranges.

- `build_chunks_from_pdf(pdf_path, max_tokens)`  
  Orchestrates extraction, cleaning, tokenization, chunking, and prints cleaned
  text to console during processing.

## `embed_json.py`

### Usage
Embed the chunk JSON using `mxbai-embed-large-v1` and save vectors to `.npz`:

```
python3 embed_json.py /path/to/out.json /path/to/out_vectors.npz
```

Optional flags:
```
--model mxbai-embed-large-v1
--batch-size 8
--cache-dir /path/to/cache
```

### Key functions
- `mean_pooling(last_hidden_state, attention_mask)`  
  Averages token embeddings using the attention mask.

- `load_texts(json_path)`  
  Loads chunk text from the JSON file.

- `embed_texts(texts, model_name, batch_size, cache_dir)`  
  Runs the transformer model and returns a NumPy matrix of embeddings.

### Output format
Vectors are saved as compressed NumPy archive:
```
out_vectors.npz
  - vectors: float32 array [num_chunks, dim]
  - metadata: JSON string with model and source info
```

## `search_vectors.py`

### Usage
Embed a query locally and return the top matching chunks:

```
python3 search_vectors.py /path/to/out_vectors.npz /path/to/out.json "your query" --top-k 5
```

Optional flags:
```
--model-path /home/linkages/cursor/pdftext/models/mxbai-embed-large-v1
```

### Key functions
- `embed_query(query, model_path)`  
  Embeds the query using the local model.

- `cosine_sim(a, b)`  
  Cosine similarity between a matrix of vectors and a single query vector.

- `load_vectors(npz_path)` / `load_chunks(json_path)`  
  Loads vectors and chunks.

- `top_k(vectors, query_vec, k)`  
  Returns the top K most similar chunk indexes and scores.

## `app.py` + `templates/index.html`

### Usage
Run the Flask UI to view chunks and run searches:

```
python3 app.py --json /path/to/out.json --vectors /path/to/out_vectors.npz
```

Open: `http://127.0.0.1:5000/`

### Key functions
- `mean_pooling(...)`, `embed_query(...)`, `cosine_sim(...)`  
  Same embedding + similarity logic as CLI search.

- `create_app(default_json, default_vectors, model_path)`  
  Sets up the Flask app. The `/` route loads JSON, optionally embeds a query,
  and returns ranked results to the template.

### Template behavior
`templates/index.html` renders:
- A search form (`q`, `k`) for query input.
- The top-ranked search results (with score and chunk metadata).
- The full list of chunks for browsing.

## Recommended Workflow
1) Chunk the PDF:
```
python3 pdf_to_text_chunker.py input.pdf out.json
```
2) Embed the chunks:
```
python3 embed_json.py out.json out_vectors.npz --cache-dir ./model_cache
```
3) Search via CLI or web:
```
python3 search_vectors.py out_vectors.npz out.json "your query"
python3 app.py --json out.json --vectors out_vectors.npz
```
