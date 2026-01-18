# Tooling Report (Flow)

This report describes the tools in this workspace and how to use them in an
end‑to‑end flow: PDFs → cleaned chunks → embeddings → search/LLM.

## 1) Chunk PDFs into a collective JSON
**Tool:** `pdf_to_text_chunker.py`  
**What it does:** Extracts text from one or more PDFs, cleans it, chunks into
overlapping token windows, and appends into a single JSON file.

**Run:**
```
python3 pdf_to_text_chunker.py /path/to/combined.json /path/to/a.pdf /path/to/b.pdf
```

You can also pass a directory to process all `*.pdf` files inside:
```
python3 pdf_to_text_chunker.py /path/to/combined.json /path/to/pdf_dir
```

## 2) Embed the JSON into vectors
**Tool:** `embed_json.py`  
**What it does:** Embeds each chunk using `mxbai-embed-large-v1` and saves a
compressed `.npz` file with vectors.

**Run:**
```
python3 embed_json.py /path/to/combined.json /path/to/vectors.npz
```

## 3) Search vectors from CLI
**Tool:** `search_vectors.py`  
**What it does:** Embeds a query locally and returns the most similar chunks.

**Run:**
```
python3 search_vectors.py /path/to/vectors.npz /path/to/combined.json "your query" --top-k 5
```

## 4) Use the Flask UI (search + RAG)
**Tool:** `app.py`  
**What it does:** Web UI to search chunks and optionally generate a RAG answer
with the local Ollama model.

**Run:**
```
python3 app.py --json /path/to/combined.json --vectors /path/to/vectors.npz --ollama-model llama3.2
```

Then open: `http://127.0.0.1:5000/`

## 5) Use the Flask UI with vector tree (coarse‑to‑fine)
**Tool:** `app_tree.py`  
**What it does:** Web UI that uses the vector tree (document → section → chunk)
for coarse‑to‑fine search, then optionally generates a RAG answer with the local
Ollama model.

**Run:**
```
python3 app_tree.py --json /path/to/combined.json --vectors /path/to/vectors.npz --tree-json /path/to/tree.json --tree-vectors /path/to/tree_vectors.npz --ollama-model llama3.2
```

Open: `http://127.0.0.1:5001/`

## 6) Build a vector tree (optional)
**Tool:** `build_vector_tree.py`  
**What it does:** Creates a hierarchy: chunk → section → document using mean
pooled vectors.

**Run:**
```
python3 build_vector_tree.py /path/to/vectors.npz /path/to/combined.json /path/to/tree.json --section-size 5 --output-npz /path/to/tree_vectors.npz
```

## 7) Visualize vectors in 3D (optional)
**Tool:** `visualize_vectors_3d.py`  
**What it does:** PCA projection of vectors with clustering‑based colors.

**Run:**
```
python3 visualize_vectors_3d.py /path/to/vectors.npz --clusters 8 --sample 2000
```

## 8) Zip a folder (optional utility)
**Tool:** `folder_archive.py`  
**What it does:** Compress or decompress folders to ZIP.

**Run:**
```
python3 folder_archive.py compress /path/to/folder /path/to/output.zip
python3 folder_archive.py decompress /path/to/output.zip /path/to/output_folder
```
