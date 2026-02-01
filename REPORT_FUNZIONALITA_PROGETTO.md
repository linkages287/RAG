# Report Funzionalità Progetto PDFText RAG System

**Data**: 2025-01-31  
**Ambiente**: Python 3.x + Weaviate + Ollama  
**Progetto**: Sistema RAG (Retrieval-Augmented Generation) per l'analisi di documenti PDF

---

## Indice
1. [Panoramica del Progetto](#panoramica-del-progetto)
2. [Programmi Principali](#programmi-principali)
3. [Utilità e Script di Supporto](#utilità-e-script-di-supporto)
4. [Programmi di Gestione Weaviate](#programmi-di-gestione-weaviate)
5. [Dipendenze e Requisiti](#dipendenze-e-requisiti)
6. [Deployment con Docker](#deployment-con-docker)

---

## Panoramica del Progetto

Questo progetto implementa un sistema completo di RAG (Retrieval-Augmented Generation) per l'analisi e l'interrogazione di documenti PDF tramite:
- **Estrazione e chunking** di testo da PDF
- **Vectorizzazione** con embedding models (mxbai-embed-large-v1)
- **Database vettoriale Weaviate** per storage e ricerca
- **LLM Ollama** per generazione di risposte intelligenti
- **Knowledge Graph** per collegamenti logici tra chunks
- **Interfacce web Flask** per interazione utente

---

## Programmi Principali

### 1. `app_weaviate_rag.py`
**Descrizione**: Applicazione web principale per chatbot RAG multi-sorgente con Weaviate

**Funzionalità**:
- Interfaccia web Flask per chat interattiva
- Ricerca vettoriale multi-collection su Weaviate
- Streaming di risposte da Ollama API
- Supporto per collezioni multiple (es: countrymodels, copd)
- Filtraggio automatico del prefisso `[country:]` nei risultati
- Salvataggio cronologia chat in JSON

**Utilizzo**:
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

**Parametri**:
- `--collections`: Collezioni Weaviate nel formato "nome:alias"
- `--model-path`: Path al modello di embedding locale
- `--ollama-model`: Nome del modello Ollama (default: llama3.2)
- `--weaviate-url`: URL del server Weaviate
- `--chat-log`: File JSON per cronologia chat
- `--host/--port`: Host e porta del server Flask

**Note**:
- Richiede Weaviate running su `localhost:8080` (o URL specificato)
- Usa API Ollama su `http://0.0.0.0:11434/v1/chat/completions`
- Supporta Docker con `--network host` per accesso a servizi locali

---

### 2. `app_weaviate_rag_graph.py`
**Descrizione**: Chatbot RAG avanzato con espansione tramite Knowledge Graph

**Funzionalità**:
- Estende `app_weaviate_rag.py` con graph expansion
- Post-retrieval: espande risultati seguendo collegamenti del grafo
- Trova chunk correlati attraverso relazioni semantiche
- Migliora contesto per LLM con informazioni collegate

**Utilizzo**:
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

**Parametri Aggiuntivi**:
- `--knowledge-graph`: Path al file JSON del grafo di conoscenza
- `--expansion-hops`: Numero di salti nel grafo (default: 1-2)

**Vantaggi**:
- Contesto più ricco rispetto a RAG standard
- Scopre connessioni non ovvie tra documenti
- Utile per query complesse multi-documento

---

### 3. `app.py`
**Descrizione**: Interfaccia web Flask per visualizzare chunks PDF e ricerca vettoriale

**Funzionalità**:
- Visualizzazione strutturata di chunk testuali estratti da PDF
- Ricerca vettoriale per similarità semantica
- Query LLM per risposte basate su contesto
- Usa file JSON e NPZ locali (non Weaviate)

**Utilizzo**:
```bash
python app.py \
  --json out.json \
  --vectors out_vectors.npz \
  --model-path mxbai-embed-large-v1 \
  --ollama-model llama3.2 \
  --host 127.0.0.1 \
  --port 5000
```

**Parametri**:
- `--json`: File JSON con chunks (default: out.json)
- `--vectors`: File NPZ con vettori (default: out_vectors.npz)
- `--model-path`: Path al modello di embedding
- `--ollama-model`: Modello Ollama (default: llama3.2)
- `--host/--port`: Host e porta del server

**Note**:
- Versione base senza Weaviate
- Utile per sviluppo e test locali

---

### 4. `app_chat.py`
**Descrizione**: Chatbot Flask con interfaccia UI e capacità RAG

**Funzionalità**:
- Chat interattiva con memoria conversazionale
- RAG su file JSON/NPZ locali
- Salvataggio cronologia chat per analisi
- UI web responsive

**Utilizzo**:
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

**Parametri**:
- `--json`: JSON con chunks (default: out.json)
- `--vectors`: Vettori NPZ (default: out_vectors.npz)
- `--model-path`: Path al modello locale (default: mxbai-embed-large-v1)
- `--ollama-model`: Modello Ollama (default: llama3.2)
- `--chat-log`: File JSON per salvare cronologia (opzionale)

---

### 5. `app_multi_rag.py`
**Descrizione**: Chatbot RAG multi-sorgente con streaming di risposte

**Funzionalità**:
- Supporto per sorgenti multiple di documenti
- Ricerca parallela su più collezioni JSON/NPZ
- Streaming delle risposte LLM in tempo reale
- Aggregazione di risultati da fonti diverse

**Utilizzo**:
```bash
python app_multi_rag.py \
  --sources "cms:cms.json:cms.npz" "docs:docs.json:docs.npz" \
  --model-path mxbai-embed-large-v1 \
  --ollama-model llama3.2 \
  --chat-log multi_chat_history.json \
  --host 127.0.0.1 \
  --port 5002
```

**Parametri**:
- `--sources`: Configurazioni sorgenti nel formato `nome:json_path:npz_path` (multipli)
- `--model-path`: Path al modello di embedding
- `--ollama-model`: Modello Ollama (default: llama3.2)
- `--chat-log`: Path per salvare cronologia chat

**Note**:
- Specifica più sorgenti per RAG multi-documento
- Utile per interrogare dataset eterogenei

---

### 6. `app_tree.py`
**Descrizione**: UI Flask con ricerca gerarchica coarse-to-fine tramite albero vettoriale

**Funzionalità**:
- Ricerca vettoriale su struttura ad albero (chunk → section → document)
- Ricerca rapida a livello coarse (documenti/sezioni)
- Raffinamento fine-grained sui chunk rilevanti
- Ottimizzazione performance per dataset grandi

**Utilizzo**:
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

**Parametri**:
- `--json`: JSON con chunks
- `--vectors`: NPZ con vettori chunk
- `--tree-json`: JSON con struttura ad albero
- `--tree-vectors`: NPZ con vettori sezioni/documenti
- `--model-path`: Path al modello locale
- `--ollama-model`: Modello Ollama (default: llama3.2)

**Note**:
- Richiede `build_vector_tree.py` per generare albero
- Efficiente per grandi volumi di dati

---

## Utilità e Script di Supporto

### 7. `pdf_to_text_chunker.py`
**Descrizione**: Estrazione testo da PDF e chunking basato su token

**Funzionalità**:
- Estrae testo da file PDF multipli o directory
- Visualizza layout delle pagine
- Divide testo in chunks con limite token configurabile
- Esporta JSON strutturato con metadati

**Utilizzo**:
```bash
python pdf_to_text_chunker.py output.json path/to/pdfs/ --max-tokens 250
```

**Parametri**:
- `output_json`: Path al file JSON di output
- `pdf_paths`: Uno o più file PDF e/o directory (posizionale)
- `--max-tokens`: Massimo token per chunk (default: 250)

**Output JSON**:
```json
[
  {
    "chunk_id": 0,
    "source_pdf": "documento.pdf",
    "page": 1,
    "text": "testo del chunk...",
    "token_count": 245
  }
]
```

---

### 8. `embed_json.py`
**Descrizione**: Vectorizza chunk testuali da JSON usando mxbai-embed-large-v1

**Funzionalità**:
- Carica chunks da JSON
- Genera embeddings con modello transformer locale
- Salva vettori in formato NPZ compresso
- Batch processing per efficienza

**Utilizzo**:
```bash
python embed_json.py input.json output_vectors.npz \
  --model mxbai-embed-large-v1 \
  --batch-size 8 \
  --cache-dir ./models
```

**Parametri**:
- `input_json`: JSON con chunks (posizionale)
- `output_vectors`: File NPZ di output (posizionale)
- `--model`: Nome o path del modello (default: mxbai-embed-large-v1)
- `--batch-size`: Dimensione batch per embedding (default: 8)
- `--cache-dir`: Directory per cache/download del modello

**Note**:
- Genera embeddings di dimensione 1024 (mxbai-embed-large-v1)
- Output: file NPZ con array numpy dei vettori

---

### 9. `search_vectors.py`
**Descrizione**: Ricerca nei vettori chunk tramite query testuale

**Funzionalità**:
- Embedding della query di ricerca
- Calcolo similarità coseno con vettori chunk
- Restituzione top-k risultati più rilevanti
- Visualizzazione chunk con score

**Utilizzo**:
```bash
python search_vectors.py vectors.npz chunks.json "query di ricerca" \
  --top-k 5 \
  --model-path mxbai-embed-large-v1
```

**Parametri**:
- `vectors_npz`: File NPZ con vettori (posizionale)
- `chunks_json`: JSON con chunks (posizionale)
- `query`: Testo della query (posizionale)
- `--top-k`: Numero di risultati top (default: 5)
- `--model-path`: Path al modello locale (default: mxbai-embed-large-v1)

**Output**:
```
Top 5 results:
1. [Score: 0.87] chunk_id=42, source=doc.pdf, page=3
   Text: "..."
```

---

### 10. `build_vector_tree.py`
**Descrizione**: Costruisce albero vettoriale (chunk → section → document)

**Funzionalità**:
- Raggruppa chunks in sezioni (configurabile)
- Crea vettori aggregati per sezioni/documenti
- Esporta JSON con struttura gerarchica
- Salva vettori sezioni/documenti in NPZ

**Utilizzo**:
```bash
python build_vector_tree.py vectors.npz chunks.json tree_output.json \
  --section-size 5 \
  --output-npz tree_vectors.npz
```

**Parametri**:
- `vectors_npz`: File NPZ con vettori chunk (posizionale)
- `chunks_json`: JSON con chunks (posizionale)
- `output_json`: Path al JSON albero di output (posizionale)
- `--section-size`: Chunks per sezione (default: 5)
- `--output-npz`: Path NPZ opzionale per vettori sezione/doc

**Output**:
- JSON con struttura: `{documents: [{sections: [{chunks: [...]}]}]}`
- NPZ con vettori aggregati (media dei chunk)

---

### 11. `coords_finder.py`
**Descrizione**: Trova coordinate geografiche in JSON (formati decimali o DMS)

**Funzionalità**:
- Scansione testo JSON per pattern di coordinate
- Supporto formati: decimali (lat/lon) e DMS (gradi-minuti-secondi)
- Estrazione e validazione coordinate
- Export JSON con match trovati

**Utilizzo**:
```bash
python coords_finder.py input.json output_coords.json
```

**Parametri**:
- `input_json`: JSON da analizzare (posizionale)
- `output_json`: JSON con coordinate estratte (posizionale)

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
**Descrizione**: Visualizzazione 3D di embeddings vettoriali con PCA

**Funzionalità**:
- Riduzione dimensionale con PCA (1024 → 3 dimensioni)
- Clustering k-means per colorazione
- Plot 3D interattivo (matplotlib o plotly)
- Campionamento per grandi dataset

**Utilizzo**:
```bash
python visualize_vectors_3d.py vectors.npz \
  --sample 2000 \
  --clusters 8
```

**Parametri**:
- `vectors_npz`: File NPZ con vettori (posizionale)
- `--sample`: Numero massimo di punti da plottare (default: 2000)
- `--clusters`: Numero di cluster per colorazione (default: 8)

**Output**:
- Visualizzazione 3D interattiva
- Utile per analisi esplorativa embeddings

---

### 13. `downloadmodel.py`
**Descrizione**: Download di modelli transformer da HuggingFace

**Funzionalità**:
- Scarica modelli pre-trained (es: mxbai-embed-large-v1)
- Cache locale in `./models/`
- Gestione automatica di tokenizer e model files

**Utilizzo**:
```bash
python downloadmodel.py
```

**Note**:
- Download automatico di mxbai-embed-large-v1 (default)
- Output directory: `./models/mxbai-embed-large-v1`
- Richiede connessione internet

---

### 14. `folder_archive.py`
**Descrizione**: Compressione/decompressione di cartelle tramite ZIP

**Funzionalità**:
- Compressione cartelle in archivi ZIP
- Decompressione archivi ZIP
- Supporto per backup e distribuzione

**Utilizzo**:
```bash
# Comprimi cartella
python folder_archive.py compress /path/to/folder output.zip

# Decomprimi archivio
python folder_archive.py decompress archive.zip /path/to/output
```

**Comandi**:
- `compress`: Comprimi una cartella
- `decompress`: Decomprimi un file ZIP

---

## Programmi di Gestione Weaviate

### 15. `weaviate_collection_manager.py`
**Descrizione**: TUI (Text User Interface) interattiva per gestione collezioni Weaviate

**Funzionalità**:
- Menu interattivo con colori ANSI
- Visualizzazione collezioni esistenti con statistiche
- Import di collezioni da JSON+NPZ
- Rimozione collezioni
- Gestione batch di oggetti
- Supporto per embedding models

**Utilizzo**:
```bash
python weaviate_collection_manager.py
```

**Menu Opzioni**:
1. **List Collections**: Elenca tutte le collezioni con conteggio oggetti
2. **Import Collection**: Importa da JSON+NPZ con auto-vectorizzazione
3. **Remove Collection**: Elimina collezione (conferma richiesta)
4. **Exit**: Esci dal programma

**Funzioni Interne**:
- `extract_country()`: Estrae codice paese da nome PDF
- `batch_import()`: Import ottimizzato con batch di 100 oggetti
- `connect_to_weaviate()`: Gestione connessione (locale/remota)

**Note**:
- Richiede Weaviate in esecuzione
- Auto-detection di modelli embedding (sentence-transformers o transformers)

---

### 16. `import_to_weaviate.py`
**Descrizione**: Script di import batch da JSON+NPZ a Weaviate

**Funzionalità**:
- Import massivo di chunks con vettori
- Creazione automatica schema collezione
- Batch processing ottimizzato
- Estrazione metadati automatici (country, source)

**Utilizzo**:
```bash
python import_to_weaviate.py \
  --collection MyCollection \
  --json data.json \
  --vectors data.npz \
  --weaviate-url http://localhost:8080
```

**Parametri**:
- `--collection`: Nome della collezione Weaviate
- `--json`: File JSON con chunks
- `--vectors`: File NPZ con vettori
- `--weaviate-url`: URL server Weaviate

**Schema Automatico**:
- Properties: `text`, `source_pdf`, `page`, `chunk_id`, `country`, `token_count`
- Vectorizer: none (vettori pre-generati)
- Index: HNSW per ricerca efficiente

---

### 17. `query_weaviate.py`
**Descrizione**: Query interattiva per collezioni Weaviate

**Funzionalità**:
- Query testuale con vector search
- Filtri su metadati (country, source_pdf)
- Visualizzazione risultati con score
- Supporto per query complesse

**Utilizzo**:
```bash
python query_weaviate.py \
  --collection MyCollection \
  --query "cerca questo testo" \
  --top-k 10 \
  --model-path models/mxbai-embed-large-v1
```

**Parametri**:
- `--collection`: Nome collezione da interrogare
- `--query`: Testo della query
- `--top-k`: Numero di risultati (default: 5)
- `--model-path`: Path al modello di embedding

---

### 18. `remove_weaviate_collections.py`
**Descrizione**: Rimozione batch di collezioni Weaviate

**Funzionalità**:
- Elimina collezioni specificate
- Conferma interattiva
- Logging operazioni

**Utilizzo**:
```bash
python remove_weaviate_collections.py Collection1 Collection2 Collection3
```

**Note**:
- Operazione irreversibile (richiede conferma)
- Utile per pulizia ambiente di sviluppo

---

### 19. `connect_weaviate.py`
**Descrizione**: Test di connessione a Weaviate con gestione metodi di connessione

**Funzionalità**:
- Test connettività a server Weaviate
- Auto-detection localhost vs remoto
- Supporto per `connect_to_local()` e `connect_to_custom()`
- Diagnostics e troubleshooting

**Utilizzo**:
```bash
python connect_weaviate.py
```

**Output**:
```
✓ Connesso a Weaviate su http://localhost:8080
Collezioni disponibili: ['Cms', 'DocumentChunk', 'Copd']
```

**Note**:
- Richiede Weaviate running
- Usa per verificare configurazione prima di app complesse

---

## Programmi Avanzati

### 20. `knowledge_graph_builder.py`
**Descrizione**: Costruzione di grafi di conoscenza tra chunks di collezioni diverse

**Funzionalità**:
- Estrazione automatica di entità (acronimi, codici paese, termini chiave)
- Collegamento chunks tramite:
  - Metadati comuni (country, source_pdf)
  - Entità condivise (NATO, COPD, JWC, ecc.)
  - Similarità semantica vettoriale (opzionale)
- Export grafo in JSON e GraphML (per Gephi)
- Statistiche su nodi/archi/comunità

**Utilizzo**:
```bash
# Basic: solo metadata ed entità
python knowledge_graph_builder.py \
  --collections Cms DocumentChunk \
  --export knowledge_graph.json

# Avanzato: con similarità vettoriale
python knowledge_graph_builder.py \
  --collections Cms DocumentChunk \
  --with-vectors \
  --similarity-threshold 0.75 \
  --sample 1000 \
  --export knowledge_graph.json \
  --graphml graph.graphml
```

**Parametri**:
- `--collections`: Nomi delle collezioni Weaviate (multipli)
- `--weaviate-url`: URL Weaviate (default: http://localhost:8080)
- `--sample`: Limita a N oggetti per collezione (0 = tutti)
- `--export`: Path JSON di export del grafo
- `--graphml`: Export GraphML per Gephi/Cytoscape
- `--with-vectors`: Abilita collegamenti per similarità vettoriale
- `--similarity-threshold`: Soglia similarità (0.0-1.0, default: 0.75)

**Entità Estratte**:
- Acronimi: NATO, COPD, JWC, CM, SHAPE, ACT, ACO
- Codici paese: DUSHMAN, MURINUS, ecc.
- Termini chiave: crisis, operation, planning, directive

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
**Descrizione**: Visualizzazione interattiva/statica del Knowledge Graph

**Funzionalità**:
- Visualizzazione 2D statica (matplotlib PNG)
- Visualizzazione 2D interattiva (pyvis HTML)
- Visualizzazione 3D statica (matplotlib PNG)
- Visualizzazione 3D interattiva (plotly HTML)
- Visualizzazione 3D window (matplotlib interattivo, no HTML)
- Export GraphML per Gephi
- Campionamento per grandi grafi (500+ nodi)
- Layout spring-based per distribuzione ottimale

**Utilizzo**:
```bash
# 2D statico PNG
python visualize_knowledge_graph.py knowledge_graph.json \
  --output graph_2d.png \
  --sample 300 \
  --figsize 20,20

# 2D interattivo HTML (pyvis)
python visualize_knowledge_graph.py knowledge_graph.json \
  --interactive \
  --output graph_interactive.html

# 3D interattivo HTML (plotly)
python visualize_knowledge_graph.py knowledge_graph.json \
  --3d-html \
  --output graph_3d.html \
  --sample 200

# 3D window interattivo (matplotlib)
python visualize_knowledge_graph.py knowledge_graph.json \
  --3d-window \
  --sample 150

# Export GraphML per Gephi
python visualize_knowledge_graph.py knowledge_graph.json \
  --graphml graph_for_gephi.graphml
```

**Parametri**:
- `graph_json`: File JSON del grafo (posizionale)
- `--sample`: Limite nodi (0 = tutti, consigliato 150-300 per grafi grandi)
- `--output`: File di output (.png o .html)
- `--interactive`: Genera HTML interattivo (pyvis)
- `--3d`: Plot 3D statico (matplotlib PNG)
- `--3d-window`: 3D interattivo window (matplotlib, no HTML)
- `--3d-html`: 3D interattivo HTML (plotly)
- `--graphml`: Export GraphML per Gephi/Cytoscape
- `--figsize`: Dimensioni figura per plot statici (width,height)

**Colori Nodi**:
- Per collezione: colori diversi per ogni collection
- Per country: colori per codici paese
- Per community: rilevamento comunità con algoritmo Louvain

**Note**:
- Per grafi > 500 nodi, usare `--sample` per performance
- Pyvis e Plotly generano HTML interattivi navigabili
- Gephi offre analisi avanzate (centralità, modularità)

---

### 22. `rag_integrity_check.py`
**Descrizione**: Verifica integrità e coerenza logica dei dati RAG da Weaviate

**Funzionalità**:
- Controllo duplicati (testi identici)
- Validazione metadati (country, source_pdf, page)
- Verifica consistenza vettori (dimensione, valori anomali)
- Analisi distribuzione lunghezza testi
- Test di ricerca vettoriale (query di esempio)
- Report dettagliato con statistiche

**Utilizzo**:
```bash
python rag_integrity_check.py \
  --collections Cms DocumentChunk \
  --weaviate-url http://localhost:8080 \
  --report integrity_report.txt
```

**Parametri**:
- `--collections`: Collezioni da verificare
- `--weaviate-url`: URL Weaviate
- `--report`: File di output per report

**Controlli Eseguiti**:
1. **Duplicati**: Identifica testi identici o molto simili
2. **Metadati**: Verifica campi obbligatori e formati
3. **Vettori**: Dimensione uniforme, valori NaN/Inf
4. **Lunghezza testi**: Min/max/media caratteri, outliers
5. **Ricerca**: Test query vettoriali per verifica funzionalità

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
**Descrizione**: Analisi performance del sistema RAG da cronologia chat

**Funzionalità**:
- Parsing cronologia chat JSON
- Calcolo metriche:
  - Response time medio/mediano
  - Precisione retrieval (se disponibile ground truth)
  - Token usage per query
  - Coverage delle sorgenti
- Identificazione query problematiche
- Export report testuale e JSON

**Utilizzo**:
```bash
python rag_analyzer.py chat_history.json \
  --output rag_analysis_report.txt \
  --json-output rag_analysis.json
```

**Parametri**:
- `chat_log`: File JSON cronologia chat (posizionale)
- `--output`: File report testuale (default: rag_analysis_report.txt)
- `--json-output`: File JSON con dati strutturati (opzionale)

**Metriche Calcolate**:
- **Latency**: Tempo di risposta per query
- **Retrieval Quality**: Score medio chunks recuperati
- **Source Coverage**: Distribuzione sorgenti utilizzate
- **Query Complexity**: Lunghezza e complessità query
- **Error Rate**: Frequenza errori/fallimenti

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
**Descrizione**: Agente di ricerca autonomo per generare report PDF da RAG

**Funzionalità**:
- Query autonoma su RAG per topic di ricerca
- Retrieval multi-step con filtri di score
- Generazione report strutturato con LLM
- Export PDF con formattazione professionale
- Citazioni automatiche delle sorgenti

**Utilizzo**:
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

**Parametri**:
- `research_query`: Domanda/topic di ricerca (posizionale)
- `--json`: File JSON con chunks (default: cms.json)
- `--vectors`: File NPZ con vettori (default: cms.npz)
- `--model-path`: Path al modello di embedding
- `--ollama-model`: Modello Ollama (default: llama3.2)
- `--output`: File PDF di output (default: research_report_<timestamp>.pdf)
- `--max-chunks`: Massimo chunks da usare (default: 20)
- `--score-threshold`: Soglia minima di score (default: 0.7)

**Pipeline**:
1. **Query Expansion**: Genera varianti della query di ricerca
2. **Retrieval**: Cerca chunks rilevanti con score filtering
3. **Ranking**: Ordina per rilevanza e diversità
4. **Generation**: LLM genera report strutturato
5. **PDF Export**: Formattazione professionale con citazioni

**Output PDF**:
```
Research Report: [Topic]
Generated: [Date]

Executive Summary
... sintesi ...

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
**Descrizione**: Agente LangGraph per task definition e execution con RAG

**Funzionalità**:
- Definizione task complessi con LangGraph
- Integration con Weaviate per retrieval
- Execution multi-step con state management
- Export risultati in JSON e Markdown
- Supporto per task iterativi e condizionali

**Utilizzo**:
```bash
python langgraph_task_agent.py \
  --task "Analyze crisis relationship between MURINUS and DUSHMAN" \
  --collections Cms DocumentChunk \
  --ollama-model llama3.2 \
  --output tasks.json \
  --markdown-report report.md
```

**Parametri**:
- `--task`: Descrizione del task da eseguire
- `--collections`: Collezioni Weaviate per retrieval
- `--ollama-model`: Modello Ollama
- `--output`: File JSON risultati
- `--markdown-report`: Report Markdown (opzionale)

**Note**:
- Deprecation warning: Aggiornare a `langchain-ollama`
- Errore JSON parsing: Escape di caratteri speciali in prompts
- Variabile `markdown_report` non definita: Fix in sviluppo

---

### 26. `langchain_pdf_to_weaviate.py`
**Descrizione**: Import PDF a Weaviate usando LangChain

**Funzionalità**:
- Caricamento PDF con LangChain loaders
- Text splitting con LangChain splitters
- Embedding automatico
- Import batch a Weaviate
- Integrazione LangChain-Weaviate

**Utilizzo**:
```bash
python langchain_pdf_to_weaviate.py \
  --pdf-dir /path/to/pdfs \
  --collection MyDocs \
  --chunk-size 500 \
  --chunk-overlap 50
```

**Note**:
- Richiede: `pip install langchain langchain-community langchain-text-splitters`
- Alternative a `pdf_to_text_chunker.py` con LangChain

---

### 27. `langchain_rag_chatbot.py`
**Descrizione**: Chatbot RAG basato su LangChain

**Funzionalità**:
- RAG pipeline con LangChain
- Retrieval da Weaviate via LangChain
- Conversational memory
- Integration LLM (Ollama/OpenAI)

**Utilizzo**:
```bash
python langchain_rag_chatbot.py \
  --collection MyDocs \
  --ollama-model llama3.2
```

**Note**:
- Errore `Embeddings` non definito: Fix import `from langchain.embeddings import Embeddings`

---

### 28. `ollama_api.py`
**Descrizione**: Modulo API per interfacciarsi con Ollama server

**Funzionalità**:
- `call_ollama_api()`: Chiamata sincrona a Ollama
- `stream_ollama_api()`: Streaming di risposte
- Gestione endpoint `/v1/chat/completions`
- Error handling e retry logic

**Utilizzo**:
```python
from ollama_api import call_ollama_api, stream_ollama_api

# Chiamata sincrona
response = call_ollama_api(
    prompt="Ciao!",
    model="llama3.2",
    base_url="http://0.0.0.0:11434"
)

# Streaming
for chunk in stream_ollama_api(prompt, model, base_url):
    print(chunk, end='', flush=True)
```

**Note**:
- Usato da `app_weaviate_rag.py` e altri chatbot
- Endpoint: `http://0.0.0.0:11434/v1/chat/completions`

---

## Dipendenze e Requisiti

### Pacchetti Python Principali
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

# LangChain (opzionale)
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

### Servizi Esterni
1. **Weaviate**: Vector database
   - URL: `http://localhost:8080`
   - Installazione: Docker o standalone
   
2. **Ollama**: LLM server
   - URL: `http://0.0.0.0:11434`
   - Modello default: `llama3.2`
   - Installazione: `curl https://ollama.ai/install.sh | sh`

### Modelli di Embedding
- **mxbai-embed-large-v1**: 1024 dimensioni
  - Path: `./models/mxbai-embed-large-v1`
  - Download: `python downloadmodel.py`

---

## Deployment con Docker

### Dockerfile Principale
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

### Build e Run
```bash
# Build immagine
docker build -t pdftext-rag:latest .

# Run con network host (per accesso a Weaviate/Ollama locali)
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

# Run con port mapping (se Weaviate è in altro container)
docker run --rm -it \
  -p 5003:5003 \
  -v "$(pwd)":/app \
  pdftext-rag:latest \
  python3 app_weaviate_rag.py \
    --weaviate-url http://host.docker.internal:8080 \
    ...
```

### Docker Compose (opzionale)
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

### Note Docker
- **Network host**: Necessario per accesso a servizi su localhost (Weaviate, Ollama) quando sono esterni al container
- **Port mapping**: Usa `-p 5003:5003` per esporre Flask app
- **Volume mount**: `-v "$(pwd)":/app` per accesso a modelli e dati locali
- **host.docker.internal**: Usa per riferirsi a localhost della macchina host da dentro container

---

## Troubleshooting Comune

### Errori Weaviate

1. **`TypeError: Client.__init__() got an unexpected keyword argument 'url'`**
   - Causa: Versione Weaviate client 4.x cambia API
   - Soluzione: Usa `weaviate.connect_to_local()` o `weaviate.connect_to_custom()`
   
2. **`WeaviateQueryError: vector lengths don't match: 384 vs 1024`**
   - Causa: Mismatch dimensioni vettori tra query ed index
   - Soluzione: Verifica modello embedding (mxbai = 1024 dim)

3. **`Connection refused to http://0.0.0.0:8080`**
   - Causa: Weaviate non in esecuzione
   - Soluzione: Avvia Weaviate: `docker run -p 8080:8080 semitechnologies/weaviate`

### Errori LLM

1. **`Error calling Ollama API: Expecting value: line 1 column 1 (char 0)`**
   - Causa: Ollama server non risponde o formato risposta errato
   - Soluzione: Verifica `curl http://0.0.0.0:11434/v1/models`

2. **`LangChainDeprecationWarning: The class Ollama was deprecated`**
   - Causa: Versione vecchia di langchain
   - Soluzione: `pip install -U langchain-ollama` e cambia import

### Errori Embedding

1. **`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`**
   - Causa: Path locale non riconosciuto come repo name
   - Soluzione: Usa nome modello HF o verifica path esiste

2. **`ModuleNotFoundError: No module named 'sentence_transformers'`**
   - Causa: Libreria non installata
   - Soluzione: `pip install sentence-transformers`

### Errori Docker

1. **Container non si connette a localhost**
   - Causa: Isolation networking di Docker
   - Soluzione: Usa `--network host` o `http://host.docker.internal:PORT`

2. **`FileNotFoundError: models/mxbai-embed-large-v1`**
   - Causa: Modello non montato in volume
   - Soluzione: Aggiungi `-v "$(pwd)/models:/app/models"` o download nel container

---

## Best Practices

### Performance
1. **Batch Processing**: Usa batch di 50-100 oggetti per import Weaviate
2. **Sampling**: Limita a 200-500 nodi per visualizzazioni grafi grandi
3. **Caching**: Riutilizza embeddings pre-calcolati (NPZ files)
4. **Vector Tree**: Usa per dataset > 10K chunks

### Sicurezza
1. **API Keys**: Non committare credenziali (usa `.env`)
2. **Network**: Limita accesso Weaviate/Ollama con firewall
3. **Input Validation**: Sanitizza input utente prima di query LLM

### Manutenzione
1. **Logging**: Abilita chat logs per analisi performance
2. **Monitoring**: Controlla latency e error rate con `rag_analyzer.py`
3. **Integrity Checks**: Esegui `rag_integrity_check.py` periodicamente
4. **Backup**: Archivia collezioni Weaviate regolarmente

---

## Roadmap e Sviluppi Futuri

### In Sviluppo
- [ ] Fix deprecation warnings LangChain
- [ ] Migrazione completa a weaviate-client 4.x
- [ ] Supporto multi-lingua embeddings
- [ ] UI web migliorata con rendering Markdown

### Pianificati
- [ ] Graph-RAG con reranking
- [ ] Fine-tuning modelli embedding custom
- [ ] Supporto per PDF multimodali (immagini, tabelle)
- [ ] API RESTful per integration esterna
- [ ] Dashboard analytics in tempo reale

---

## Conclusioni

Questo progetto fornisce un ecosistema completo per RAG su documenti PDF:
- **Pipeline completa**: Da PDF a chatbot intelligente
- **Flessibilità**: Supporto file locali (JSON/NPZ) e Weaviate
- **Scalabilità**: Ottimizzazioni per grandi dataset
- **Visualizzazione**: Grafi di conoscenza 2D/3D interattivi
- **Production-ready**: Docker, logging, integrity checks

Per domande o contributi, consultare la documentazione individuale di ogni script con `--help`.

---

**Report generato**: 2025-01-31  
**Versione progetto**: 1.0  
**Autore**: AI Assistant
