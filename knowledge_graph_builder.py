#!/usr/bin/env python3
"""
Programma per creare grafi di conoscenza tra chunk di collezioni Weaviate diverse,
collegando spezzoni tramite dipendenze logiche per migliorare la ricerca RAG.

Dipendenze costruite tramite:
  - Metadati: stesso country, stesso source_pdf
  - Entità/parole chiave: acronimi (NATO, COPD, JWC, CM), codici paese, termini significativi
  - Similarità semantica (opzionale): vettori per collegare chunk simili tra collezioni

Uso:
  python knowledge_graph_builder.py --collections Cms DocumentChunk
  python knowledge_graph_builder.py --collections Cms DocumentChunk --sample 1000 --export knowledge_graph.json
  python knowledge_graph_builder.py --collections Cms DocumentChunk --with-vectors --similarity-threshold 0.75
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import weaviate
except ImportError:
    print("Errore: pacchetto weaviate non installato. pip install weaviate-client")
    sys.exit(1)

try:
    import networkx as nx
except ImportError:
    print("Errore: pacchetto networkx non installato. pip install networkx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Connessione Weaviate (stessa logica di connect_weaviate.py)
# ---------------------------------------------------------------------------

def connect_to_weaviate(weaviate_url: str = "http://localhost:8080"):
    """Connessione a Weaviate. Restituisce client o None."""
    try:
        if weaviate_url in ["http://localhost:8080", "http://127.0.0.1:8080"]:
            client = weaviate.connect_to_local()
        else:
            url_clean = weaviate_url.replace("http://", "").replace("https://", "")
            host, port = (url_clean.split(":", 1)[0], int(url_clean.split(":", 1)[1])) if ":" in url_clean else (url_clean, 8080)
            http_secure = weaviate_url.startswith("https://")
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=http_secure,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=http_secure,
            )
        if client.is_ready():
            return client
        return None
    except Exception as e:
        print(f"✗ Errore connessione Weaviate: {e}")
        return None


# ---------------------------------------------------------------------------
# Strutture dati
# ---------------------------------------------------------------------------

@dataclass
class ChunkNode:
    """Nodo del grafo: un chunk con metadati."""
    node_id: str
    collection: str
    uuid: str
    chunk_id: Optional[int]
    text: str
    source_pdf: str
    country: str
    page_start: int
    page_end: int
    entities: Set[str] = field(default_factory=set)
    vector: Optional[List[float]] = None


def make_node_id(collection: str, uuid: str) -> str:
    return f"{collection}:{uuid}"


# ---------------------------------------------------------------------------
# Fetch chunk da Weaviate
# ---------------------------------------------------------------------------

def fetch_chunks(
    client: weaviate.WeaviateClient,
    collection_name: str,
    max_objects: int = 2000,
    include_vectors: bool = False,
) -> Tuple[List[ChunkNode], List[Optional[List[float]]]]:
    """Recupera chunk da una collezione. Restituisce (lista ChunkNode, lista vettori)."""
    if not client.collections.exists(collection_name):
        return [], []

    collection = client.collections.get(collection_name)
    nodes: List[ChunkNode] = []
    vectors: List[Optional[List[float]]] = []
    limit = min(500, max_objects)
    offset = 0
    fetched = 0

    while fetched < max_objects:
        result = collection.query.fetch_objects(limit=limit, offset=offset)
        if not result.objects:
            break
        for obj in result.objects:
            if fetched >= max_objects:
                break
            props = dict(obj.properties) if obj.properties else {}
            uuid_str = str(obj.uuid)
            node_id = make_node_id(collection_name, uuid_str)

            text = (props.get("text") or "").strip()
            source_pdf = (props.get("source_pdf") or props.get("source") or "unknown").strip()
            country = (props.get("country") or "").strip()
            page_start = int(props.get("page_start") or props.get("page") or 0)
            page_end = int(props.get("page_end") or page_start or 0)
            chunk_id = props.get("chunk_id")
            if chunk_id is not None:
                try:
                    chunk_id = int(chunk_id)
                except (TypeError, ValueError):
                    chunk_id = None

            node = ChunkNode(
                node_id=node_id,
                collection=collection_name,
                uuid=uuid_str,
                chunk_id=chunk_id,
                text=text,
                source_pdf=source_pdf,
                country=country,
                page_start=page_start,
                page_end=page_end,
            )
            vec = None
            if include_vectors:
                try:
                    obj_with_vec = collection.data.get_by_id(obj.uuid, include_vector=True)
                    if obj_with_vec:
                        v = getattr(obj_with_vec, "vector", None) or getattr(obj_with_vec, "vectors", None)
                        if isinstance(v, dict):
                            vec = v.get("default") or (list(v.values())[0] if v else None)
                        elif v is not None:
                            vec = v
                except Exception:
                    pass
            node.vector = vec
            nodes.append(node)
            vectors.append(vec)
            fetched += 1
        if len(result.objects) < limit:
            break
        offset += limit

    return nodes, vectors


# ---------------------------------------------------------------------------
# Estrazione entità / parole chiave
# ---------------------------------------------------------------------------

# Pattern per acronimi e entità note (NATO, COPD, JWC, CM, OPLAN, SHAPE, etc.)
ENTITY_PATTERNS = [
    r"\bNATO\b",
    r"\bCOPD\b",
    r"\bCOPC\b",
    r"\bJWC\b",
    r"\bOPLAN\b",
    r"\bOPLANs\b",
    r"\bSHAPE\b",
    r"\bCM\b",
    r"\bCMs\b",
    r"\bETI\b",
    r"\bSTDA\b",
    r"\bSTDC\b",
    r"\bSTDU\b",
    r"\bSACEUR\b",
    r"\bACT\b",
    r"\bNCS\b",
    r"\bRAG\b",
    r"\bRCC\b",
    r"\bCrisis\b",
    r"\bCrisis Management\b",
    r"\bComprehensive Approach\b",
    r"\bStrategic\b",
    r"\bOperational\b",
    r"\bTactical\b",
    r"\bMilitary\b",
    r"\bCivilian\b",
    r"\bDiplomatic\b",
    r"\bExercise\b",
    r"\bExercise\b",
]

# Codici paese tipici (2-4 lettere maiuscole)
COUNTRY_CODE_PATTERN = re.compile(r"\b([A-Z]{2,4})\b")


def extract_entities(text: str) -> Set[str]:
    """Estrae entità e termini significativi dal testo."""
    if not text or not text.strip():
        return set()
    text_upper = text.upper()
    entities: Set[str] = set()

    # Pattern predefiniti
    for pat in ENTITY_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            entities.add(m.group(0).upper())

    # Acronimi 2-5 lettere maiuscole
    for m in re.finditer(r"\b([A-Z]{2,5})\b", text):
        w = m.group(1)
        if len(w) >= 2 and w not in {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "ITS", "MAY", "DID", "GET", "LET", "PUT", "SAY", "SHE", "TOO", "USE"}:
            entities.add(w)

    # Parole significative (lunghezza >= 4, prima lettera maiuscola o tutto maiuscolo)
    words = re.findall(r"\b[A-Za-z][a-zA-Z]{3,}\b", text)
    for w in words:
        if w.upper() not in {"THAT", "THIS", "WITH", "FROM", "HAVE", "MORE", "WILL", "THEY", "BEEN", "THAN", "WHICH", "THEIR", "WOULD", "THERE", "COULD", "OTHER", "ABOUT", "AFTER", "BEFORE", "BETWEEN", "DURING", "WHERE", "WHILE", "AGAINST", "BECAUSE", "WITHOUT"}:
            entities.add(w.upper())

    return entities


# ---------------------------------------------------------------------------
# Costruzione archi
# ---------------------------------------------------------------------------

def build_edges_metadata(
    nodes_by_collection: Dict[str, List[ChunkNode]],
    link_same_country: bool = True,
    link_same_source: bool = True,
) -> List[Tuple[str, str, str, float]]:
    """Archi per stesso country o stesso source_pdf (solo tra chunk di collezioni diverse o stessa)."""
    edges: List[Tuple[str, str, str, float]] = []  # (from, to, type, weight)
    all_nodes: List[ChunkNode] = []
    for lst in nodes_by_collection.values():
        all_nodes.extend(lst)

    for i, a in enumerate(all_nodes):
        for j, b in enumerate(all_nodes):
            if i >= j:
                continue
            if a.node_id == b.node_id:
                continue
            best_w, best_rel = 0.0, ""
            if link_same_country and a.country and b.country and a.country.upper() == b.country.upper():
                if 0.8 > best_w:
                    best_w, best_rel = 0.8, "same_country"
            if link_same_source and a.source_pdf and b.source_pdf and a.source_pdf == b.source_pdf:
                if 0.9 > best_w:
                    best_w, best_rel = 0.9, "same_source"
            if best_w > 0:
                edges.append((a.node_id, b.node_id, best_rel, best_w))
    return edges


def build_edges_entities(
    nodes_by_collection: Dict[str, List[ChunkNode]],
    min_shared: int = 2,
    max_edges_per_node: int = 50,
) -> List[Tuple[str, str, str, float]]:
    """Archi tra chunk che condividono almeno min_shared entità (preferenza per collezioni diverse)."""
    edges: List[Tuple[str, str, str, float]] = []
    collections = list(nodes_by_collection.keys())

    for c1 in collections:
        for c2 in collections:
            for a in nodes_by_collection[c1]:
                a_ent = a.entities if a.entities else extract_entities(a.text)
                a.entities = a_ent
                candidates: List[Tuple[float, str]] = []
                for b in nodes_by_collection[c2]:
                    if a.node_id == b.node_id:
                        continue
                    b_ent = b.entities if b.entities else extract_entities(b.text)
                    b.entities = b_ent
                    shared = a_ent & b_ent
                    if len(shared) >= min_shared:
                        w = 0.5 + 0.2 * min(len(shared), 5)
                        if c1 != c2:
                            w += 0.2
                        candidates.append((w, b.node_id))
                candidates.sort(key=lambda x: -x[0])
                for w, bid in candidates[:max_edges_per_node]:
                    edges.append((a.node_id, bid, "shared_entities", round(w, 2)))
    return edges


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Similarità coseno tra due vettori."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_edges_semantic(
    nodes_by_collection: Dict[str, List[ChunkNode]],
    threshold: float = 0.75,
    max_edges_per_node: int = 10,
) -> List[Tuple[str, str, str, float]]:
    """Archi per similarità vettoriale (solo tra collezioni diverse)."""
    edges: List[Tuple[str, str, str, float]] = []
    collections = list(nodes_by_collection.keys())
    node_map: Dict[str, ChunkNode] = {}
    for c in collections:
        for n in nodes_by_collection[c]:
            node_map[n.node_id] = n

    for c1 in collections:
        for c2 in collections:
            if c1 >= c2:
                continue
            list1 = nodes_by_collection[c1]
            list2 = nodes_by_collection[c2]
            for a in list1:
                if not a.vector:
                    continue
                candidates: List[Tuple[float, str]] = []
                for b in list2:
                    if a.node_id == b.node_id or not b.vector:
                        continue
                    sim = cosine_similarity(a.vector, b.vector)
                    if sim >= threshold:
                        candidates.append((sim, b.node_id))
                candidates.sort(key=lambda x: -x[0])
                for sim, bid in candidates[:max_edges_per_node]:
                    edges.append((a.node_id, bid, "semantic_similar", round(sim, 3)))
    return edges


# ---------------------------------------------------------------------------
# Grafo NetworkX
# ---------------------------------------------------------------------------

def build_knowledge_graph(
    nodes_by_collection: Dict[str, List[ChunkNode]],
    edges_metadata: List[Tuple[str, str, str, float]],
    edges_entities: List[Tuple[str, str, str, float]],
    edges_semantic: Optional[List[Tuple[str, str, str, float]]] = None,
) -> nx.Graph:
    """Costruisce il grafo di conoscenza con nodi e archi."""
    G = nx.Graph()
    for coll, nodes in nodes_by_collection.items():
        for n in nodes:
            G.add_node(
                n.node_id,
                collection=n.collection,
                uuid=n.uuid,
                chunk_id=n.chunk_id,
                text_preview=(n.text[:200] + "..." if len(n.text) > 200 else n.text),
                source_pdf=n.source_pdf,
                country=n.country,
                page_start=n.page_start,
                page_end=n.page_end,
                entities=list(n.entities) if n.entities else [],
            )
    for e in edges_metadata:
        u, v, rel, w = e
        if G.has_node(u) and G.has_node(v):
            if not G.has_edge(u, v):
                G.add_edge(u, v, relation=rel, weight=w)
            else:
                # aggiorna peso massimo
                d = G.edges[u, v]
                d["weight"] = max(d.get("weight", 0), w)
                if "relation" not in d:
                    d["relation"] = rel
    for e in edges_entities:
        u, v, rel, w = e
        if G.has_node(u) and G.has_node(v):
            if not G.has_edge(u, v):
                G.add_edge(u, v, relation=rel, weight=w)
            else:
                d = G.edges[u, v]
                d["weight"] = max(d.get("weight", 0), w)
    if edges_semantic:
        for e in edges_semantic:
            u, v, rel, w = e
            if G.has_node(u) and G.has_node(v):
                if not G.has_edge(u, v):
                    G.add_edge(u, v, relation=rel, weight=w)
                else:
                    d = G.edges[u, v]
                    d["weight"] = max(d.get("weight", 0), w)
    return G


def get_related_chunks(G: nx.Graph, node_id: str, k: int = 10, same_collection: bool = False) -> List[Tuple[str, float]]:
    """Restituisce i k chunk più correlati a node_id (per espansione RAG)."""
    if not G.has_node(node_id):
        return []
    coll = G.nodes[node_id].get("collection")
    neighbors = list(G.neighbors(node_id))
    with_weight = [(n, G.edges[node_id, n].get("weight", 0.5)) for n in neighbors]
    if same_collection and coll:
        with_weight = [(n, w) for n, w in with_weight if G.nodes[n].get("collection") == coll]
    with_weight.sort(key=lambda x: -x[1])
    return with_weight[:k]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_graph_json(G: nx.Graph, path: Path) -> None:
    """Esporta nodi e archi in JSON per uso esterno."""
    nodes_list = []
    for nid in G.nodes():
        attrs = dict(G.nodes[nid])
        nodes_list.append({"id": nid, **attrs})
    edges_list = []
    for u, v in G.edges():
        d = dict(G.edges[u, v])
        edges_list.append({"source": u, "target": v, **d})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes_list, "edges": edges_list}, f, ensure_ascii=False, indent=2)
    print(f"  Grafo JSON salvato: {path}")


def load_graph_from_json(path: Path) -> nx.Graph:
    """Carica un grafo esportato in JSON (per uso in RAG / espansione query)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    G = nx.Graph()
    for n in data.get("nodes", []):
        nid = n.pop("id", None)
        if nid is not None:
            G.add_node(nid, **n)
    for e in data.get("edges", []):
        u = e.pop("source", None)
        v = e.pop("target", None)
        if u is not None and v is not None:
            G.add_edge(u, v, **e)
    return G


def export_graph_graphml(G: nx.Graph, path: Path) -> None:
    """Esporta in GraphML (apribile con Gephi, etc.)."""
    H = nx.Graph()
    for nid in G.nodes():
        attrs = dict(G.nodes[nid])
        for k, v in list(attrs.items()):
            if isinstance(v, list):
                attrs[k] = "|".join(str(x) for x in v)
        H.add_node(nid, **attrs)
    for u, v in G.edges():
        H.add_edge(u, v, **dict(G.edges[u, v]))
    nx.write_graphml(H, path)
    print(f"  Grafo GraphML salvato: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Costruisce grafi di conoscenza tra chunk di collezioni Weaviate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", default="http://localhost:8080", help="URL Weaviate")
    parser.add_argument("--collections", nargs="+", required=True, help="Nomi collezioni (es. Cms DocumentChunk)")
    parser.add_argument("--sample", type=int, default=1500, help="Max chunk per collezione (default 1500)")
    parser.add_argument("--with-vectors", action="store_true", help="Includi vettori per archi semantici")
    parser.add_argument("--similarity-threshold", type=float, default=0.78, help="Soglia similarità coseno (default 0.78)")
    parser.add_argument("--min-shared-entities", type=int, default=2, help="Min entità condivise per arco (default 2)")
    parser.add_argument("--max-entity-edges", type=int, default=50, help="Max archi entità per nodo (default 50)")
    parser.add_argument("--export", metavar="PATH", help="Salva grafo (estensione .json o .graphml)")
    parser.add_argument("--no-same-source", action="store_true", help="Non collegare stesso source_pdf")
    parser.add_argument("--no-same-country", action="store_true", help="Non collegare stesso country")
    args = parser.parse_args()

    print("Grafo di conoscenza – collegamento chunk tra collezioni")
    print("Connessione a Weaviate...")
    client = connect_to_weaviate(args.url)
    if not client:
        sys.exit(1)
    print("✓ Connesso\n")

    nodes_by_collection: Dict[str, List[ChunkNode]] = {}
    for coll in args.collections:
        if not client.collections.exists(coll):
            print(f"✗ Collezione '{coll}' non trovata.")
            continue
        print(f"  Caricamento '{coll}' (max {args.sample} chunk)...")
        nodes, vecs = fetch_chunks(client, coll, max_objects=args.sample, include_vectors=args.with_vectors)
        for i, n in enumerate(nodes):
            if i < len(vecs) and vecs[i] is not None:
                n.vector = vecs[i]
            n.entities = extract_entities(n.text)
        nodes_by_collection[coll] = nodes
        print(f"    Caricati {len(nodes)} chunk (vettori: {sum(1 for v in vecs if v is not None)}/{len(vecs)})")

    if not nodes_by_collection:
        print("Nessun chunk caricato.")
        return

    print("\nCostruzione archi...")
    edges_meta = build_edges_metadata(
        nodes_by_collection,
        link_same_country=not args.no_same_country,
        link_same_source=not args.no_same_source,
    )
    edges_ent = build_edges_entities(nodes_by_collection, min_shared=args.min_shared_entities, max_edges_per_node=args.max_entity_edges)
    edges_sem = None
    if args.with_vectors:
        edges_sem = build_edges_semantic(nodes_by_collection, threshold=args.similarity_threshold)
        print(f"  Archi metadati: {len(edges_meta)}, entità: {len(edges_ent)}, semantici: {len(edges_sem)}")
    else:
        print(f"  Archi metadati: {len(edges_meta)}, entità: {len(edges_ent)}")

    G = build_knowledge_graph(nodes_by_collection, edges_meta, edges_ent, edges_sem)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"\n✓ Grafo costruito: {n_nodes} nodi, {n_edges} archi")

    if args.export:
        p = Path(args.export)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".graphml":
            export_graph_graphml(G, p)
        else:
            export_graph_json(G, p)

    # Esempio: primi nodi e relativi collegati
    sample_node = next(iter(G.nodes()), None)
    if sample_node:
        related = get_related_chunks(G, sample_node, k=5)
        print(f"\nEsempio – chunk correlati a '{sample_node[:50]}...':")
        for rid, w in related[:5]:
            print(f"  -> {rid} (peso {w})")

    try:
        if hasattr(client, "__exit__"):
            client.__exit__(None, None, None)
    except Exception:
        pass
    print("\nFine.")


if __name__ == "__main__":
    main()
