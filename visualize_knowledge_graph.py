#!/usr/bin/env python3
"""
Visualize the knowledge graph exported by knowledge_graph_builder.py.

Options:
  1. Static image (matplotlib + networkx) – good for small/medium graphs or a sampled subgraph.
  2. 3D static (matplotlib mplot3d) – 3D layout, save as PNG.
  3. 3D interactive window (matplotlib) – native window, rotate/zoom with mouse; no HTML.
  4. 3D interactive (plotly) – rotate/zoom in browser; requires: pip install plotly
  5. Interactive HTML (pyvis) – zoom, pan, click nodes; requires: pip install pyvis
  6. Export to GraphML and open in Gephi (https://gephi.org).

Usage:
  python visualize_knowledge_graph.py knowledge_graph.json
  python visualize_knowledge_graph.py knowledge_graph.json --3d -o graph_3d.png
  python visualize_knowledge_graph.py knowledge_graph.json --3d-window   # interactive native window
  python visualize_knowledge_graph.py knowledge_graph.json --3d --3d-html -o graph_3d.html
  python visualize_knowledge_graph.py knowledge_graph.json --sample 150 --output graph.png
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    print("Error: networkx required. pip install networkx")
    sys.exit(1)


def load_graph_from_json(path: Path) -> nx.Graph:
    """Load graph from JSON export."""
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


def sample_graph(G: nx.Graph, max_nodes: int = 200, by_degree: bool = True) -> nx.Graph:
    """Return a subgraph with at most max_nodes, keeping high-degree nodes if by_degree."""
    if G.number_of_nodes() <= max_nodes:
        return G
    if by_degree:
        deg = dict(G.degree())
        top = sorted(deg.keys(), key=lambda x: -deg[x])[:max_nodes]
    else:
        top = list(G.nodes())[:max_nodes]
    return G.subgraph(top).copy()


def layout_3d(G: nx.Graph):
    """Compute 3D node positions. Returns dict node_id -> (x, y, z)."""
    import numpy as np
    nodes = list(G.nodes())
    n = len(nodes)

    try:
        from sklearn.manifold import SpectralEmbedding
        A = nx.to_numpy_array(G, nodelist=nodes)
        A = A + np.eye(n) * 0.01  # ensure connectivity for spectral
        emb = SpectralEmbedding(n_components=3, affinity="precomputed", random_state=42)
        pos_3d = emb.fit_transform(A)
    except Exception:
        # Fallback: 2D spring + degree as Z
        pos_2d = nx.spring_layout(G, seed=42, k=1.2)
        deg = dict(G.degree())
        pos_3d = {}
        for node in nodes:
            x, y = pos_2d[node]
            z = (deg.get(node, 0) / max(max(deg.values()), 1)) * 2 - 1
            pos_3d[node] = (x, y, z)
        return pos_3d

    pos_3d = {nodes[i]: (float(pos_3d[i, 0]), float(pos_3d[i, 1]), float(pos_3d[i, 2])) for i in range(n)}
    return pos_3d


def draw_3d_static(
    G: nx.Graph,
    output_path: Path,
    pos_3d: dict,
    figsize: tuple = (12, 10),
    node_size: int = 40,
) -> None:
    """Draw 3D graph with matplotlib and save to PNG."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
    except ImportError:
        print("Error: matplotlib required for 3D. pip install matplotlib")
        sys.exit(1)

    nodes = list(G.nodes())
    xs = [pos_3d[n][0] for n in nodes]
    ys = [pos_3d[n][1] for n in nodes]
    zs = [pos_3d[n][2] for n in nodes]

    collections = nx.get_node_attributes(G, "collection")
    if collections:
        coll_set = sorted(set(collections.values()))
        color_map = {c: i for i, c in enumerate(coll_set)}
        colors = [color_map.get(collections.get(node, ""), 0) for node in nodes]
        try:
            cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(coll_set), 1))
        except AttributeError:
            cmap = plt.cm.get_cmap("tab10", max(len(coll_set), 1))
        node_colors = [cmap(c) for c in colors]
    else:
        node_colors = "#1f78b4"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=node_colors, s=node_size, alpha=0.9)

    for u, v in G.edges():
        x = [pos_3d[u][0], pos_3d[v][0]]
        y = [pos_3d[u][1], pos_3d[v][1]]
        z = [pos_3d[u][2], pos_3d[v][2]]
        ax.plot(x, y, z, "k-", alpha=0.15, linewidth=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Knowledge graph 3D ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 3D (static): {output_path}")


def draw_3d_window(G: nx.Graph, pos_3d: dict, figsize: tuple = (12, 10), node_size: int = 40) -> None:
    """Open an interactive 3D window (matplotlib). Rotate with left-drag, zoom with scroll."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Error: matplotlib required for interactive 3D window. pip install matplotlib")
        sys.exit(1)

    nodes = list(G.nodes())
    xs = [pos_3d[n][0] for n in nodes]
    ys = [pos_3d[n][1] for n in nodes]
    zs = [pos_3d[n][2] for n in nodes]

    collections = nx.get_node_attributes(G, "collection")
    if collections:
        coll_set = sorted(set(collections.values()))
        color_map = {c: i for i, c in enumerate(coll_set)}
        colors = [color_map.get(collections.get(node, ""), 0) for node in nodes]
        try:
            cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(coll_set), 1))
        except AttributeError:
            cmap = plt.cm.get_cmap("tab10", max(len(coll_set), 1))
        node_colors = [cmap(c) for c in colors]
    else:
        node_colors = "#1f78b4"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=node_colors, s=node_size, alpha=0.9)

    for u, v in G.edges():
        x = [pos_3d[u][0], pos_3d[v][0]]
        y = [pos_3d[u][1], pos_3d[v][1]]
        z = [pos_3d[u][2], pos_3d[v][2]]
        ax.plot(x, y, z, "k-", alpha=0.15, linewidth=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Knowledge graph 3D – interactive ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)\nRotate: left-drag | Zoom: scroll | Pan: right-drag")
    plt.tight_layout()
    print("Opening interactive 3D window. Close the window to exit.")
    plt.show()


def draw_3d_interactive(G: nx.Graph, output_path: Path, pos_3d: dict, node_size: float = 8) -> None:
    """Export 3D interactive HTML with plotly (rotate, zoom, hover in browser)."""
    try:
        import plotly.graph_objects as go
        import numpy as np
    except ImportError:
        print("Error: plotly required for 3D interactive. pip install plotly")
        sys.exit(1)

    nodes = list(G.nodes())
    xs = [pos_3d[n][0] for n in nodes]
    ys = [pos_3d[n][1] for n in nodes]
    zs = [pos_3d[n][2] for n in nodes]

    collections = nx.get_node_attributes(G, "collection")
    text = []
    for n in nodes:
        attrs = G.nodes[n]
        coll = attrs.get("collection", "")
        prev = (attrs.get("text_preview", "") or "")[:100]
        text.append(f"{coll} | {prev}…")

    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        edge_x.extend([pos_3d[u][0], pos_3d[v][0], None])
        edge_y.extend([pos_3d[u][1], pos_3d[v][1], None])
        edge_z.extend([pos_3d[u][2], pos_3d[v][2], None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(100,100,100,0.2)", width=1),
        hoverinfo="none",
    )
    node_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers+text",
        text=[n.split(":")[0][:12] if ":" in n else str(n)[:12] for n in nodes],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(size=node_size, color=zs, colorscale="Viridis", opacity=0.9, line=dict(width=0.5, color="white")),
        hovertext=text,
        hoverinfo="text",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Knowledge graph 3D ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
    )
    fig.write_html(str(output_path))
    print(f"Saved 3D (interactive HTML): {output_path}")
    print("  Open in a browser to rotate, zoom, and hover nodes.")


def draw_static(G: nx.Graph, output_path: Path, figsize: tuple = (14, 10), node_size: int = 80) -> None:
    """Draw graph with matplotlib and save to PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib required for static plot. pip install matplotlib")
        sys.exit(1)

    # Layout: use spring for small graphs, otherwise kamada_kawa or random
    n = G.number_of_nodes()
    if n > 500:
        pos = nx.random_layout(G, seed=42)
    elif n > 100:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=1.5, seed=42, iterations=50)

    # Node colors by collection if present
    collections = nx.get_node_attributes(G, "collection")
    if collections:
        coll_set = sorted(set(collections.values()))
        color_map = {c: i for i, c in enumerate(coll_set)}
        colors = [color_map.get(collections.get(node, ""), 0) for node in G.nodes()]
        try:
            cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(coll_set), 1))
        except AttributeError:
            cmap = plt.cm.get_cmap("tab10", max(len(coll_set), 1))
        node_colors = [cmap(c) for c in colors]
    else:
        node_colors = "#1f78b4"

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.9)
    # Labels: short id (collection:uuid prefix) only for small graphs
    if n <= 80:
        labels = {n: n.split(":")[0] + ":" + n.split(":")[1][:8] + "…" if ":" in n else n[:12] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=5)
    plt.title(f"Knowledge graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def draw_interactive(G: nx.Graph, output_path: Path, height: str = "700px") -> None:
    """Generate interactive HTML with pyvis."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("Error: pyvis required for interactive HTML. pip install pyvis")
        sys.exit(1)

    net = Network(height=height, directed=False)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    for node in G.nodes():
        attrs = G.nodes[node]
        label = attrs.get("collection", "") + " " + (str(attrs.get("chunk_id", "")))
        title = attrs.get("text_preview", "")[:200] + "…" if attrs.get("text_preview", "") else node
        net.add_node(node, label=label[:30], title=title)

    for u, v in G.edges():
        w = G.edges[u, v].get("weight", 0.5)
        rel = G.edges[u, v].get("relation", "")
        net.add_edge(u, v, value=float(w), title=f"{rel} ({w})")

    net.save_graph(str(output_path))
    print(f"Saved interactive graph: {output_path}")
    print("  Open in a browser to zoom, pan, and click nodes.")


def export_graphml(G: nx.Graph, output_path: Path) -> None:
    """Export to GraphML for Gephi."""
    H = nx.Graph()
    for nid in G.nodes():
        attrs = dict(G.nodes[nid])
        for k, v in list(attrs.items()):
            if isinstance(v, list):
                attrs[k] = "|".join(str(x) for x in v)
        H.add_node(nid, **attrs)
    for u, v in G.edges():
        H.add_edge(u, v, **dict(G.edges[u, v]))
    nx.write_graphml(H, output_path)
    print(f"Saved GraphML: {output_path}")
    print("  Open with Gephi: https://gephi.org")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize knowledge graph from JSON export.")
    parser.add_argument("graph_json", type=Path, help="Path to knowledge_graph.json")
    parser.add_argument("--sample", type=int, default=0, help="Max nodes to show (0 = all). Use 150–300 for large graphs.")
    parser.add_argument("--output", "-o", type=Path, help="Output file: .png (static) or .html (interactive)")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive HTML (pyvis)")
    parser.add_argument("--3d", dest="three_d", action="store_true", help="3D static plot (matplotlib PNG)")
    parser.add_argument("--3d-window", dest="three_d_window", action="store_true", help="3D interactive window (matplotlib, no HTML)")
    parser.add_argument("--3d-html", dest="three_d_html", action="store_true", help="3D interactive HTML (plotly)")
    parser.add_argument("--graphml", type=Path, metavar="FILE", help="Also export GraphML for Gephi")
    parser.add_argument("--figsize", default="14,10", help="Figure size for static plot (width,height)")
    args = parser.parse_args()

    if not args.graph_json.exists():
        print(f"File not found: {args.graph_json}")
        sys.exit(1)

    G = load_graph_from_json(args.graph_json)
    n, m = G.number_of_nodes(), G.number_of_edges()
    print(f"Loaded graph: {n} nodes, {m} edges")

    if args.sample and n > args.sample:
        G = sample_graph(G, max_nodes=args.sample)
        print(f"Sampled subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if args.graphml:
        export_graphml(G, args.graphml)

    figsize = tuple(int(x) for x in args.figsize.split(","))

    if args.three_d or args.three_d_html or args.three_d_window:
        pos_3d = layout_3d(G)
        base = args.graph_json.parent / (args.graph_json.stem + "_3d")
        if args.three_d_window:
            draw_3d_window(G, pos_3d, figsize=figsize)
        if args.three_d_html:
            out_html = args.output if args.output else (base.with_suffix(".html"))
            draw_3d_interactive(G, out_html, pos_3d)
        if args.three_d:
            out_png = args.output if args.output and not args.three_d_html else (base.with_suffix(".png"))
            draw_3d_static(G, out_png, pos_3d, figsize=figsize)
    elif args.interactive:
        out = args.output or args.graph_json.with_suffix(".html")
        draw_interactive(G, out)
    elif args.output:
        draw_static(G, args.output, figsize=figsize)
    else:
        # Default: static PNG next to JSON
        out = args.graph_json.with_suffix(".png")
        if G.number_of_nodes() > 300:
            G = sample_graph(G, max_nodes=300)
            out = args.graph_json.parent / (args.graph_json.stem + "_sampled.png")
        draw_static(G, out)
        print("Tip: use --3d -o graph_3d.png for 3D, --3d --3d-html -o graph_3d.html for interactive 3D.")


if __name__ == "__main__":
    main()
