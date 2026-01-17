#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def load_vectors(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=False)
    return data["vectors"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D visualization of vector embeddings with PCA."
    )
    parser.add_argument("vectors_npz", help="Path to vectors .npz file.")
    parser.add_argument(
        "--sample",
        type=int,
        default=2000,
        help="Max number of points to plot (default: 2000).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=8,
        help="Number of clusters for coloring (default: 8).",
    )
    args = parser.parse_args()

    vectors = load_vectors(Path(args.vectors_npz))
    if vectors.size == 0:
        raise SystemExit("No vectors found in file.")

    if vectors.shape[0] > args.sample:
        idx = np.random.choice(vectors.shape[0], size=args.sample, replace=False)
        vectors = vectors[idx]

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(vectors)

    clusters = min(args.clusters, reduced.shape[0])
    labels = (
        KMeans(n_clusters=clusters, n_init=10, random_state=42)
        .fit(reduced)
        .labels_
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        reduced[:, 2],
        s=6,
        alpha=0.7,
        c=labels,
        cmap="tab10",
    )
    ax.set_title("Vector Embeddings (PCA 3D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
