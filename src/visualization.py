"""
visualization.py
----------------
Step 5 della pipeline: tutte le funzioni di visualizzazione.

Ogni funzione è autonoma e può essere chiamata indipendentemente.
Nessuna funzione produce side-effect globali (no plt.show(), no stato condiviso).

Funzioni esportate:
  - plot_knn_graph          → grafo spring layout colorato per cluster
  - plot_cluster_summary    → barplot distribuzione campioni per cluster
  - plot_svd_scree          → scree plot valori singolari SVD
  - plot_phylum_heatmap     → heatmap biologica phylum x cluster
  - plot_umap               → proiezione UMAP 2D (richiede umap-learn)
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper interno: palette colori coerente con il numero di cluster
# ---------------------------------------------------------------------------
def _get_cmap(n_clusters: int):
    if n_clusters <= 20:
        return cm.get_cmap("tab20", n_clusters)
    return cm.get_cmap("nipy_spectral", n_clusters)


# ---------------------------------------------------------------------------
# 1. Grafo KNN con cluster colorati
# ---------------------------------------------------------------------------
def plot_knn_graph(
    G: nx.Graph,
    partition: dict[int, int],
    sample_names: list[str],
    output_path: Path | str,
    fig_size: tuple = (32, 28),
    dpi: int = 200,
    node_size: int = 80,
    font_size: float = 2.5,
    bg_color: str = "#111111",
) -> None:
    """
    Visualizza il grafo KNN con:
    - Layout spring (Fruchterman-Reingold) pesato sulla similarità coseno
    - Nodi colorati per cluster Louvain
    - Etichette = nome del campione (font ridotto per alta densità)
    - Legenda cluster con conteggio campioni
    """
    logger.info("[VIZ] Grafo KNN + cluster (spring layout) ...")

    n_clusters  = len(set(partition.values()))
    cmap        = _get_cmap(n_clusters)
    color_map   = {c: cmap(c) for c in range(n_clusters)}
    node_colors = [color_map[partition[n]] for n in G.nodes()]
    cluster_dist = pd.Series(partition.values()).value_counts().sort_index()

    logger.info("  Calcolo spring layout ...")
    pos = nx.spring_layout(
        G,
        weight="weight",
        seed=42,
        k=1.8 / np.sqrt(G.number_of_nodes()),
        iterations=60,
    )

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    # Archi
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        alpha=0.10,
        width=edge_weights,
        edge_color="#cccccc",
    )

    # Nodi
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.92,
        linewidths=0.5,
        edgecolors="white",
    )

    # Etichette
    labels = {n: sample_names[n] for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=font_size,
        font_color="white",
        font_weight="bold",
    )

    # Legenda
    legend_patches = [
        mpatches.Patch(
            color=color_map[c],
            label=f"Cluster {c}  (n={cluster_dist.get(c, 0)})",
        )
        for c in sorted(set(partition.values()))
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        fontsize=7,
        framealpha=0.35,
        facecolor="#1a1a1a",
        edgecolor="#555555",
        labelcolor="white",
        ncol=max(1, n_clusters // 10),
    )

    ax.set_title(
        f"FibroGraph – KNN Graph (K=15, cosine) + Louvain Clustering\n"
        f"{G.number_of_nodes()} campioni | {G.number_of_edges()} archi | {n_clusters} cluster",
        color="white",
        fontsize=15,
        pad=18,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("  Salvato: %s", output_path)


# ---------------------------------------------------------------------------
# 2. Barplot distribuzione campioni per cluster
# ---------------------------------------------------------------------------
def plot_cluster_summary(
    partition: dict[int, int],
    output_path: Path | str,
    fig_height: int = 5,
    dpi: int = 150,
) -> None:
    """
    Barplot con numero di campioni per cluster Louvain.
    Colori coerenti con plot_knn_graph.
    """
    logger.info("[VIZ] Cluster summary barplot ...")

    cluster_dist = pd.Series(partition.values()).value_counts().sort_index()
    n_clusters   = len(cluster_dist)
    cmap         = _get_cmap(n_clusters)

    fig, ax = plt.subplots(figsize=(max(10, n_clusters * 0.7), fig_height))
    bars = ax.bar(
        [f"C{c}" for c in sorted(cluster_dist.index)],
        [cluster_dist[c] for c in sorted(cluster_dist.index)],
        color=[cmap(c) for c in sorted(cluster_dist.index)],
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, c in zip(bars, sorted(cluster_dist.index)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(cluster_dist[c]),
            ha="center", va="bottom", fontsize=7,
        )

    ax.set_xlabel("Cluster Louvain", fontsize=11)
    ax.set_ylabel("Numero campioni", fontsize=11)
    ax.set_title(
        f"Distribuzione campioni per cluster "
        f"({n_clusters} cluster, {sum(cluster_dist)} campioni totali)",
        fontsize=12,
    )
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Salvato: %s", output_path)


# ---------------------------------------------------------------------------
# 3. Scree plot valori singolari SVD
# ---------------------------------------------------------------------------
def plot_svd_scree(
    singular_values: np.ndarray,
    k: int,
    output_path: Path | str,
    fig_size: tuple = (10, 4),
    dpi: int = 150,
) -> None:
    """
    Scree plot in scala log dei valori singolari SVD.
    Utile per valutare quanta varianza è catturata dalle prime k componenti.
    """
    logger.info("[VIZ] SVD scree plot ...")

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(
        range(1, len(singular_values) + 1),
        singular_values,
        "o-",
        markersize=3,
        color="steelblue",
        linewidth=1,
    )
    ax.set_xlabel("Componente SVD", fontsize=11)
    ax.set_ylabel("Valore singolare", fontsize=11)
    ax.set_title(f"Scree plot – SVD troncata (k={k})", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Salvato: %s", output_path)


# ---------------------------------------------------------------------------
# 4. Heatmap biologica: abbondanza media per phylum × cluster
# ---------------------------------------------------------------------------
def _extract_phylum(full_name: str) -> str:
    """Estrae il campo phylum (p__...) dalla stringa tassonomica SILVA."""
    try:
        for part in full_name.split(";"):
            if "p__" in part:
                return part.strip().replace("p__", "")
    except Exception:
        pass
    return "Unknown"


def plot_phylum_heatmap(
    df_raw: pd.DataFrame,
    partition: dict[int, int],
    sample_names: list[str],
    output_path: Path | str,
    top_n_phyla: int = 20,
    dpi: int = 150,
) -> None:
    """
    Heatmap: righe = top N phyla per abbondanza totale,
             colonne = cluster Louvain,
             valori = abbondanza media normalizzata per riga.

    Permette di identificare i phyla caratterizzanti ciascun cluster.
    """
    logger.info("[VIZ] Heatmap phylum × cluster ...")

    # Aggrega le abbondanze per phylum
    df_work = df_raw.copy()
    df_work["phylum"] = [_extract_phylum(idx) for idx in df_raw.index]
    df_phylum_agg = df_work.groupby("phylum").sum()   # (n_phyla, n_campioni)

    # Calcola abbondanza media per cluster
    cluster_labels = {sname: partition[i] for i, sname in enumerate(sample_names)}
    phylum_cluster: dict[str, pd.Series] = {}
    for c in sorted(set(partition.values())):
        cols = [s for s, cl in cluster_labels.items() if cl == c and s in df_phylum_agg.columns]
        if cols:
            phylum_cluster[f"C{c}"] = df_phylum_agg[cols].mean(axis=1)

    df_heatmap = pd.DataFrame(phylum_cluster)

    # Seleziona top N phyla
    top_phyla  = df_heatmap.sum(axis=1).nlargest(top_n_phyla).index
    df_heatmap = df_heatmap.loc[top_phyla]

    # Normalizzazione per riga (0–1 rispetto al valore massimo del phylum)
    df_norm = df_heatmap.div(df_heatmap.max(axis=1) + 1e-9, axis=0)

    n_clusters = df_norm.shape[1]
    fig, ax = plt.subplots(figsize=(max(8, n_clusters * 0.6), 8))
    im = ax.imshow(df_norm.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels(df_norm.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df_norm.index)))
    ax.set_yticklabels(df_norm.index, fontsize=7)
    ax.set_title(
        f"Heatmap abbondanza media (top-{top_n_phyla} phyla) per cluster Louvain",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, shrink=0.5, label="Abbondanza normalizzata")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Salvato: %s", output_path)


# ---------------------------------------------------------------------------
# 5. UMAP 2D (opzionale – richiede umap-learn)
# ---------------------------------------------------------------------------
def plot_umap(
    X_reduced: np.ndarray,
    partition: dict[int, int],
    output_path: Path | str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    dpi: int = 150,
) -> None:
    """
    Proiezione UMAP 2D dell'embedding SVD, colorata per cluster Louvain.
    Richiede: pip install umap-learn

    Salta silenziosamente se umap-learn non è installato.
    """
    try:
        import umap as umap_lib
    except ImportError:
        logger.warning(
            "  [VIZ] umap-learn non trovato – skip UMAP. "
            "Installa con: pip install umap-learn"
        )
        return

    logger.info("[VIZ] Proiezione UMAP 2D ...")

    reducer = umap_lib.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False,
    )
    embedding = reducer.fit_transform(X_reduced.T)   # (n_campioni, 2)

    n_clusters = len(set(partition.values()))
    cmap       = _get_cmap(n_clusters)

    fig, ax = plt.subplots(figsize=(12, 9))
    for c in sorted(set(partition.values())):
        mask = [i for i, p in partition.items() if p == c]
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            label=f"C{c}",
            color=cmap(c),
            alpha=0.8,
        )

    ax.set_title(
        f"UMAP 2D – {n_clusters} cluster Louvain su {X_reduced.shape[1]} campioni",
        fontsize=13,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=7, ncol=3, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Salvato: %s", output_path)
