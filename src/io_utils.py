"""
io_utils.py
-----------
Tutte le operazioni di I/O della pipeline: salvataggio e caricamento
di embedding, grafi, e risultati tabulari.

Separare l'I/O dalla logica computazionale permette di:
  - testare i moduli computazionali senza scrivere su disco
  - sostituire il backend di storage (es. S3) senza toccare altro codice
"""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Salvataggio embedding SVD
# ---------------------------------------------------------------------------
def save_svd_artifacts(
    X_reduced: np.ndarray,
    singular_values: np.ndarray,
    embedding_path: Path | str,
    singvals_path: Path | str,
) -> None:
    """Salva l'embedding ridotto e i valori singolari come file .npy."""
    np.save(embedding_path, X_reduced)
    np.save(singvals_path, singular_values)
    logger.info("[IO] SVD embedding salvato: %s", embedding_path)
    logger.info("[IO] Valori singolari salvati: %s", singvals_path)


def load_svd_artifacts(
    embedding_path: Path | str,
    singvals_path: Path | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Carica embedding e valori singolari da file .npy."""
    X_reduced = np.load(embedding_path)
    s = np.load(singvals_path)
    logger.info("[IO] SVD embedding caricato: %s %s", X_reduced.shape, X_reduced.dtype)
    return X_reduced, s


# ---------------------------------------------------------------------------
# Salvataggio/caricamento grafo
# ---------------------------------------------------------------------------
def save_graph(G: nx.Graph, path: Path | str) -> None:
    """Serializza il grafo in formato GraphML (leggibile da Gephi, Cytoscape, etc.)."""
    nx.write_graphml(G, path)
    logger.info("[IO] Grafo salvato (GraphML): %s", path)


def load_graph(path: Path | str) -> nx.Graph:
    """Carica un grafo da file GraphML."""
    G = nx.read_graphml(path)
    logger.info("[IO] Grafo caricato: %d nodi | %d archi", G.number_of_nodes(), G.number_of_edges())
    return G


# ---------------------------------------------------------------------------
# Salvataggio risultati clustering
# ---------------------------------------------------------------------------
def save_cluster_assignments(
    partition: dict[int, int],
    sample_names: list[str],
    output_path: Path | str,
) -> pd.DataFrame:
    """
    Salva la mappatura campione → cluster in formato CSV.

    Colonne: sample, cluster
    Ordinato per cluster poi per nome campione (utile per review manuale).

    Returns il DataFrame per uso downstream (es. statistiche).
    """
    records = [
        {"sample": sample_names[node_id], "cluster": cluster_id}
        for node_id, cluster_id in sorted(partition.items())
    ]
    df_out = (
        pd.DataFrame(records)
        .sort_values(["cluster", "sample"])
        .reset_index(drop=True)
    )
    df_out.to_csv(output_path, index=False)
    logger.info("[IO] Cluster assignments salvati: %s", output_path)
    return df_out


def load_cluster_assignments(path: Path | str) -> pd.DataFrame:
    """Carica il CSV dei cluster assignments."""
    df = pd.read_csv(path)
    logger.info("[IO] Cluster assignments caricati: %d righe", len(df))
    return df
