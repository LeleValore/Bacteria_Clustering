"""
graph.py
--------
Step 3 della pipeline: costruzione del grafo K-NN sui campioni.

Strategia:
  - Input: embedding (k, n_campioni) → trasposto in (n_campioni, k)
  - Metrica: similarità del coseno (tramite distanza coseno = 1 - cos_sim)
  - Algoritmo: NNDescent (Dong et al. 2011) - KNN approssimata in O(n log n)
  - Output: grafo NetworkX non diretto, archi pesati con cosine similarity

NNDescent vs KNN esatta:
  La KNN esatta su 1344 campioni x 300 feature richiede O(n² · d) operazioni.
  NNDescent converge in ~O(n^1.14) mantenendo alta recall (>95%).
  Ideale per dataset di questa dimensione.
"""

import logging
import time

import networkx as nx
import numpy as np
from pynndescent import NNDescent

logger = logging.getLogger(__name__)


def build_knn_graph(
    X_reduced: np.ndarray,
    sample_names: list[str],
    k: int = 15,
    metric: str = "cosine",
    n_jobs: int = -1,
    random_state: int = 42,
    low_memory: bool = False,
) -> nx.Graph:
    """
    Costruisce un grafo K-NN non diretto (simmetrizzato) sui campioni.

    Parameters
    ----------
    X_reduced : np.ndarray, shape (k_svd, n_campioni)
        Embedding ridotto dei campioni (output di truncated_svd).
    sample_names : list[str]
        Nomi dei campioni, usati come attributo `label` dei nodi.
    k : int
        Numero di vicini per ogni nodo.
    metric : str
        Metrica di distanza per NNDescent (default: "cosine").
    n_jobs : int
        Thread da usare (-1 = tutti i core).
    random_state : int
        Seed per riproducibilità.
    low_memory : bool
        Modalità low-memory NNDescent (più lento ma meno RAM).

    Returns
    -------
    G : nx.Graph
        Grafo non diretto con:
        - nodi: indice intero, attributo `label` = nome campione
        - archi: attributo `weight` = cosine similarity ∈ [0, 1]
    """
    logger.info("[STEP 3] Costruzione grafo KNN (K=%d, metric=%s) ...", k, metric)
    t0 = time.time()

    # NNDescent vuole shape (n_samples, n_features)
    data = X_reduced.T.astype(np.float32)   # (n_campioni, k_svd)
    n_samples = data.shape[0]

    # --- NNDescent -----------------------------------------------------------
    # n_neighbors = k+1 perché il primo vicino restituito è il nodo stesso
    index = NNDescent(
        data,
        n_neighbors=k + 1,
        metric=metric,
        n_jobs=n_jobs,
        random_state=random_state,
        low_memory=low_memory,
        verbose=False,
    )
    nn_indices, nn_distances = index.neighbor_graph
    # nn_indices   shape: (n_samples, k+1)
    # nn_distances shape: (n_samples, k+1)  – distanza coseno ∈ [0, 2]

    logger.info("  NNDescent completato in %.2fs", time.time() - t0)

    # --- Costruzione grafo NetworkX -----------------------------------------
    G = nx.Graph()

    # Nodi: un nodo per campione, attributo label = nome
    for i, name in enumerate(sample_names):
        G.add_node(i, label=name)

    # Archi: cosine_similarity = 1 − cosine_distance
    # Grafo non diretto → aggiungiamo l'arco solo se non esiste già (simmetrizzazione)
    for i in range(n_samples):
        for j_pos in range(1, k + 1):          # saltiamo j_pos=0 (sé stesso)
            j = int(nn_indices[i, j_pos])
            if j < 0 or j >= n_samples:
                continue
            dist = float(nn_distances[i, j_pos])
            sim  = max(0.0, 1.0 - dist)        # cosine similarity
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=sim)

    logger.info(
        "  Grafo costruito: %d nodi | %d archi | %.2fs",
        G.number_of_nodes(),
        G.number_of_edges(),
        time.time() - t0,
    )

    return G
