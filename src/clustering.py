"""
clustering.py
-------------
Step 4 della pipeline: clustering con algoritmo di Louvain.
"""

import logging
import time
import numpy as np
import community as community_louvain
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def louvain_clustering(
    G: nx.Graph,
    weight: str = "weight",
    random_state: int = 42,
) -> dict[int, int]:
    """
    Applica l'algoritmo di Louvain per la community detection sul grafo G,
    prevenendo crash dovuti a pesi infiniti.
    """
    logger.info("[STEP 4] Clustering Louvain ...")
    
    # -----------------------------------------------------------------------
    # SANITIZZAZIONE DEI PESI (Risoluzione del bug Max: inf)
    # -----------------------------------------------------------------------
    # Estraiamo tutti i pesi finiti per trovare un valore massimo sensato
    finite_weights = [
        d[weight] for u, v, d in G.edges(data=True) 
        if weight in d and np.isfinite(d[weight])
    ]
    
    # Se non ci sono pesi validi usiamo 1.0 come default, altrimenti prendiamo il max
    max_finite = max(finite_weights) if finite_weights else 1.0
    
    # Sostituiamo gli 'inf' con un valore pari a 2 volte il massimo valore finito trovato
    # (o un valore arbitrariamente alto ma sicuro)
    replacement_weight = max_finite * 2 if max_finite > 0 else 100.0
    
    inf_count = 0
    for u, v, d in G.edges(data=True):
        if weight in d and not np.isfinite(d[weight]):
            d[weight] = replacement_weight
            inf_count += 1
            
    if inf_count > 0:
        logger.warning(
            f"  [BONIFICA] Trovati {inf_count} archi con peso infinito (inf)! "
            f"Sostituiti con il valore finito stabile: {replacement_weight:.4f}"
        )
    # -----------------------------------------------------------------------

    t0 = time.time()

    # Esecuzione dell'algoritmo nativo su valori ora sicuri
    partition: dict[int, int] = community_louvain.best_partition(
        G,
        weight=weight,
        random_state=random_state,
    )

    n_clusters = len(set(partition.values()))
    cluster_sizes = (
        pd.Series(partition.values())
        .value_counts()
        .sort_index()
    )

    logger.info(
        "  Louvain completato in %.2fs | cluster trovati: %d",
        time.time() - t0,
        n_clusters,
    )
    logger.info("  Distribuzione cluster:\n%s", cluster_sizes.to_string())

    return partition