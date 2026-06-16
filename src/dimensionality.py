"""
dimensionality.py
-----------------
Step 2 della pipeline: riduzione dimensionale tramite SVD troncata.

Dato X (batteri x campioni), calcola l'embedding ridotto dei CAMPIONI:

    X ≈ U · Σ · Vᵀ
    X_reduced = Σ_k · Vᵀ_k     shape: (k, n_campioni)

Ogni colonna di X_reduced è l'embedding di un campione in spazio latente k-dim,
catturando le co-variazioni batteriche più rilevanti.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)


@dataclass
class SVDResult:
    """Contenitore tipizzato per i risultati della SVD."""
    X_reduced: np.ndarray   # (k, n_campioni)  – embedding campioni
    U: np.ndarray           # (n_batteri, k)   – componenti batteri
    s: np.ndarray           # (k,)             – valori singolari ordinati desc
    Vt: np.ndarray          # (k, n_campioni)  – proiezione campioni (non scalata)


def truncated_svd(X: np.ndarray, k: int = 300) -> SVDResult:
    """
    Applica SVD troncata a X e restituisce l'embedding ridotto dei campioni.

    Parameters
    ----------
    X : np.ndarray, shape (n_batteri, n_campioni)
        Matrice di abbondanza preprocessata.
    k : int
        Numero di valori singolari da mantenere.

    Returns
    -------
    SVDResult
        Struttura con X_reduced, U, s, Vt.
    """
    logger.info("[STEP 2] SVD troncata con k=%d valori singolari ...", k)
    t0 = time.time()

    # scipy.sparse.linalg.svds restituisce i k valori singolari in ordine CRESCENTE
    U, s, Vt = svds(X, k=k)

    # Invertiamo per avere ordine decrescente (convenzione standard)
    idx = np.argsort(s)[::-1]
    s  = s[idx]
    U  = U[:, idx]
    Vt = Vt[idx, :]

    logger.info("  Top-10 singular values: %s", np.round(s[:10], 2))
    logger.info("  U:%s  Σ:%s  Vᵀ:%s", U.shape, s.shape, Vt.shape)

    # Proiezione finale: Σ_k · Vᵀ_k  →  (k, n_campioni)
    # Ogni colonna è l'embedding del campione nello spazio latente k-dim.
    Sigma_k = np.diag(s)         # (k, k)
    X_reduced = Sigma_k @ Vt     # (k, n_campioni)

    logger.info(
        "  Embedding campioni: %s | completato in %.2fs",
        X_reduced.shape,
        time.time() - t0,
    )

    return SVDResult(X_reduced=X_reduced, U=U, s=s, Vt=Vt)
