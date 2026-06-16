"""
preprocessing.py
----------------
Step 1 della pipeline: lettura della matrice OTU e preprocessing.

Trasformazioni applicate:
  1. Total Sum Scaling (TSS): normalizza ogni campione per la sua read depth
  2. log(CPM + 1): stabilizza la varianza della distribuzione iper-sparsa

Input:  file TSV  (righe = generi batterici, colonne = campioni)
Output: X_log (np.float32, stessa shape), lista nomi campioni, DataFrame raw
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_and_preprocess(
    filepath: Path | str,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Carica la matrice di abbondanza microbica e applica preprocessing standard.

    Parameters
    ----------
    filepath : Path | str
        Percorso al file TSV (header = campioni, index = tassonomia).

    Returns
    -------
    X_log : np.ndarray, shape (n_batteri, n_campioni), float32
        Matrice preprocessata: TSS + log(CPM + 1).
    sample_names : list[str]
        Nomi delle colonne (campioni), nell'ordine originale.
    df_raw : pd.DataFrame
        DataFrame originale grezzo (read counts), utile per analisi biologiche downstream.
    """
    logger.info("[STEP 1] Lettura file: %s", filepath)
    t0 = time.time()

    df_raw = pd.read_csv(filepath, sep="\t", index_col=0)
    n_batteri, n_campioni = df_raw.shape
    logger.info("  Matrice caricata: %d batteri × %d campioni", n_batteri, n_campioni)

    sample_names: list[str] = list(df_raw.columns)
    X = df_raw.values.astype(np.float32)

    # --- Total Sum Scaling (TSS) -------------------------------------------
    # Normalizza ogni campione (colonna) per la sua read depth totale.
    # Rende i campioni comparabili indipendentemente dalla profondità di sequenziamento.
    col_sums = X.sum(axis=0, keepdims=True)        # (1, n_campioni)
    col_sums[col_sums == 0] = 1.0                  # evita divisione per zero
    X_norm = X / col_sums                           # relative abundance ∈ [0, 1]

    # --- log(CPM + 1) ---------------------------------------------------------
    # CPM = Counts Per Million: porta le abbondanze relative su scala 0–1M,
    # poi log1p comprime gli outlier e rende la distribuzione più simmetrica.
    X_log = np.log1p(X_norm * 1_000_000)

    sparsity = (X == 0).sum() / X.size
    logger.info(
        "  Preprocessing completato in %.2fs | sparsità: %.1f%%",
        time.time() - t0,
        sparsity * 100,
    )

    return X_log, sample_names, df_raw
