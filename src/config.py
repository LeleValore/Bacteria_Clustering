"""
config.py
---------
Unico punto di verità per tutti i path e gli iperparametri del progetto.
Modifica qui per cambiare il comportamento dell'intera pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root del progetto (due livelli sopra questo file: src/ -> root/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Cartelle I/O
# ---------------------------------------------------------------------------
DATA_DIR    = ROOT / "data"
IMG_DIR     = ROOT / "img"
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"

# Crea le cartelle se non esistono (idempotente)
for _d in (IMG_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File dati
# ---------------------------------------------------------------------------
DATA_FILE = DATA_DIR / "genere_table_spectrum.txt"

# ---------------------------------------------------------------------------
# Iperparametri pipeline
# ---------------------------------------------------------------------------
K_SVD: int = 300   # numero di valori singolari per la SVD troncata
K_KNN: int = 15    # vicini per il grafo KNN
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Iperparametri NNDescent
# ---------------------------------------------------------------------------
NNDESCENT_METRIC: str  = "cosine"
NNDESCENT_N_JOBS: int  = -1        # usa tutti i core disponibili
NNDESCENT_LOW_MEM: bool = False

# ---------------------------------------------------------------------------
# Iperparametri visualizzazione
# ---------------------------------------------------------------------------
GRAPH_FIG_SIZE   = (32, 28)
GRAPH_DPI        = 200
GRAPH_NODE_SIZE  = 80
GRAPH_FONT_SIZE  = 2.5
GRAPH_BG_COLOR   = "#111111"

SUMMARY_FIG_HEIGHT = 5
SUMMARY_DPI        = 150

SCREE_FIG_SIZE = (10, 4)
SCREE_DPI      = 150

HEATMAP_TOP_N_PHYLA = 20
HEATMAP_DPI         = 150

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1

# ---------------------------------------------------------------------------
# Path output
# ---------------------------------------------------------------------------
OUT_GRAPH_PNG    = IMG_DIR     / "knn_louvain_graph.png"
OUT_SUMMARY_PNG  = IMG_DIR     / "cluster_summary.png"
OUT_SCREE_PNG    = IMG_DIR     / "svd_scree_plot.png"
OUT_HEATMAP_PNG  = IMG_DIR     / "phylum_cluster_heatmap.png"
OUT_UMAP_PNG     = IMG_DIR     / "umap_clusters.png"

OUT_CLUSTERS_CSV  = RESULTS_DIR / "sample_clusters.csv"
OUT_EMBEDDING_NPY = RESULTS_DIR / "svd_embedding.npy"
OUT_SINGVALS_NPY  = RESULTS_DIR / "singular_values.npy"
OUT_GRAPH_GRAPHML = RESULTS_DIR / "knn_graph.graphml"

LOG_FILE = LOGS_DIR / "pipeline.log"
