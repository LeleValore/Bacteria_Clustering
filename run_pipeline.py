"""
run_pipeline.py
---------------
Orchestratore della pipeline FibroGraph.

Questo file NON contiene logica computazionale.
Ogni step delega al modulo di competenza in src/.
Leggilo come pseudocodice eseguibile.

Utilizzo:
    python run_pipeline.py
"""

import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup logging (prima di qualsiasi import da src/)
# ---------------------------------------------------------------------------
# Aggiunge src/ al path per import puliti
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DATA_FILE, LOG_FILE,
    K_SVD, K_KNN, RANDOM_STATE,
    NNDESCENT_METRIC, NNDESCENT_N_JOBS, NNDESCENT_LOW_MEM,
    HEATMAP_TOP_N_PHYLA,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    OUT_GRAPH_PNG, OUT_SUMMARY_PNG, OUT_SCREE_PNG,
    OUT_HEATMAP_PNG, OUT_UMAP_PNG,
    OUT_CLUSTERS_CSV, OUT_EMBEDDING_NPY,
    OUT_SINGVALS_NPY, OUT_GRAPH_GRAPHML,
    GRAPH_FIG_SIZE, GRAPH_DPI, GRAPH_NODE_SIZE,
    GRAPH_FONT_SIZE, GRAPH_BG_COLOR,
    SUMMARY_FIG_HEIGHT, SUMMARY_DPI,
    SCREE_FIG_SIZE, SCREE_DPI,
    HEATMAP_DPI,
)
from src.preprocessing   import load_and_preprocess
from src.dimensionality  import truncated_svd
from src.graph           import build_knn_graph
from src.clustering      import louvain_clustering
from src.visualization   import (
    plot_knn_graph,
    plot_cluster_summary,
    plot_svd_scree,
    plot_phylum_heatmap,
    plot_umap,
)
from src.io_utils        import (
    save_svd_artifacts,
    save_graph,
    save_cluster_assignments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("  FibroGraph – Microbial Abundance Pipeline")
    logger.info("=" * 60)
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1 – Preprocessing
    # ------------------------------------------------------------------
    X_log, sample_names, df_raw = load_and_preprocess(DATA_FILE)

    # ------------------------------------------------------------------
    # Step 2 – Riduzione dimensionale (SVD troncata)
    # ------------------------------------------------------------------
    svd_result = truncated_svd(X_log, k=K_SVD)

    save_svd_artifacts(
        svd_result.X_reduced,
        svd_result.s,
        OUT_EMBEDDING_NPY,
        OUT_SINGVALS_NPY,
    )

    # ------------------------------------------------------------------
    # Step 3 – Costruzione grafo KNN
    # ------------------------------------------------------------------
    G = build_knn_graph(
        svd_result.X_reduced,
        sample_names,
        k=K_KNN,
        metric=NNDESCENT_METRIC,
        n_jobs=NNDESCENT_N_JOBS,
        random_state=RANDOM_STATE,
        low_memory=NNDESCENT_LOW_MEM,
    )

    save_graph(G, OUT_GRAPH_GRAPHML)

    # ------------------------------------------------------------------
    # Step 4 – Clustering Louvain
    # ------------------------------------------------------------------
    partition = louvain_clustering(G, random_state=RANDOM_STATE)

    save_cluster_assignments(partition, sample_names, OUT_CLUSTERS_CSV)

    # ------------------------------------------------------------------
    # Step 5 – Visualizzazioni
    # ------------------------------------------------------------------
    plot_knn_graph(
        G, partition, sample_names,
        output_path=OUT_GRAPH_PNG,
        fig_size=GRAPH_FIG_SIZE,
        dpi=GRAPH_DPI,
        node_size=GRAPH_NODE_SIZE,
        font_size=GRAPH_FONT_SIZE,
        bg_color=GRAPH_BG_COLOR,
    )

    plot_cluster_summary(
        partition,
        output_path=OUT_SUMMARY_PNG,
        fig_height=SUMMARY_FIG_HEIGHT,
        dpi=SUMMARY_DPI,
    )

    plot_svd_scree(
        svd_result.s,
        k=K_SVD,
        output_path=OUT_SCREE_PNG,
        fig_size=SCREE_FIG_SIZE,
        dpi=SCREE_DPI,
    )

    plot_phylum_heatmap(
        df_raw, partition, sample_names,
        output_path=OUT_HEATMAP_PNG,
        top_n_phyla=HEATMAP_TOP_N_PHYLA,
        dpi=HEATMAP_DPI,
    )

    plot_umap(
        svd_result.X_reduced,
        partition,
        output_path=OUT_UMAP_PNG,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_clusters = len(set(partition.values()))
    logger.info("\n" + "=" * 60)
    logger.info("  SUMMARY FINALE")
    logger.info("=" * 60)
    logger.info("  Campioni      : %d", len(sample_names))
    logger.info("  Batteri       : %d", df_raw.shape[0])
    logger.info("  SVD k         : %d", K_SVD)
    logger.info("  KNN K         : %d", K_KNN)
    logger.info("  Cluster       : %d", n_clusters)
    logger.info("  Tempo totale  : %.1fs", time.time() - t_start)
    logger.info("=" * 60)

    print("\n✅ Pipeline completata.")
    print(f"   img/      → {OUT_GRAPH_PNG.parent}")
    print(f"   results/  → {OUT_CLUSTERS_CSV.parent}")


if __name__ == "__main__":
    main()
