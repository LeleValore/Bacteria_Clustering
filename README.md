# FibroGraph – Microbial Abundance Analysis Pipeline

Pipeline computazionale per l'analisi di abbondanza microbica tramite riduzione dimensionale SVD, costruzione di un grafo K-NN e clustering Louvain.

---

## Struttura del progetto

```
fibro_graph/
├── run_pipeline.py          # ← ENTRY POINT principale (eseguire questo)
├── pipeline.py              # Moduli riutilizzabili (funzioni)
├── requirements.txt
├── README.md
├── data/
│   └── genere_table_spectrum.txt   # matrice OTU (1295 batteri × 1344 campioni)
├── results/
│   ├── sample_clusters.csv         # campione → cluster ID
│   ├── svd_embedding.npy           # embedding ridotto (300 × 1344)
│   ├── singular_values.npy         # valori singolari SVD
│   └── knn_graph.graphml           # grafo esportato in GraphML
├── figures/
│   ├── knn_louvain_graph.png       # grafo KNN con cluster colorati
│   ├── cluster_summary.png         # barplot distribuzione cluster
│   ├── svd_scree_plot.png          # scree plot valori singolari
│   ├── phylum_cluster_heatmap.png  # heatmap biologica phylum × cluster
│   └── umap_clusters.png           # proiezione UMAP 2D (se installato)
└── logs/
    └── pipeline.log
```

---

## Dati

| Proprietà | Valore |
|-----------|--------|
| Tipo | Matrice abbondanza microbica (genere-level) |
| Righe | 1295 (batteri / ASV a livello di genere) |
| Colonne | 1344 (campioni ambientali) |
| Formato | TSV con header (nomi campioni) e index (tassonomia SILVA) |
| Sparsità | ~93% |
| Valori | Conteggi interi (read counts) |

---

## Pipeline

### STEP 1 – Preprocessing
- Lettura della matrice OTU
- **Total Sum Scaling (TSS)**: normalizzazione per profondità di sequenziamento
  `X_norm[i,j] = X[i,j] / sum(X[:,j])`
- **log(CPM + 1)** transformation per stabilizzare la varianza (Counts Per Million)

### STEP 2 – SVD Troncata (k=300)
La Singular Value Decomposition riduce la dimensionalità da 1295 features (batteri) a 300:

```
X ≈ U · Σ · V^T
```

Proiezione finale dei **campioni**:
```
X_reduced = Σ_k · V_k^T    →    shape (300, 1344)
```
Ogni colonna rappresenta un campione nello spazio latente a 300 dimensioni.

### STEP 3 – KNN Graph (K=15, cosine similarity)
- Usiamo **pynndescent** (NNDescent approssimato, Dong et al. 2011)
- Metrica: **cosine distance** = `1 - cosine_similarity`
- Ogni campione → archi verso i 15 campioni più simili
- Il grafo è **non diretto** (simmetrizzato)
- Peso arco = cosine similarity ∈ [0, 1]

### STEP 4 – Louvain Clustering
- Algoritmo di Louvain: ottimizzazione della **modularità** Q
- Input: grafo pesato KNN
- Output: partizione dei campioni in cluster

### STEP 5 – Visualizzazioni
1. **Grafo spring layout**: nodi colorati per cluster, etichette = nome campione
2. **Barplot**: distribuzione campioni per cluster
3. **Scree plot**: varianza spiegata dalla SVD
4. **Heatmap biologica**: abbondanza media top-20 phyla per cluster
5. **UMAP 2D**: proiezione 2D dello spazio SVD, colorata per cluster

---

## Installazione

```bash
pip install -r requirements.txt
```

## Esecuzione

```bash
cd fibro_graph
python run_pipeline.py
```

---

## Interpretazione biologica

I cluster Louvain raggruppano **campioni microbicamente simili**:
- Campioni nello stesso cluster condividono profili di abbondanza batterica simili
- La heatmap phylum × cluster mostra quali phyla caratterizzano ciascun cluster
- Cluster con alta abbondanza di Firmicutes/Bacteroidetes = tipico microbioma intestinale
- La struttura del grafo KNN riflette il "paesaggio" di similarità tra campioni

---

## Riferimenti
- Dong et al. (2011) – *Efficient k-nearest neighbor graph construction for generic similarity measures*
- Blondel et al. (2008) – *Fast unfolding of communities in large networks* (Louvain)
- McInnes et al. (2018) – *UMAP: Uniform Manifold Approximation and Projection*
