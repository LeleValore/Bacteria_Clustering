"""
view_graph.py
-------------
Converte knn_graph.graphml in un HTML interattivo visualizzabile
direttamente in VSCode (Live Preview) o nel browser.

Utilizzo:
    python view_graph.py

Output:
    img/knn_graph_interactive.html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx
import pandas as pd
from pyvis.network import Network

from src.config import OUT_GRAPH_GRAPHML, OUT_CLUSTERS_CSV, IMG_DIR


# ---------------------------------------------------------------------------
# Carica grafo e cluster assignments
# ---------------------------------------------------------------------------
print("Caricamento grafo ...")
G = nx.read_graphml(OUT_GRAPH_GRAPHML)

df_clusters = pd.read_csv(OUT_CLUSTERS_CSV)
# Mappa label → cluster_id
label_to_cluster = dict(zip(df_clusters["sample"], df_clusters["cluster"]))

n_clusters = df_clusters["cluster"].nunique()
print(f"  {G.number_of_nodes()} nodi | {G.number_of_edges()} archi | {n_clusters} cluster")

# ---------------------------------------------------------------------------
# Palette colori per cluster
# ---------------------------------------------------------------------------
import matplotlib.cm as cm
import matplotlib.colors as mcolors

cmap = cm.get_cmap("tab20", n_clusters) if n_clusters <= 20 else cm.get_cmap("nipy_spectral", n_clusters)

def cluster_color(cluster_id: int) -> str:
    rgba = cmap(cluster_id)
    return mcolors.to_hex(rgba)


# ---------------------------------------------------------------------------
# Costruisce il network pyvis
# ---------------------------------------------------------------------------
net = Network(
    height="95vh",
    width="100%",
    bgcolor="#111111",
    font_color="white",
    notebook=False,
    directed=False,
)

# Opzioni fisiche: Barnes-Hut simile a ForceAtlas2
net.set_options("""
{
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -8000,
      "centralGravity": 0.3,
      "springLength": 120,
      "springConstant": 0.04,
      "damping": 0.09
    },
    "stabilization": {
      "enabled": true,
      "iterations": 200
    }
  },
  "nodes": {
    "borderWidth": 1,
    "borderWidthSelected": 3,
    "font": { "size": 9, "color": "white" }
  },
  "edges": {
    "smooth": { "enabled": false },
    "color": { "color": "#444444", "highlight": "#ffffff" }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "navigationButtons": true,
    "keyboard": true
  }
}
""")

# Aggiungi nodi
for node_id, data in G.nodes(data=True):
    label = data.get("label", str(node_id))
    cluster_id = label_to_cluster.get(label, 0)
    color = cluster_color(cluster_id)

    net.add_node(
        node_id,
        label=label,
        title=f"<b>{label}</b><br>Cluster: {cluster_id}",   # tooltip hover
        color=color,
        size=10,
    )

# Aggiungi archi (peso → spessore visivo)
for u, v, data in G.edges(data=True):
    weight = data.get("weight", 0.5)
    net.add_edge(
        u, v,
        value=weight,          # pyvis usa value per lo spessore
        title=f"sim: {weight:.3f}",
    )

# ---------------------------------------------------------------------------
# Salva HTML
# ---------------------------------------------------------------------------
output_path = IMG_DIR / "knn_graph_interactive.html"
net.save_graph(str(output_path))
print(f"\n✅ Grafo interattivo salvato: {output_path}")
print("   → Apri in VSCode con 'Live Preview' oppure con il browser")
print("   → Hover su un nodo per vedere il nome e il cluster")
print("   → Scroll per zoom, drag per spostare, click per selezionare")
