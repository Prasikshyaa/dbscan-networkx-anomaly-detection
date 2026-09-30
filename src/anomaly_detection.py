"""
DBSCAN + NetworkX anomaly detection project.

Run:
    pip install -r requirements.txt
    python src/anomaly_detection.py

Outputs are written to results/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FEATURES = [
    "amount",
    "transactions_24h",
    "unique_merchants_7d",
    "location_changes_7d",
    "device_changes_30d",
]

df = pd.read_csv(DATA_PATH)

# Standardize features so that large-valued variables do not dominate DBSCAN.
X = StandardScaler().fit_transform(df[FEATURES])

# DBSCAN labels dense groups and uses -1 for noise.
model = DBSCAN(eps=1.15, min_samples=7)
df["cluster"] = model.fit_predict(X)
df["dbscan_anomaly"] = df["cluster"] == -1

df.to_csv(RESULTS / "transactions_with_dbscan_labels.csv", index=False)

# Scatter visualization
fig, ax = plt.subplots(figsize=(9, 6))
normal = ~df["dbscan_anomaly"]
ax.scatter(
    df.loc[normal, "amount"],
    df.loc[normal, "transactions_24h"],
    s=35, alpha=0.65, label="Clustered observations"
)
ax.scatter(
    df.loc[df["dbscan_anomaly"], "amount"],
    df.loc[df["dbscan_anomaly"], "transactions_24h"],
    s=65, marker="x", label="DBSCAN noise / potential anomalies"
)
ax.set_title("DBSCAN-Based Transaction Anomaly Detection")
ax.set_xlabel("Transaction amount")
ax.set_ylabel("Transactions in previous 24 hours")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "dbscan_anomalies.png", dpi=180)
plt.close(fig)

# NetworkX transaction graph
G = nx.Graph()

for _, row in df.iterrows():
    tx = row["transaction_id"]
    customer = row["customer_id"]
    merchant = row["merchant_id"]

    G.add_node(customer, node_type="customer")
    G.add_node(merchant, node_type="merchant")
    G.add_node(tx, node_type="transaction",
               anomaly=bool(row["dbscan_anomaly"]))

    G.add_edge(customer, tx)
    G.add_edge(tx, merchant)

nx.write_edgelist(G, RESULTS / "transaction_network.edgelist", data=False)

# Focus the graph on anomaly transactions and their direct neighbors.
anomaly_transactions = set(
    df.loc[df["dbscan_anomaly"], "transaction_id"]
)
nodes_to_show = set(anomaly_transactions)

for tx in anomaly_transactions:
    nodes_to_show.update(G.neighbors(tx))

H = G.subgraph(nodes_to_show).copy()
pos = nx.spring_layout(H, seed=42, k=1.2)

fig, ax = plt.subplots(figsize=(11, 8))

customers = [n for n, d in H.nodes(data=True)
             if d.get("node_type") == "customer"]
merchants = [n for n, d in H.nodes(data=True)
             if d.get("node_type") == "merchant"]
transactions = [n for n, d in H.nodes(data=True)
                if d.get("node_type") == "transaction"]
anomalies = [n for n in transactions
             if H.nodes[n].get("anomaly")]

nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.35)
nx.draw_networkx_nodes(
    H, pos, nodelist=customers, node_size=180,
    node_shape="o", label="Customer"
)
nx.draw_networkx_nodes(
    H, pos, nodelist=merchants, node_size=220,
    node_shape="s", label="Merchant"
)
nx.draw_networkx_nodes(
    H, pos, nodelist=transactions, node_size=130,
    node_shape="^", label="Transaction"
)
nx.draw_networkx_nodes(
    H, pos, nodelist=anomalies, node_size=260,
    node_shape="X", label="DBSCAN anomaly"
)

nx.draw_networkx_labels(
    H, pos,
    labels={n: n for n in anomalies},
    font_size=7
)

ax.set_title("NetworkX View of Transactions Flagged by DBSCAN")
ax.axis("off")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "networkx_anomaly_subgraph.png", dpi=180)
plt.close(fig)

print("Analysis complete.")
print(f"Records analyzed: {len(df)}")
print(f"DBSCAN anomalies/noise: {df['dbscan_anomaly'].sum()}")
print(f"Network nodes: {G.number_of_nodes()}")
print(f"Network edges: {G.number_of_edges()}")
