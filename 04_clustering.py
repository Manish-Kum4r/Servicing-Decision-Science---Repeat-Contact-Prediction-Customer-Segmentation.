"""
04_clustering.py — K-Means customer segmentation on the SQL-derived
customer feature view (v_customer_features).
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["tenure_months", "n_products", "total_interactions",
            "avg_handle_time_min", "fcr_rate", "avg_severity",
            "billing_payments_share", "phone_share"]

conn = sqlite3.connect(DATA / "gsds_servicing.db")
cust = pd.read_sql_query("SELECT * FROM v_customer_features", conn)
conn.close()

X = cust[FEATURES].fillna(cust[FEATURES].median())
Xs = StandardScaler().fit_transform(X)

# elbow method to choose k
inertias = []
ks = range(2, 9)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    inertias.append(km.inertia_)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(ks), inertias, "o-", color="#1F4E79")
ax.set_xlabel("k (number of clusters)"); ax.set_ylabel("inertia (WCSS)")
ax.set_title("Elbow method — customer segmentation")
fig.tight_layout(); fig.savefig(OUT / "cluster_elbow.png", dpi=130)
plt.close(fig)

# final model
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Xs)
cust["cluster"] = km.labels_

profiles = cust.groupby("cluster")[FEATURES].mean().round(3)
profiles.insert(0, "customers", cust.groupby("cluster").size())
profiles.insert(1, "share", (cust.groupby("cluster").size() / len(cust)).round(3))
profiles.to_csv(OUT / "cluster_profiles.csv")

# PCA (2 components) for visualisation only
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(Xs)
fig, ax = plt.subplots(figsize=(7, 5.5))
for c in range(K):
    m = cust["cluster"].values == c
    ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.5, label=f"cluster {c} (n={m.sum()})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
ax.set_title("Customer segments — K-Means (k=4), PCA view")
ax.legend()
fig.tight_layout(); fig.savefig(OUT / "cluster_pca.png", dpi=130)
plt.close(fig)

print(f"Customers segmented: {len(cust):,} | k = {K}")
print(profiles.to_string())
