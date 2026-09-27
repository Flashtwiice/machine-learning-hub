import matplotlib
matplotlib.use("Agg")

import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CSV_PATH = "data/energy_consumption.csv"
FEATURES = ["avg_daily_kwh", "peak_hour_kwh"]
K = 3
COLORS = ["#0d6efd", "#dc3545", "#198754"]
KMEANS_CONFIG = {
    "n_clusters": K,
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# ---------------- Preprocessing ----------------
raw = pd.read_csv(CSV_PATH)
PREPROCESSING = []


def _log(step, removed, remaining, detail):
    PREPROCESSING.append({"step": step, "removed": removed, "remaining": remaining, "detail": detail})


_log("Load raw CSV", 0, len(raw), "Records read with Pandas from data/energy_consumption.csv.")

df = raw.drop_duplicates()
_log("Remove duplicate records", len(raw) - len(df), len(df),
     "Smart meters occasionally transmit the same reading twice; exact duplicates were dropped.")

before = len(df)
df = df.dropna(subset=FEATURES)
_log("Remove records with missing values", before - len(df), len(df),
     "Rows with a missing average or peak-hour reading cannot be placed in the feature space.")

before = len(df)
df = df[(df[FEATURES] > 0).all(axis=1) & (df["peak_hour_kwh"] <= df["avg_daily_kwh"])]
_log("Remove physically invalid readings", before - len(df), len(df),
     "Peak-hour consumption is part of the daily total, so it cannot exceed it; values must also be positive.")

df = df.reset_index(drop=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])
_log("Standardize variables (StandardScaler)", 0, len(df),
     "Both variables are rescaled to mean 0 and standard deviation 1 so neither dominates the Euclidean distance.")

RAW_COUNT = len(raw)
RECORD_COUNT = len(df)

# ---------------- Model ----------------
model = KMeans(**KMEANS_CONFIG)
_raw_labels = model.fit_predict(X_scaled)

# Renumber clusters so cluster 1 is always the lowest-consumption group.
_centroids_orig = scaler.inverse_transform(model.cluster_centers_)
_order = np.argsort(_centroids_orig[:, 0])
_remap = {int(old): new for new, old in enumerate(_order)}
labels = np.array([_remap[int(l)] for l in _raw_labels])
centroids = _centroids_orig[_order]
df["cluster"] = labels + 1

SILHOUETTE = round(float(silhouette_score(X_scaled, labels)), 3)
INERTIA = round(float(model.inertia_), 2)
N_ITER = int(model.n_iter_)

K_COMPARISON = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    lab = km.fit_predict(X_scaled)
    K_COMPARISON.append({
        "k": k,
        "inertia": round(float(km.inertia_), 1),
        "silhouette": round(float(silhouette_score(X_scaled, lab)), 3),
    })


def interpret_silhouette(score):
    if score >= 0.71:
        return "strong structure"
    if score >= 0.51:
        return "reasonable structure"
    if score >= 0.26:
        return "weak structure"
    return "no substantial structure"


def _cluster_summary():
    names = ["Low-consumption households", "Evening-peak households", "High-consumption households"]
    rows = []
    for k in range(K):
        members = df[df["cluster"] == k + 1]
        avg = float(centroids[k][0])
        peak = float(centroids[k][1])
        rows.append({
            "cluster": k + 1,
            "count": int(len(members)),
            "share": round(len(members) / RECORD_COUNT * 100, 1),
            "avg": round(avg, 2),
            "peak": round(peak, 2),
            "peak_share": round(peak / avg * 100, 1),
            "name": names[k],
        })
    return rows


CLUSTER_SUMMARY = _cluster_summary()

TABLE_ROWS = [
    {
        "id": r.household_id,
        "avg": float(r.avg_daily_kwh),
        "peak": float(r.peak_hour_kwh),
        "cluster": int(r.cluster),
    }
    for r in df.itertuples()
]


def generate_plot():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k in range(K):
        members = df[df["cluster"] == k + 1]
        ax.scatter(members["avg_daily_kwh"], members["peak_hour_kwh"], color=COLORS[k],
                   alpha=0.5, s=18, label=f"Cluster {k + 1}: {CLUSTER_SUMMARY[k]['name']}")
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=260, c=COLORS,
               edgecolors="black", linewidths=1.6, label="Centroids")
    ax.set_title("K-Means Clusters of Household Energy Consumption")
    ax.set_xlabel("Average Daily Consumption (kWh)")
    ax.set_ylabel("Peak-Hour Consumption (kWh)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=90)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
