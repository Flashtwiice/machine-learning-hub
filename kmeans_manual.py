import matplotlib
matplotlib.use("Agg")

import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/energy_manual.csv"
X_COL = "avg_daily_kwh"
Y_COL = "peak_hour_kwh"
COLORS = ["#0d6efd", "#dc3545", "#198754"]
K = 3
ITERATIONS = 3

INITIAL_CENTROIDS = np.array([[8.0, 3.0], [16.0, 4.0], [28.0, 9.0]])

df = pd.read_csv(CSV_PATH)
POINTS = df[[X_COL, Y_COL]].to_numpy()
RECORD_COUNT = len(df)


def _to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=90)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _plot(labels, centroids, title, previous=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if labels is None:
        ax.scatter(POINTS[:, 0], POINTS[:, 1], color="#6c757d", alpha=0.7, label="Households")
    else:
        for k in range(K):
            mask = labels == k
            ax.scatter(POINTS[mask, 0], POINTS[mask, 1], color=COLORS[k], alpha=0.7, label=f"Cluster {k + 1}")
    if previous is not None:
        ax.scatter(previous[:, 0], previous[:, 1], marker="o", s=90, facecolors="none",
                   edgecolors="#212529", linewidths=1.2, label="Previous centroids")
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=220, c=COLORS[:K],
               edgecolors="black", linewidths=1.5, label="Centroids")
    ax.set_title(title)
    ax.set_xlabel("Average Daily Consumption (kWh)")
    ax.set_ylabel("Peak-Hour Consumption (kWh)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _to_base64(fig)


def _run():
    initial_plot = _plot(None, INITIAL_CENTROIDS, "Households and Initial Centroids")
    centroids = INITIAL_CENTROIDS.copy()
    iterations = []
    for n in range(1, ITERATIONS + 1):
        # 1. Euclidean distance from every record to every centroid
        distances = np.linalg.norm(POINTS[:, None, :] - centroids[None, :, :], axis=2)
        # 2. Assign each record to the nearest centroid
        labels = distances.argmin(axis=1)
        # 3. New centroids = mean of the records assigned to each cluster
        new_centroids = centroids.copy()
        for k in range(K):
            if np.any(labels == k):
                new_centroids[k] = POINTS[labels == k].mean(axis=0)
        # 4. Within-cluster variance (mean squared distance to the updated centroid)
        cluster_stats = []
        total_ss = 0.0
        for k in range(K):
            members = POINTS[labels == k]
            ss = float(((members - new_centroids[k]) ** 2).sum()) if len(members) else 0.0
            total_ss += ss
            cluster_stats.append({
                "cluster": k + 1,
                "count": int(len(members)),
                "cx": round(float(new_centroids[k][0]), 2),
                "cy": round(float(new_centroids[k][1]), 2),
                "old_cx": round(float(centroids[k][0]), 2),
                "old_cy": round(float(centroids[k][1]), 2),
                "variance": round(ss / len(members), 3) if len(members) else 0.0,
                "ss": round(ss, 2),
            })
        rows = [
            {
                "id": df["household_id"].iloc[i],
                "x": float(POINTS[i, 0]),
                "y": float(POINTS[i, 1]),
                "d1": round(float(distances[i, 0]), 2),
                "d2": round(float(distances[i, 1]), 2),
                "d3": round(float(distances[i, 2]), 2),
                "cluster": int(labels[i]) + 1,
            }
            for i in range(RECORD_COUNT)
        ]
        iterations.append({
            "n": n,
            "rows": rows,
            "clusters": cluster_stats,
            "wcss": round(total_ss, 2),
            "plot": _plot(labels, new_centroids, f"Iteration {n}: Clusters and Updated Centroids", previous=centroids),
        })
        centroids = new_centroids
    return initial_plot, iterations, centroids


INITIAL_PLOT, ITERATION_RESULTS, FINAL_CENTROIDS = _run()
INITIAL_CENTROID_ROWS = [
    {"cluster": k + 1, "x": float(INITIAL_CENTROIDS[k][0]), "y": float(INITIAL_CENTROIDS[k][1])}
    for k in range(K)
]
DESCRIBE = df[[X_COL, Y_COL]].describe().round(2).to_dict()
