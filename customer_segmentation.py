"""
Customer Segmentation using K-Means Clustering
================================================
Features:
  - RFM feature engineering (Recency, Frequency, Monetary)
  - Demographic features (Age, Region, Gender)
  - Purchase category preferences
  - Optimal k selection via Elbow + Silhouette
  - PCA for 2D scatter visualization
  - Segment profiling & export
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC CUSTOMER DATA
# ─────────────────────────────────────────────

np.random.seed(42)
N = 1200

def make_customers(n):
    today = datetime(2024, 6, 1)

    # Simulate 4 natural clusters
    segments = np.random.choice([0, 1, 2, 3], size=n, p=[0.24, 0.31, 0.27, 0.18])

    data = []
    for s in segments:
        if s == 0:   # Champions
            age        = np.random.randint(26, 44)
            recency    = np.random.randint(1, 14)
            frequency  = np.random.randint(10, 22)
            monetary   = round(np.random.uniform(250, 450), 2)
            electronics= np.random.randint(60, 100)
            fashion    = np.random.randint(40, 80)
            food       = np.random.randint(20, 60)
            home       = np.random.randint(20, 55)
            beauty     = np.random.randint(20, 50)
            gender     = np.random.choice(["M", "F", "Other"], p=[0.45, 0.48, 0.07])
            region     = np.random.choice(["North", "South", "East", "West"], p=[0.3, 0.2, 0.3, 0.2])
        elif s == 1: # Loyalists
            age        = np.random.randint(35, 56)
            recency    = np.random.randint(10, 30)
            frequency  = np.random.randint(6, 14)
            monetary   = round(np.random.uniform(120, 220), 2)
            electronics= np.random.randint(20, 55)
            fashion    = np.random.randint(40, 70)
            food       = np.random.randint(55, 90)
            home       = np.random.randint(60, 95)
            beauty     = np.random.randint(25, 55)
            gender     = np.random.choice(["M", "F", "Other"], p=[0.38, 0.55, 0.07])
            region     = np.random.choice(["North", "South", "East", "West"], p=[0.25, 0.3, 0.2, 0.25])
        elif s == 2: # At-risk
            age        = np.random.randint(22, 42)
            recency    = np.random.randint(40, 80)
            frequency  = np.random.randint(2, 7)
            monetary   = round(np.random.uniform(50, 130), 2)
            electronics= np.random.randint(40, 75)
            fashion    = np.random.randint(55, 90)
            food       = np.random.randint(20, 55)
            home       = np.random.randint(15, 45)
            beauty     = np.random.randint(45, 80)
            gender     = np.random.choice(["M", "F", "Other"], p=[0.42, 0.5, 0.08])
            region     = np.random.choice(["North", "South", "East", "West"], p=[0.2, 0.3, 0.25, 0.25])
        else:        # Dormant
            age        = np.random.randint(40, 66)
            recency    = np.random.randint(100, 180)
            frequency  = np.random.randint(1, 3)
            monetary   = round(np.random.uniform(10, 60), 2)
            electronics= np.random.randint(10, 40)
            fashion    = np.random.randint(15, 45)
            food       = np.random.randint(35, 65)
            home       = np.random.randint(40, 75)
            beauty     = np.random.randint(10, 40)
            gender     = np.random.choice(["M", "F", "Other"], p=[0.48, 0.45, 0.07])
            region     = np.random.choice(["North", "South", "East", "West"])

        data.append({
            "customer_id": f"CUST{len(data)+1:05d}",
            "age":         age,
            "gender":      gender,
            "region":      region,
            "recency_days":recency,
            "frequency":   frequency,
            "monetary_avg":monetary,
            "cat_electronics": electronics,
            "cat_fashion":  fashion,
            "cat_food":     food,
            "cat_home":     home,
            "cat_beauty":   beauty,
            "true_segment": s,
        })
    return pd.DataFrame(data)

df = make_customers(N)
print(f"Dataset shape: {df.shape}")
print(df.head())

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

features = [
    "age", "recency_days", "frequency", "monetary_avg",
    "cat_electronics", "cat_fashion", "cat_food", "cat_home", "cat_beauty"
]

X_raw = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

print("\nScaled feature stats:")
print(X_scaled_df.describe().round(2))

# ─────────────────────────────────────────────
# 3. OPTIMAL K — ELBOW + SILHOUETTE
# ─────────────────────────────────────────────

k_range = range(2, 9)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Optimal k Selection", fontsize=14, fontweight="bold")

axes[0].plot(k_range, inertias, "o-", color="#534AB7", linewidth=2)
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow method")
axes[0].grid(alpha=0.3)

axes[1].plot(k_range, sil_scores, "s-", color="#1D9E75", linewidth=2)
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette scores")
axes[1].grid(alpha=0.3)

best_k = k_range[np.argmax(sil_scores)]
axes[1].axvline(best_k, color="#D85A30", linestyle="--", label=f"Best k={best_k}")
axes[1].legend()

plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nBest k by silhouette: {best_k}  (score={max(sil_scores):.3f})")

# ─────────────────────────────────────────────
# 4. FIT FINAL K-MEANS MODEL
# ─────────────────────────────────────────────

K = 4  # use domain knowledge: 4 segments
kmeans = KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=500)
df["cluster"] = kmeans.fit_predict(X_scaled)

final_sil = silhouette_score(X_scaled, df["cluster"])
print(f"\nFinal model — k={K}, silhouette={final_sil:.3f}")

# ─────────────────────────────────────────────
# 5. MAP CLUSTER IDs → SEGMENT NAMES
# ─────────────────────────────────────────────

# Sort clusters by descending monetary value to assign meaningful names
cluster_means = df.groupby("cluster")["monetary_avg"].mean().sort_values(ascending=False)
rank_to_label = {
    0: "Champions",
    1: "Loyalists",
    2: "At-risk",
    3: "Dormant",
}
cluster_id_to_name = {cid: rank_to_label[rank] for rank, cid in enumerate(cluster_means.index)}
df["segment"] = df["cluster"].map(cluster_id_to_name)
print("\nSegment distribution:")
print(df["segment"].value_counts())

# ─────────────────────────────────────────────
# 6. SEGMENT PROFILES
# ─────────────────────────────────────────────

profile_cols = ["age", "recency_days", "frequency", "monetary_avg",
                "cat_electronics", "cat_fashion", "cat_food", "cat_home", "cat_beauty"]

profile = df.groupby("segment")[profile_cols].mean().round(1)
profile["count"] = df["segment"].value_counts()
profile["pct"] = (profile["count"] / N * 100).round(1)

print("\n── Segment Profiles ──────────────────────────────────")
print(profile.to_string())

# ─────────────────────────────────────────────
# 7. PCA SCATTER VISUALIZATION
# ─────────────────────────────────────────────

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
df["pca1"] = pca_coords[:, 0]
df["pca2"] = pca_coords[:, 1]

COLORS = {
    "Champions": "#534AB7",
    "Loyalists": "#1D9E75",
    "At-risk":   "#D85A30",
    "Dormant":   "#888780",
}

fig, ax = plt.subplots(figsize=(9, 6))
for seg, grp in df.groupby("segment"):
    ax.scatter(grp["pca1"], grp["pca2"],
               c=COLORS[seg], alpha=0.55, s=18, label=seg)

# Plot centroids in PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)
for cid, (x, y) in enumerate(centroids_pca):
    seg_name = cluster_id_to_name[cid]
    ax.scatter(x, y, c=COLORS[seg_name], s=200, marker="*",
               edgecolors="white", linewidths=0.8, zorder=5)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("Customer Segments — PCA Projection", fontsize=13, fontweight="bold")
ax.legend(title="Segment", framealpha=0.85)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("segment_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (CLUSTER CENTERS)
# ─────────────────────────────────────────────

centers_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)
centers_df.index = [cluster_id_to_name[i] for i in range(K)]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Segment Characteristics (cluster centers)", fontsize=13, fontweight="bold")

rfm_features = ["recency_days", "frequency", "monetary_avg"]
cat_features  = ["cat_electronics", "cat_fashion", "cat_food", "cat_home", "cat_beauty"]

for ax, seg in zip(axes.flat, ["Champions", "Loyalists", "At-risk", "Dormant"]):
    vals = centers_df.loc[seg, rfm_features + ["age"]]
    bars = ax.bar(["Recency (days)", "Frequency", "Avg spend ($)", "Age"],
                  vals, color=COLORS[seg], alpha=0.8, edgecolor="white")
    ax.set_title(seg, fontweight="bold", color=COLORS[seg])
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("segment_centers.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# 9. EXPORT RESULTS
# ─────────────────────────────────────────────

output_cols = ["customer_id", "age", "gender", "region",
               "recency_days", "frequency", "monetary_avg",
               "cluster", "segment"]
df[output_cols].to_csv("customer_segments.csv", index=False)
profile.to_csv("segment_profiles.csv")

print("\n✓ Outputs saved:")
print("  customer_segments.csv  — full data with segment labels")
print("  segment_profiles.csv   — aggregated segment statistics")
print("  elbow_silhouette.png   — k selection charts")
print("  segment_scatter.png    — PCA 2D cluster plot")
print("  segment_centers.png    — per-segment feature bars")

print("\n── Done ───────────────────────────────────────────────")
