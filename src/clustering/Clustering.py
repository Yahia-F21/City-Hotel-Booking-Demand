"""
Task C: Baseline Clustering Pipeline (Milestone 1)
CSE382 - Introduction to Machine Learning: City Hotel Booking Demand

This script focuses purely on unsupervised clustering:
- Consumes the preprocessed data from 'Clustering data/Clustering train data.csv'
- Contains NO redundant data preprocessing (handled upstream by preprocessing_data.py)
- Evaluates K-Means baseline across k in range [2, 7]
- Computes internal evaluation criteria: Inertia (Elbow), Silhouette Score, and Davies-Bouldin Index
- Selects the optimal baseline k
- Generates 2D PCA cluster visualizations and evaluation curve plots
- Saves metric tables, figures, and serialized model artifacts
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Paths setup
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parents[1]  # City-Hotel-Booking-Demand root

# Directly use the preprocessed clustering data
TRAIN_DATA_PATH = CURRENT_DIR / "Clustering data" / "Clustering train data.csv"
OUTPUT_DIR = ROOT / "outputs"
ARTIFACTS_DIR = ROOT / "artifacts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_evaluation_curves(results_df: pd.DataFrame, best_k: int, output_path: Path):
    """Plots Inertia (Elbow Method), Silhouette Score, and Davies-Bouldin Index vs k."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Inertia / Elbow Method
    ax1.plot(results_df["NumberOfClusters"], results_df["Inertia"], marker="o", color="#2E86AB", lw=2)
    ax1.axvline(best_k, color="#D1495B", linestyle="--", alpha=0.8, label=f"Selected k={best_k}")
    ax1.set_title("Elbow Method (Inertia vs k)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 2. Silhouette and Davies-Bouldin Scores
    color_sil = "#2B9348"
    color_db = "#D1495B"

    ax2.plot(results_df["NumberOfClusters"], results_df["SilhouetteScore"], marker="s", color=color_sil, lw=2, label="Silhouette Score")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score (Higher is better)", color=color_sil)
    ax2.tick_params(axis="y", labelcolor=color_sil)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(results_df["NumberOfClusters"], results_df["DaviesBouldinIndex"], marker="^", color=color_db, lw=2, label="Davies-Bouldin")
    ax2_twin.set_ylabel("Davies-Bouldin Index (Lower is better)", color=color_db)
    ax2_twin.tick_params(axis="y", labelcolor=color_db)

    ax2.set_title("Validation Metrics vs k", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved evaluation curves to: {output_path}")


def plot_2d_clusters(X: pd.DataFrame, labels: np.ndarray, centroids: np.ndarray, k: int, output_path: Path):
    """Plots 2D PCA projection of the clusters using PC1 and PC2 with centroids."""
    plt.figure(figsize=(9, 6))

    # Subsample points for a clear and lightweight visualization
    sample_size = min(12000, len(X))
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(X), size=sample_size, replace=False)

    x_sub = X.iloc[sample_indices]
    labels_sub = labels[sample_indices]

    scatter = plt.scatter(
        x_sub["PC1"],
        x_sub["PC2"],
        c=labels_sub,
        cmap="tab10",
        s=12,
        alpha=0.45,
        edgecolors="none"
    )

    # Overlay cluster centroids on PC1 and PC2
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=160,
        c="black",
        edgecolors="white",
        linewidth=1.5,
        label="Centroids"
    )

    plt.xlabel("Principal Component 1 (PC1)", fontsize=11)
    plt.ylabel("Principal Component 2 (PC2)", fontsize=11)
    plt.title(f"K-Means Baseline Clusters on PC1 vs PC2 (k={k})", fontsize=12, fontweight="bold")
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved 2D cluster visualization to: {output_path}")


def main():
    print("=" * 70)
    print("CSE382 Milestone 1: Task C - Clustering Baseline")
    print("=" * 70)

    # 1. Load the preprocessed clustering data directly
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clustering data not found at: {TRAIN_DATA_PATH}\n"
            f"Please run 'src/preprocessing/preprocessing_data.py' first to generate it."
        )

    print(f"[1/5] Loading preprocessed data from: {TRAIN_DATA_PATH.name}")
    X = pd.read_csv(TRAIN_DATA_PATH)
    print(f"      Data shape: {X.shape} ({X.shape[0]:,} samples, {X.shape[1]} features)")

    # 2. Iterate through k=2..7 and compute metrics
    print("\n[2/5] Evaluating K-Means baseline across k in range [2, 7]...")
    results = []
    fitted_models = {}

    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Compute internal evaluation metrics
        # Using a sample of 3,000 for silhouette score ensures fast execution without memory issues
        sample_size = min(3000, len(X))
        sil_score = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
        db_index = davies_bouldin_score(X, labels)

        results.append({
            "Algorithm": "K-Means",
            "NumberOfClusters": k,
            "Inertia": round(kmeans.inertia_, 2),
            "SilhouetteScore": round(sil_score, 4),
            "DaviesBouldinIndex": round(db_index, 4)
        })

        fitted_models[k] = (kmeans, labels)
        print(f"      k={k} | Silhouette: {sil_score:.4f} | Davies-Bouldin: {db_index:.4f} | Inertia: {kmeans.inertia_:.1f}")

    results_df = pd.DataFrame(results)

    # 3. Save baseline evaluation table
    baseline_csv_path = OUTPUT_DIR / "clustering_baseline_results.csv"
    results_df.to_csv(baseline_csv_path, index=False)
    print(f"\n[3/5] Baseline metrics saved to: {baseline_csv_path}")

    # 4. Select best baseline k (highest Silhouette Score and lowest Davies-Bouldin Index)
    best_row = results_df.sort_values(by=["SilhouetteScore", "DaviesBouldinIndex"], ascending=[False, True]).iloc[0]
    best_k = int(best_row["NumberOfClusters"])
    best_kmeans, best_labels = fitted_models[best_k]

    print(f"\n[4/5] Selected optimal baseline: k = {best_k}")
    print(f"      Silhouette Score: {best_row['SilhouetteScore']} | Davies-Bouldin Index: {best_row['DaviesBouldinIndex']}")

    # Display cluster distribution
    print("\n      Cluster Distribution:")
    unique, counts = np.unique(best_labels, return_counts=True)
    for u, c in zip(unique, counts):
        pct = (c / len(best_labels)) * 100
        print(f"        - Cluster {u}: {c:,} bookings ({pct:.2f}%)")

    # 5. Generate and save diagnostic visualizations
    print("\n[5/5] Generating diagnostic plots...")
    curves_plot_path = OUTPUT_DIR / "clustering_evaluation_curves.png"
    plot_evaluation_curves(results_df, best_k, curves_plot_path)

    cluster_plot_path = OUTPUT_DIR / "cluster_visualization.png"
    plot_2d_clusters(X, best_labels, best_kmeans.cluster_centers_, best_k, cluster_plot_path)

    # Save model artifact
    artifact_path = ARTIFACTS_DIR / "kmeans_baseline.joblib"
    joblib.dump(best_kmeans, artifact_path)
    print(f"\nSaved trained K-Means artifact to: {artifact_path}")

    print("=" * 70)
    print("Baseline Clustering Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
