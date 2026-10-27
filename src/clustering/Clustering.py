from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "H2.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load the City Hotel dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # 2. Select initial numerical clustering features
    features = [
        "LeadTime",
        "StaysInWeekendNights",
        "StaysInWeekNights",
        "Adults",
        "Children",
        "PreviousBookingsNotCanceled",
    ]

    # Confirm that the required columns exist 
    missing_columns = [
        column for column in features
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    X = df[features].copy()

    # Optional preprocessing will be done outside this script.
    # Keep the data in a clean matrix for clustering.
    X_processed = X.copy()

    # 3. Try several cluster counts
    results = []
    best_silhouette = float("-inf")
    best_k = None
    best_labels = None

    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X_processed)

        # Silhouette requires at least 2 clusters
        # and fewer clusters than observations.
        silhouette = silhouette_score(
            X_processed,
            labels,
            sample_size=min(10000, len(X_processed)),
            random_state=42,
        )

        db_index = davies_bouldin_score(
            X_processed,
            labels,
        )

        results.append({
            "Algorithm": "KMeans",
            "NumberOfClusters": k,
            "SilhouetteScore": silhouette,
            "DaviesBouldinIndex": db_index,
            "Inertia": model.inertia_,
        })

        print(
            f"k={k} | "
            f"Silhouette={silhouette:.4f} | "
            f"Davies-Bouldin={db_index:.4f}"
        )

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
            best_labels = labels.copy()

    # 4. Save the metric results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "clustering_baseline_results.csv",
        index=False,
    )

    # 5. Summarize the selected clusters
    clustered_data = X_processed.copy()
    clustered_data["Cluster"] = best_labels

    print("\nSelected cluster count:", best_k)
    print("\nCluster sizes:")
    print(clustered_data["Cluster"].value_counts().sort_index())

    print("\nCluster feature averages:")
    print(
        clustered_data.groupby("Cluster")[features]
        .mean()
        .round(2)
    )

    # 6. Reduce features to two dimensions for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_processed)

    plt.figure(figsize=(9, 6))

    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=best_labels,
        s=8,
        alpha=0.5,
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"K-Means Clusters (k={best_k})")
    plt.legend(
        *scatter.legend_elements(),
        title="Cluster",
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "cluster_visualization.png",
        dpi=200,
    )

    plt.close()

    print("\nClustering results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
