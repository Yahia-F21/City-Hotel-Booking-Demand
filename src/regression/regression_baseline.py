

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGRESSION_DATA_DIR = Path(__file__).resolve().parent / "Regression data"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "regression_baseline_results.csv"
RANDOM_STATE = 42
TARGET = "ADR"
POST_BOOKING_FEATURES = ("IsCanceled",)


def load_features(path: Path) -> pd.DataFrame:
    
    features = pd.read_csv(path)
    if features.empty:
        raise ValueError(f"Feature file contains no rows: {path}")

    leaked_features = [name for name in POST_BOOKING_FEATURES if name in features.columns]
    features = features.drop(columns=leaked_features)

    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Features must be numeric; found: {non_numeric}")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(f"Features contain missing or non-finite values: {path}")

    return features


def load_target(path: Path) -> pd.Series:
    
    target_data = pd.read_csv(path)
    if list(target_data.columns) != [TARGET]:
        raise ValueError(
            f"Expected a single {TARGET!r} target column in {path}; "
            f"found {target_data.columns.tolist()}"
        )
    target = target_data[TARGET]
    if target.empty or not np.isfinite(target.to_numpy()).all():
        raise ValueError(f"Target contains no rows or non-finite values: {path}")
    return target


def main() -> None:
    """Fit a linear regression baseline and save held-out test metrics."""
    X_train = load_features(REGRESSION_DATA_DIR / "Regression train data.csv")
    X_test = load_features(REGRESSION_DATA_DIR / "Regression test data.csv")
    y_train = load_target(REGRESSION_DATA_DIR / "Regression train result.csv")
    y_test = load_target(REGRESSION_DATA_DIR / "Regression test result.csv")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test feature columns do not match.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = pd.DataFrame(
        [
            {
                "Model": "LinearRegression",
                "Target": TARGET,
                "PredictionTime": "At booking creation",
                "TrainingSamples": len(X_train),
                "TestSamples": len(X_test),
                "FeatureCount": X_train.shape[1],
                "ExcludedFeatures": ", ".join(POST_BOOKING_FEATURES),
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                "R2": r2_score(y_test, predictions),
            }
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(results.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
