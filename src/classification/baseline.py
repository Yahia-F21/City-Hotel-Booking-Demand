import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_DATA_DIR = Path(__file__).resolve().parent / "Classification data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "classification_baseline_results.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
RANDOM_STATE = 42
TARGET = "IsCanceled"
THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def build_candidate_models():
    """Return a few simple, low-risk classifiers for validation-based selection."""
    return {
        "balanced_logit": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "plain_logit": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
            ]
        ),
    }


def load_features(path: Path) -> pd.DataFrame:
    """Load the feature matrix and ensure it is numeric and finite."""
    features = pd.read_csv(path)
    if features.empty:
        raise ValueError(f"Feature file contains no rows: {path}")

    if TARGET in features.columns:
        features = features.drop(columns=[TARGET])

    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Features must be numeric; found: {non_numeric}")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(f"Features contain missing or non-finite values: {path}")

    return features


def load_target(path: Path) -> pd.Series:
    """Load the single-column target series."""
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


def select_best_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Choose the model and decision threshold using only a validation split from train data."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    best_record = None
    best_model_name = None
    best_model = None

    for model_name, model in build_candidate_models().items():
        model.fit(X_fit, y_fit)
        probs = model.predict_proba(X_val)[:, 1]

        for threshold in THRESHOLDS:
            predictions = (probs >= threshold).astype(int)
            record = {
                "model_name": model_name,
                "threshold": threshold,
                "accuracy": accuracy_score(y_val, predictions),
                "precision": precision_score(y_val, predictions, zero_division=0),
                "recall": recall_score(y_val, predictions, zero_division=0),
                "f1": f1_score(y_val, predictions),
            }

            if best_record is None or record["f1"] > best_record["f1"]:
                best_record = record
                best_model_name = model_name
                best_model = model

    if best_model is None or best_record is None:
        raise RuntimeError("No valid model was selected during validation.")

    return best_model_name, best_record["threshold"], best_record


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """Save a confusion matrix visualization for the test set."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ["Not canceled", "Canceled"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("IsCanceled confusion matrix")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = cm[row, col]
            color = "white" if value > cm.max() / 2 else "black"
            ax.text(col, row, format(value, "d"), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    """Fit a model selected by validation and evaluate it once on the hold-out test set."""
    X_train = load_features(CLASSIFICATION_DATA_DIR / "Classification train data.csv")
    X_test = load_features(CLASSIFICATION_DATA_DIR / "Classification test data.csv")
    y_train = load_target(CLASSIFICATION_DATA_DIR / "Classification train result.csv")
    y_test = load_target(CLASSIFICATION_DATA_DIR / "Classification test result.csv")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test feature columns do not match.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")

    model_name, chosen_threshold, validation_metrics = select_best_model(X_train, y_train)

    final_model = build_candidate_models()[model_name]
    final_model.fit(X_train, y_train)

    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    predictions = (test_probabilities >= chosen_threshold).astype(int)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    results = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Target": TARGET,
                "PredictionTime": "At booking creation",
                "ValidationF1": validation_metrics["f1"],
                "ValidationAccuracy": validation_metrics["accuracy"],
                "ValidationPrecision": validation_metrics["precision"],
                "ValidationRecall": validation_metrics["recall"],
                "DecisionThreshold": chosen_threshold,
                "TrainingSamples": len(X_train),
                "TestSamples": len(X_test),
                "FeatureCount": X_train.shape[1],
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1]),
            }
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    save_confusion_matrix(y_test, predictions)

    print(results.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
