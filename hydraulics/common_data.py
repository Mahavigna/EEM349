"""
Shared data loading, validation, train/test split, and PCA helpers
for Condition Monitoring of Hydraulic Systems (UCI).
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from typing import Any


DATA_DIR = "."

SENSOR_NAMES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1",
    "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1",
    "CE", "CP", "SE",
]

SEED = 42
TEST_SIZE = 0.20
SWEEP_SIZE = 3000
PCA_VARIANCE_THRESHOLD = 0.95
N_JOBS = -1

CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


def save_current_figure(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def remove_prefix(parameters, prefix):
    return {
        key[len(prefix):]: value
        for key, value in parameters.items()
        if key.startswith(prefix)
    }


def make_model(classifier, use_pca, n_components):
    steps: list[tuple[str, Any]] = [("sc", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=SEED)))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def evaluate_cv(model, X_data, y_data, name):
    scores = cross_val_score(
        model, X_data, y_data, cv=CV, scoring="f1", n_jobs=N_JOBS
    )
    print(f"{name} CV F1: {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores.mean(), scores.std()


def load_hydraulic_dataset():
    """Load 17 sensor files, extract mean/std features, validate, and split."""
    profile_path = os.path.join(DATA_DIR, "profile.txt")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"Could not find {profile_path}. Place the UCI Hydraulic Systems "
            f".txt files in '{DATA_DIR}' or update DATA_DIR."
        )

    profile = pd.read_csv(
        profile_path,
        sep=r"\s+",
        header=None,
        names=["cooler", "valve", "pump", "accumulator", "stable_flag"],
    )

    print("Loading 17 hydraulic sensor files...")
    feature_dict = {}
    feature_columns = []

    for sensor in SENSOR_NAMES:
        sensor_path = os.path.join(DATA_DIR, f"{sensor}.txt")
        if not os.path.exists(sensor_path):
            raise FileNotFoundError(f"Missing sensor file: {sensor_path}")

        raw = np.loadtxt(sensor_path)
        if raw.shape[0] != len(profile):
            raise ValueError(
                f"Row mismatch in {sensor}.txt: {raw.shape[0]} vs {len(profile)}"
            )

        mean_col = f"{sensor}_mean"
        std_col = f"{sensor}_std"
        feature_dict[mean_col] = raw.mean(axis=1)
        feature_dict[std_col] = raw.std(axis=1)
        feature_columns.extend([mean_col, std_col])

    df = pd.DataFrame(feature_dict)
    df["stable_flag"] = profile["stable_flag"].astype(int)

    if df.empty or df[feature_columns].isnull().any().any():
        raise ValueError("Dataset is empty or contains missing values.")

    conflicting = (
        df.groupby(feature_columns, dropna=False)["stable_flag"].nunique().gt(1).sum()
    )
    if conflicting > 0:
        raise ValueError("Conflicting labels found for identical feature vectors.")

    df = df.drop_duplicates(
        subset=feature_columns + ["stable_flag"], keep="first"
    ).reset_index(drop=True)

    X = df[feature_columns].copy()
    y = df["stable_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    actual_sweep_size = min(SWEEP_SIZE, len(X_train))
    if actual_sweep_size < len(X_train):
        X_sw, _, y_sw, _ = train_test_split(
            X_train, y_train, train_size=actual_sweep_size,
            stratify=y_train, random_state=SEED,
        )
    else:
        X_sw, y_sw = X_train.copy(), y_train.copy()

    # Fit PCA on training data only to determine N_COMP
    X_train_scaled = StandardScaler().fit_transform(X_train)
    pca_full = PCA(random_state=SEED).fit(X_train_scaled)
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    n_comp = int(np.where(cumulative_variance >= PCA_VARIANCE_THRESHOLD)[0][0] + 1)
    n_features = X.shape[1]

    print(f"Dataset shape: {df.shape} | Train: {X_train.shape} | Test: {X_test.shape}")
    print(
        f"PCA components for {PCA_VARIANCE_THRESHOLD * 100:.0f}% variance: "
        f"{n_comp} of {n_features}"
    )

    # Save shared PCA variance table and plot
    pd.DataFrame({
        "Principal component": np.arange(1, n_features + 1),
        "Explained variance ratio": explained_variance,
        "Cumulative explained variance": cumulative_variance,
    }).to_csv("pca_explained_variance.csv", index=False)

    plt.figure(figsize=(9, 4.5))
    plt.bar(np.arange(1, n_features + 1), explained_variance, label="Per component")
    plt.plot(np.arange(1, n_features + 1), cumulative_variance, "r-o", markersize=4, label="Cumulative")
    plt.axhline(PCA_VARIANCE_THRESHOLD, linestyle="--", color="gray", label=f"{PCA_VARIANCE_THRESHOLD * 100:.0f}% threshold")
    plt.axvline(n_comp, linestyle=":", color="black", label=f"Selected components = {n_comp}")
    plt.xticks(np.arange(1, n_features + 1, 2))
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA explained variance (Hydraulic Systems)")
    plt.legend()
    save_current_figure("pca_variance.png")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_sw": X_sw,
        "y_sw": y_sw,
        "n_comp": n_comp,
        "n_features": n_features,
    }


def evaluate_pca_vs_non_pca(
    configurations, X_train, X_test, y_train, y_test,
    n_comp, n_features, cm_filename, summary_prefix, extra_info=None
):
    """Fit, time, evaluate, and plot confusion matrices for a 2-config comparison."""
    summary_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for axis, (name, model) in zip(axes, configurations.items()):
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = model.predict(X_test)
        pred_time = time.perf_counter() - t1

        matrix = confusion_matrix(y_test, preds, labels=[0, 1])

        row = {
            "Configuration": name,
            "No. of features": n_comp if "with PCA" in name else n_features,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, preds, pos_label=1, zero_division=0),
            "F1": f1_score(y_test, preds, pos_label=1, zero_division=0),
            "True stable": int(matrix[0, 0]),
            "False unstable": int(matrix[0, 1]),
            "False stable": int(matrix[1, 0]),
            "True unstable": int(matrix[1, 1]),
            "Train time (s)": train_time,
            "Predict time (s)": pred_time,
        }
        if extra_info:
            row.update(extra_info)

        summary_rows.append(row)

        ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=["stable", "unstable"]
        ).plot(ax=axis, colorbar=False, cmap="Blues", values_format="d")
        axis.set_title(name)

    save_current_figure(cm_filename)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{summary_prefix}_summary_full_precision.csv", index=False)

    rounded = summary.copy()
    num_cols = ["Accuracy", "Precision", "Recall", "F1", "Train time (s)", "Predict time (s)"]
    rounded[num_cols] = rounded[num_cols].round(4)
    rounded.to_csv(f"{summary_prefix}_summary.csv", index=False)

    print(f"\nFinal test-set results ({summary_prefix.upper()}):")
    print(rounded.to_string(index=False))
    return summary