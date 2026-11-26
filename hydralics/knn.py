"""
k-NN Classification on Condition Monitoring of Hydraulic Systems:
Hyperparameter sweep + PCA vs. non-PCA evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from common_data import (
    SEED,
    CV,
    N_JOBS,
    PCA_VARIANCE_THRESHOLD,
    load_hydraulic_dataset,
    make_model,
    remove_prefix,
    evaluate_cv,
    evaluate_pca_vs_non_pca,
    save_current_figure,
)

K_VALUES = list(range(1, 32, 2))
KNN_METRICS = ["euclidean", "manhattan"]
KNN_WEIGHTS = ["uniform", "distance"]


def main():
    data = load_hydraulic_dataset()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    X_sw, y_sw = data["X_sw"], data["y_sw"]
    n_comp, n_features = data["n_comp"], data["n_features"]

    # 1. k-NN hyperparameter sweep
    knn_pipeline = Pipeline(
        [
            ("sc", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )

    knn_grid = GridSearchCV(
        estimator=knn_pipeline,
        param_grid={
            "knn__n_neighbors": K_VALUES,
            "knn__metric": KNN_METRICS,
            "knn__weights": KNN_WEIGHTS,
        },
        cv=CV,
        scoring="f1",
        n_jobs=N_JOBS,
        pre_dispatch="2*n_jobs",
        refit=True,
        verbose=1,
    )
    knn_grid.fit(X_sw, y_sw)

    best_knn_params = remove_prefix(knn_grid.best_params_, "knn__")
    print("\nBest k-NN parameters:", best_knn_params)
    print(f"Best k-NN CV F1: {knn_grid.best_score_:.4f}")

    knn_results = pd.DataFrame(knn_grid.cv_results_)
    knn_results.to_csv("knn_grid_search_results.csv", index=False)

    # Plot k-NN sweep
    plt.figure(figsize=(8, 5))
    styles = {
        ("euclidean", "uniform"): ("-", "o"),
        ("euclidean", "distance"): ("--", "s"),
        ("manhattan", "uniform"): ("-.", "^"),
        ("manhattan", "distance"): (":", "D"),
    }

    for metric in KNN_METRICS:
        for weights in KNN_WEIGHTS:
            sub = knn_results[
                (knn_results["param_knn__metric"] == metric)
                & (knn_results["param_knn__weights"] == weights)
            ].copy()
            sub["k"] = sub["param_knn__n_neighbors"].astype(int)
            sub = sub.sort_values("k")
            ls, mk = styles[(metric, weights)]
            plt.plot(
                sub["k"],
                sub["mean_test_score"],
                marker=mk,
                linestyle=ls,
                label=f"{metric}, {weights}",
            )

    plt.xlabel("Number of neighbours, k")
    plt.ylabel("Mean cross-validation F1")
    plt.title("k-NN parameter sweep")
    plt.xticks(K_VALUES, rotation=90)
    plt.grid(alpha=0.25)
    plt.legend()
    save_current_figure("knn_k_sweep.png")

    # 2. CV stability on full training set
    evaluate_cv(
        make_model(KNeighborsClassifier(**best_knn_params), False, n_comp),
        X_train,
        y_train,
        "Selected k-NN (without PCA)",
    )
    evaluate_cv(
        make_model(KNeighborsClassifier(**best_knn_params), True, n_comp),
        X_train,
        y_train,
        "Selected k-NN (with PCA)",
    )

    # 3. PCA component sweep for k-NN
    comp_values = list(range(1, n_features + 1))
    means, stds = [], []
    for k_comp in comp_values:
        pipe = Pipeline(
            [
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=k_comp, random_state=SEED)),
                ("clf", KNeighborsClassifier(**best_knn_params)),
            ]
        )
        sc = cross_val_score(pipe, X_sw, y_sw, cv=CV, scoring="f1", n_jobs=N_JOBS)
        means.append(sc.mean())
        stds.append(sc.std())

    pd.DataFrame(
        {
            "PCA components": comp_values,
            "k-NN mean CV F1": means,
            "k-NN CV F1 std": stds,
        }
    ).to_csv("knn_pca_component_sweep.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(comp_values, means, "s-", markersize=4, label="k-NN")
    plt.axvline(
        n_comp,
        linestyle="--",
        color="gray",
        label=f"{PCA_VARIANCE_THRESHOLD * 100:.0f}% variance ({n_comp})",
    )
    plt.xticks(np.arange(1, n_features + 1, 2))
    plt.xlabel("Number of PCA components")
    plt.ylabel("Mean cross-validation F1")
    plt.title("k-NN: effect of PCA dimensionality")
    plt.grid(alpha=0.25)
    plt.legend()
    save_current_figure("knn_pca_component_sweep.png")

    # 4. Final test-set comparison: without PCA vs. with PCA
    configs = {
        "k-NN - without PCA": make_model(
            KNeighborsClassifier(**best_knn_params), False, n_comp
        ),
        "k-NN - with PCA": make_model(
            KNeighborsClassifier(**best_knn_params), True, n_comp
        ),
    }
    evaluate_pca_vs_non_pca(
        configs,
        X_train,
        X_test,
        y_train,
        y_test,
        n_comp,
        n_features,
        cm_filename="knn_confusion_matrices.png",
        summary_prefix="knn",
    )


if __name__ == "__main__":
    main()

