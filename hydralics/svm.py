"""
SVM Classification on Condition Monitoring of Hydraulic Systems:
RBF vs. Polynomial kernel sweep + PCA vs. non-PCA evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

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

SVM_C_VALUES = [0.01, 0.1, 1, 10, 100]
RBF_GAMMA_VALUES = [0.001, 0.01, 0.1, 1]

POLY_C_VALUES = [0.1, 1, 10, 100]
POLY_DEGREES = [2, 3, 4]
POLY_GAMMA_VALUES = ["scale", 0.01, 0.1]
POLY_COEF0_VALUES = [0.0, 1.0]


def main():
    data = load_hydraulic_dataset()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    X_sw, y_sw = data["X_sw"], data["y_sw"]
    n_comp, n_features = data["n_comp"], data["n_features"]

    # 1. RBF SVM sweep
    rbf_grid = GridSearchCV(
        estimator=Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf"))]),
        param_grid={"svm__C": SVM_C_VALUES, "svm__gamma": RBF_GAMMA_VALUES},
        cv=CV, scoring="f1", n_jobs=N_JOBS, pre_dispatch="2*n_jobs", refit=True, verbose=1,
    )
    rbf_grid.fit(X_sw, y_sw)
    print("\nBest RBF SVM:", rbf_grid.best_params_, f"CV F1: {rbf_grid.best_score_:.4f}")

    rbf_results = pd.DataFrame(rbf_grid.cv_results_)
    rbf_results.to_csv("svm_rbf_grid_search_results.csv", index=False)

    rbf_heatmap = (
        rbf_results.pivot(index="param_svm__C", columns="param_svm__gamma", values="mean_test_score")
        .reindex(index=SVM_C_VALUES, columns=RBF_GAMMA_VALUES)
        .astype(float)
    )
    plt.figure(figsize=(7, 5))
    img = plt.imshow(rbf_heatmap.values, cmap="viridis", aspect="auto")
    plt.xticks(np.arange(len(RBF_GAMMA_VALUES)), RBF_GAMMA_VALUES)
    plt.yticks(np.arange(len(SVM_C_VALUES)), SVM_C_VALUES)
    mid = (rbf_heatmap.values.min() + rbf_heatmap.values.max()) / 2
    for r in range(rbf_heatmap.shape[0]):
        for c in range(rbf_heatmap.shape[1]):
            v = rbf_heatmap.values[r, c]
            plt.text(c, r, f"{v:.3f}", ha="center", va="center", color="white" if v < mid else "black")

    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.title("RBF SVM: mean cross-validation F1")
    plt.colorbar(img, label="Mean CV F1")
    save_current_figure("svm_rbf_c_gamma_heatmap.png")

    # 2. Polynomial SVM sweep
    poly_grid = RandomizedSearchCV(
        estimator=Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="poly"))]),
        param_distributions={
            "svm__C": POLY_C_VALUES,
            "svm__degree": POLY_DEGREES,
            "svm__gamma": POLY_GAMMA_VALUES,
            "svm__coef0": POLY_COEF0_VALUES,
        },
        n_iter=24, cv=CV, scoring="f1", n_jobs=N_JOBS, pre_dispatch="2*n_jobs",
        refit=True, random_state=SEED, verbose=1,
    )
    
    poly_grid.fit(X_sw, y_sw)
    print("\nBest Poly SVM:", poly_grid.best_params_, f"CV F1: {poly_grid.best_score_:.4f}")

    poly_results = pd.DataFrame(poly_grid.cv_results_)
    poly_results.to_csv("svm_polynomial_grid_search_results.csv", index=False)

    poly_plot = poly_results.groupby(["param_svm__degree", "param_svm__C"], as_index=False)["mean_test_score"].max()

    plt.figure(figsize=(8, 5))
    for deg in POLY_DEGREES:
        sub = poly_plot[poly_plot["param_svm__degree"] == deg].copy()
        sub["C"] = sub["param_svm__C"].astype(float)
        sub = sub.sort_values("C")
        plt.plot(sub["C"], sub["mean_test_score"], "o-", label=f"Degree {deg}")

    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("Best mean cross-validation F1")
    plt.title("Polynomial SVM: effect of C and degree")
    plt.grid(alpha=0.25)
    plt.legend()
    save_current_figure("svm_polynomial_sweep.png")

    # 3. Kernel comparison & selection
    kernel_comp = pd.DataFrame([
        {"Kernel": "RBF", "Best CV F1": rbf_grid.best_score_, "Best parameters": str(rbf_grid.best_params_)},
        {"Kernel": "Polynomial", "Best CV F1": poly_grid.best_score_, "Best parameters": str(poly_grid.best_params_)},
    ])
    kernel_comp.to_csv("svm_kernel_comparison.csv", index=False)

    if rbf_grid.best_score_ >= poly_grid.best_score_:
        selected_kernel = "rbf"
        selected_params = remove_prefix(rbf_grid.best_params_, "svm__")
    else:
        selected_kernel = "poly"
        selected_params = remove_prefix(poly_grid.best_params_, "svm__")

    print("\nSelected SVM kernel:", selected_kernel, selected_params)

    # 4. CV stability on full training set
    evaluate_cv(
        make_model(SVC(kernel=selected_kernel, **selected_params), False, n_comp),
        X_train, y_train, f"Selected {selected_kernel.upper()} SVM (without PCA)",
    )
    evaluate_cv(
        make_model(SVC(kernel=selected_kernel, **selected_params), True, n_comp),
        X_train, y_train, f"Selected {selected_kernel.upper()} SVM (with PCA)",
    )

    # 5. PCA component sweep for SVM
    comp_values = list(range(1, n_features + 1))
    means, stds = [], []
    for k_comp in comp_values:
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=k_comp, random_state=SEED)),
            ("clf", SVC(kernel=selected_kernel, **selected_params)),
        ])
        sc = cross_val_score(pipe, X_sw, y_sw, cv=CV, scoring="f1", n_jobs=N_JOBS)
        means.append(sc.mean())
        stds.append(sc.std())

    pd.DataFrame({
        "PCA components": comp_values,
        "SVM mean CV F1": means,
        "SVM CV F1 std": stds,
    }).to_csv("svm_pca_component_sweep.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(comp_values, means, "o-", markersize=4, label=f"{selected_kernel.upper()} SVM")
    plt.axvline(n_comp, linestyle="--", color="gray",
                label=f"{PCA_VARIANCE_THRESHOLD * 100:.0f}% variance ({n_comp})")
    plt.xticks(np.arange(1, n_features + 1, 2))
    plt.xlabel("Number of PCA components")
    plt.ylabel("Mean cross-validation F1")
    plt.title(f"{selected_kernel.upper()} SVM: effect of PCA dimensionality")
    plt.grid(alpha=0.25)
    plt.legend()
    save_current_figure("svm_pca_component_sweep.png")

    # 6. Final test-set comparison: without PCA vs. with PCA
    configs = {
        "SVM - without PCA": make_model(SVC(kernel=selected_kernel, **selected_params), False, n_comp),
        "SVM - with PCA": make_model(SVC(kernel=selected_kernel, **selected_params), True, n_comp),
    }
    evaluate_pca_vs_non_pca(
        configs, X_train, X_test, y_train, y_test,
        n_comp, n_features,
        cm_filename="svm_confusion_matrices.png",
        summary_prefix="svm",
        extra_info={"SVM kernel": selected_kernel},
    )


if __name__ == "__main__":
    main()