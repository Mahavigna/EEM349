"""
Smart Grid Stability for Industry 4.0:
SVM vs k-NN, with and without PCA

Dataset:
https://www.kaggle.com/datasets/pcbreviglieri/smart-grid-stability

File used:
smart_grid_stability_augmented.csv

Place smart_grid_stability_augmented.csv next to this script.

The classification target is stabf:
    0 = stable
    1 = unstable

The continuous column stab is removed because stabf is derived from it.
Using stab as an input would cause target leakage.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


SEED = 42
TEST_SIZE = 0.20
SWEEP_SIZE = 3000
PCA_VARIANCE_THRESHOLD = 0.95
N_JOBS = -1

CV = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=SEED,
)

K_VALUES = list(range(1, 32, 2))
KNN_METRICS = ["euclidean", "manhattan"]
KNN_WEIGHTS = ["uniform", "distance"]

SVM_C_VALUES = [0.01, 0.1, 1, 10, 100]
RBF_GAMMA_VALUES = [0.001, 0.01, 0.1, 1]

POLY_C_VALUES = [0.1, 1, 10, 100]
POLY_DEGREES = [2, 3, 4]
POLY_GAMMA_VALUES = ["scale", 0.01, 0.1]
POLY_COEF0_VALUES = [0.0, 1.0]


def save_current_figure(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def make_model(classifier, use_pca, n_components):
    steps = [
        ("sc", StandardScaler()),
    ]

    if use_pca:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=n_components,
                    random_state=SEED,
                ),
            )
        )

    steps.append(("clf", classifier))

    return Pipeline(steps)


def remove_prefix(parameters, prefix):
    cleaned = {}

    for key, value in parameters.items():
        if key.startswith(prefix):
            cleaned[key[len(prefix):]] = value

    return cleaned


def evaluate_cv(model, X_data, y_data, name):
    scores = cross_val_score(
        model,
        X_data,
        y_data,
        cv=CV,
        scoring="f1",
        n_jobs=N_JOBS,
    )

    print(
        f"{name} CV F1: "
        f"{scores.mean():.4f} +/- {scores.std():.4f}"
    )

    return scores.mean(), scores.std()


# ---------------------------------------------------------------- 1. Load and validate

DATA_FILE = "smart_grid_stability_augmented.csv"

df = pd.read_csv(DATA_FILE)

required_columns = {
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "p1",
    "p2",
    "p3",
    "p4",
    "g1",
    "g2",
    "g3",
    "g4",
    "stab",
    "stabf",
}

missing_columns = sorted(required_columns.difference(df.columns))

if missing_columns:
    raise ValueError(
        "Dataset is missing required columns: "
        + ", ".join(missing_columns)
    )

feature_columns = [
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "p1",
    "p2",
    "p3",
    "p4",
    "g1",
    "g2",
    "g3",
    "g4",
]

if df.empty:
    raise ValueError("The dataset is empty.")

if df[feature_columns].isnull().any().any():
    raise ValueError("The dataset contains missing feature values.")

if df["stabf"].isnull().any():
    raise ValueError("The target column stabf contains missing values.")

unexpected_labels = sorted(
    set(df["stabf"].unique()).difference({"stable", "unstable"})
)

if unexpected_labels:
    raise ValueError(
        "Unexpected values in stabf: "
        + ", ".join(str(label) for label in unexpected_labels)
    )

print("Original dataset shape:", df.shape)
print("\nOriginal class counts:")
print(df["stabf"].value_counts())
print("\nOriginal class proportions:")
print(df["stabf"].value_counts(normalize=True).round(4))

total_exact_duplicates = int(df.duplicated().sum())
duplicate_feature_vectors = int(
    df.duplicated(subset=feature_columns, keep=False).sum()
)

conflicting_groups = (
    df.groupby(feature_columns, dropna=False)["stabf"]
    .nunique()
)

conflicting_feature_vectors = int(
    conflicting_groups.gt(1).sum()
)

print("\nDuplicate analysis:")
print("Total exact duplicate rows:", total_exact_duplicates)
print(
    "Rows belonging to duplicated feature vectors:",
    duplicate_feature_vectors,
)
print(
    "Feature vectors with conflicting labels:",
    conflicting_feature_vectors,
)

if conflicting_feature_vectors > 0:
    raise ValueError(
        "Identical feature vectors have conflicting stable and unstable "
        "labels. Resolve these conflicts before training."
    )

rows_before_duplicate_removal = len(df)

df = (
    df.drop_duplicates(
        subset=feature_columns + ["stabf"],
        keep="first",
    )
    .reset_index(drop=True)
)

rows_removed = rows_before_duplicate_removal - len(df)

print("Duplicate feature-label rows removed:", rows_removed)
print("Dataset shape after duplicate removal:", df.shape)

if len(df) < 10:
    raise ValueError(
        "Too few observations remain after duplicate removal."
    )

print("\nClass counts after duplicate removal:")
print(df["stabf"].value_counts())
print("\nClass proportions after duplicate removal:")
print(df["stabf"].value_counts(normalize=True).round(4))


# ---------------------------------------------------------------- 2. Features, target, and split

# stab is excluded because stabf is derived from stab.
X = df[feature_columns].copy()
y = (df["stabf"] == "unstable").astype(int)

if y.nunique() != 2:
    raise ValueError(
        "The target must contain both stable and unstable classes."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED,
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTraining class proportions:")
print(y_train.value_counts(normalize=True).sort_index().round(4))

print("\nTest class proportions:")
print(y_test.value_counts(normalize=True).sort_index().round(4))

actual_sweep_size = min(SWEEP_SIZE, len(X_train))

if actual_sweep_size < len(X_train):
    X_sw, _, y_sw, _ = train_test_split(
        X_train,
        y_train,
        train_size=actual_sweep_size,
        stratify=y_train,
        random_state=SEED,
    )
else:
    X_sw = X_train.copy()
    y_sw = y_train.copy()

print("\nSweep dataset shape:", X_sw.shape)
print("Sweep observations:", len(X_sw))


# ---------------------------------------------------------------- 3. PCA explained variance

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca_full = PCA(random_state=SEED)
pca_full.fit(X_train_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

threshold_positions = np.where(
    cumulative_variance >= PCA_VARIANCE_THRESHOLD
)[0]

if len(threshold_positions) == 0:
    raise RuntimeError(
        "PCA did not reach the requested explained-variance threshold."
    )

N_COMP = int(threshold_positions[0] + 1)
N_FEATURES = X.shape[1]

print(
    "\nExplained variance per principal component:",
    np.round(explained_variance, 4),
)

print(
    "Cumulative explained variance:",
    np.round(cumulative_variance, 4),
)

print(
    f"Components needed for "
    f"{PCA_VARIANCE_THRESHOLD * 100:.0f}% variance: "
    f"{N_COMP} of {N_FEATURES}"
)

pca_variance_table = pd.DataFrame(
    {
        "Principal component": np.arange(1, N_FEATURES + 1),
        "Explained variance ratio": explained_variance,
        "Cumulative explained variance": cumulative_variance,
    }
)

pca_variance_table.to_csv(
    "pca_explained_variance.csv",
    index=False,
)

plt.figure(figsize=(7, 4.5))

plt.bar(
    np.arange(1, N_FEATURES + 1),
    explained_variance,
    label="Per component",
)

plt.plot(
    np.arange(1, N_FEATURES + 1),
    cumulative_variance,
    "r-o",
    label="Cumulative",
)

plt.axhline(
    PCA_VARIANCE_THRESHOLD,
    linestyle="--",
    color="gray",
    label=f"{PCA_VARIANCE_THRESHOLD * 100:.0f}% threshold",
)

plt.axvline(
    N_COMP,
    linestyle=":",
    color="black",
    label=f"Selected components = {N_COMP}",
)

plt.xticks(np.arange(1, N_FEATURES + 1))
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.title("PCA explained variance")
plt.legend()

save_current_figure("pca_variance.png")


# ---------------------------------------------------------------- 4. k-NN parameter sweep

knn_pipeline = Pipeline(
    [
        ("sc", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ]
)

knn_parameter_grid = {
    "knn__n_neighbors": K_VALUES,
    "knn__metric": KNN_METRICS,
    "knn__weights": KNN_WEIGHTS,
}

knn_grid = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_parameter_grid,
    cv=CV,
    scoring="f1",
    n_jobs=N_JOBS,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    refit=True,
    verbose=1,
)

knn_grid.fit(X_sw, y_sw)

print("\nBest k-NN parameters:")
print(knn_grid.best_params_)
print(f"Best k-NN CV F1: {knn_grid.best_score_:.4f}")

knn_results = pd.DataFrame(knn_grid.cv_results_)
knn_results.to_csv("knn_grid_search_results.csv", index=False)

plt.figure(figsize=(8, 5))

knn_line_styles = {
    ("euclidean", "uniform"): "-",
    ("euclidean", "distance"): "--",
    ("manhattan", "uniform"): "-.",
    ("manhattan", "distance"): ":",
}

knn_markers = {
    ("euclidean", "uniform"): "o",
    ("euclidean", "distance"): "s",
    ("manhattan", "uniform"): "^",
    ("manhattan", "distance"): "D",
}

for metric in KNN_METRICS:
    for weights in KNN_WEIGHTS:
        subset = knn_results[
            (knn_results["param_knn__metric"] == metric)
            & (knn_results["param_knn__weights"] == weights)
        ].copy()

        subset["k"] = subset[
            "param_knn__n_neighbors"
        ].astype(int)

        subset = subset.sort_values("k")

        plt.plot(
            subset["k"],
            subset["mean_test_score"],
            marker=knn_markers[(metric, weights)],
            linestyle=knn_line_styles[(metric, weights)],
            label=f"{metric}, {weights}",
        )

plt.xlabel("Number of neighbours, k")
plt.ylabel("Mean cross-validation F1")
plt.title("k-NN parameter sweep")
plt.xticks(K_VALUES, rotation=90)
plt.grid(alpha=0.25)
plt.legend()

save_current_figure("knn_k_sweep.png")


# ---------------------------------------------------------------- 5. RBF SVM parameter sweep

rbf_pipeline = Pipeline(
    [
        ("sc", StandardScaler()),
        ("svm", SVC(kernel="rbf")),
    ]
)

rbf_parameter_grid = {
    "svm__C": SVM_C_VALUES,
    "svm__gamma": RBF_GAMMA_VALUES,
}

rbf_grid = GridSearchCV(
    estimator=rbf_pipeline,
    param_grid=rbf_parameter_grid,
    cv=CV,
    scoring="f1",
    n_jobs=N_JOBS,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    refit=True,
    verbose=1,
)

rbf_grid.fit(X_sw, y_sw)

print("\nBest RBF SVM parameters:")
print(rbf_grid.best_params_)
print(f"Best RBF SVM CV F1: {rbf_grid.best_score_:.4f}")

rbf_results = pd.DataFrame(rbf_grid.cv_results_)
rbf_results.to_csv("svm_rbf_grid_search_results.csv", index=False)

rbf_heatmap = (
    rbf_results.pivot(
        index="param_svm__C",
        columns="param_svm__gamma",
        values="mean_test_score",
    )
    .reindex(
        index=SVM_C_VALUES,
        columns=RBF_GAMMA_VALUES,
    )
    .astype(float)
)

plt.figure(figsize=(7, 5))

image = plt.imshow(
    rbf_heatmap.values,
    cmap="viridis",
    aspect="auto",
    vmin=rbf_heatmap.values.min(),
    vmax=rbf_heatmap.values.max(),
)

plt.xticks(
    np.arange(len(RBF_GAMMA_VALUES)),
    RBF_GAMMA_VALUES,
)

plt.yticks(
    np.arange(len(SVM_C_VALUES)),
    SVM_C_VALUES,
)

for row_index in range(rbf_heatmap.shape[0]):
    for column_index in range(rbf_heatmap.shape[1]):
        value = rbf_heatmap.values[
            row_index,
            column_index,
        ]

        text_color = (
            "white"
            if value
            < (
                rbf_heatmap.values.min()
                + rbf_heatmap.values.max()
            )
            / 2
            else "black"
        )

        plt.text(
            column_index,
            row_index,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=text_color,
        )

plt.xlabel("Gamma")
plt.ylabel("C")
plt.title("RBF SVM: mean cross-validation F1")
plt.colorbar(image, label="Mean CV F1")

save_current_figure("svm_rbf_c_gamma_heatmap.png")


# ---------------------------------------------------------------- 6. Polynomial SVM parameter sweep

poly_pipeline = Pipeline(
    [
        ("sc", StandardScaler()),
        ("svm", SVC(kernel="poly")),
    ]
)

poly_parameter_grid = {
    "svm__C": POLY_C_VALUES,
    "svm__degree": POLY_DEGREES,
    "svm__gamma": POLY_GAMMA_VALUES,
    "svm__coef0": POLY_COEF0_VALUES,
}

poly_grid = RandomizedSearchCV(
    estimator=poly_pipeline,
    param_distributions=poly_parameter_grid,
    n_iter=24,
    cv=CV,
    scoring="f1",
    n_jobs=N_JOBS,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    refit=True,
    random_state=SEED,
    verbose=1,
)

poly_grid.fit(X_sw, y_sw)

print("\nBest polynomial SVM parameters:")
print(poly_grid.best_params_)
print(
    f"Best polynomial SVM CV F1: "
    f"{poly_grid.best_score_:.4f}"
)

poly_results = pd.DataFrame(poly_grid.cv_results_)
poly_results.to_csv(
    "svm_polynomial_grid_search_results.csv",
    index=False,
)

poly_plot_data = (
    poly_results.groupby(
        [
            "param_svm__degree",
            "param_svm__C",
        ],
        as_index=False,
    )["mean_test_score"]
    .max()
)

plt.figure(figsize=(8, 5))

for degree in POLY_DEGREES:
    subset = poly_plot_data[
        poly_plot_data["param_svm__degree"] == degree
    ].copy()

    subset["C"] = subset["param_svm__C"].astype(float)
    subset = subset.sort_values("C")

    plt.plot(
        subset["C"],
        subset["mean_test_score"],
        "o-",
        label=f"Degree {degree}",
    )

plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Best mean cross-validation F1")
plt.title(
    "Polynomial SVM: effect of C and polynomial degree"
)
plt.grid(alpha=0.25)
plt.legend()

save_current_figure("svm_polynomial_sweep.png")


# ---------------------------------------------------------------- 7. RBF versus polynomial SVM comparison

kernel_comparison = pd.DataFrame(
    [
        {
            "Kernel": "RBF",
            "Best CV F1": rbf_grid.best_score_,
            "Best parameters": str(rbf_grid.best_params_),
        },
        {
            "Kernel": "Polynomial",
            "Best CV F1": poly_grid.best_score_,
            "Best parameters": str(poly_grid.best_params_),
        },
    ]
)

kernel_comparison.to_csv(
    "svm_kernel_comparison.csv",
    index=False,
)

print("\nSVM kernel comparison:")
print(kernel_comparison.to_string(index=False))

plt.figure(figsize=(6, 4.5))

bars = plt.bar(
    kernel_comparison["Kernel"],
    kernel_comparison["Best CV F1"],
    color=["tab:blue", "tab:orange"],
)

lower_limit = max(
    0.0,
    kernel_comparison["Best CV F1"].min() - 0.05,
)

upper_limit = min(
    1.0,
    kernel_comparison["Best CV F1"].max() + 0.02,
)

if upper_limit <= lower_limit:
    lower_limit = 0.0
    upper_limit = 1.0

plt.ylim(lower_limit, upper_limit)
plt.ylabel("Best mean cross-validation F1")
plt.title("SVM kernel comparison")

for bar, score in zip(
    bars,
    kernel_comparison["Best CV F1"],
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score,
        f"{score:.4f}",
        ha="center",
        va="bottom",
    )

save_current_figure("svm_kernel_comparison.png")


# ---------------------------------------------------------------- 8. Select the SVM kernel from CV only

if rbf_grid.best_score_ >= poly_grid.best_score_:
    selected_svm_kernel = "rbf"
    selected_svm_parameters = remove_prefix(
        rbf_grid.best_params_,
        "svm__",
    )
else:
    selected_svm_kernel = "poly"
    selected_svm_parameters = remove_prefix(
        poly_grid.best_params_,
        "svm__",
    )

best_knn_parameters = remove_prefix(
    knn_grid.best_params_,
    "knn__",
)

print("\nSelected SVM kernel:", selected_svm_kernel)
print(
    "Selected SVM parameters:",
    selected_svm_parameters,
)

print(
    "Selected k-NN parameters:",
    best_knn_parameters,
)


# ---------------------------------------------------------------- 9. Cross-validation stability of selected models

selected_svm_cv_model = Pipeline(
    [
        ("sc", StandardScaler()),
        (
            "svm",
            SVC(
                kernel=selected_svm_kernel,
                **selected_svm_parameters,
            ),
        ),
    ]
)

selected_knn_cv_model = Pipeline(
    [
        ("sc", StandardScaler()),
        (
            "knn",
            KNeighborsClassifier(
                **best_knn_parameters,
            ),
        ),
    ]
)

svm_cv_mean, svm_cv_std = evaluate_cv(
    selected_svm_cv_model,
    X_train,
    y_train,
    f"Selected {selected_svm_kernel.upper()} SVM",
)

knn_cv_mean, knn_cv_std = evaluate_cv(
    selected_knn_cv_model,
    X_train,
    y_train,
    "Selected k-NN",
)

selected_model_cv_summary = pd.DataFrame(
    [
        {
            "Model": f"{selected_svm_kernel.upper()} SVM",
            "Mean CV F1": svm_cv_mean,
            "CV F1 standard deviation": svm_cv_std,
        },
        {
            "Model": "k-NN",
            "Mean CV F1": knn_cv_mean,
            "CV F1 standard deviation": knn_cv_std,
        },
    ]
)

selected_model_cv_summary.to_csv(
    "selected_model_cv_summary.csv",
    index=False,
)


# ---------------------------------------------------------------- 10. PCA component sweep

pca_component_values = list(
    range(1, N_FEATURES + 1)
)

pca_scores = {
    "SVM": [],
    "k-NN": [],
}

pca_score_stds = {
    "SVM": [],
    "k-NN": [],
}

for n_components in pca_component_values:
    svm_pca_pipeline = Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=n_components,
                    random_state=SEED,
                ),
            ),
            (
                "clf",
                SVC(
                    kernel=selected_svm_kernel,
                    **selected_svm_parameters,
                ),
            ),
        ]
    )

    knn_pca_pipeline = Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=n_components,
                    random_state=SEED,
                ),
            ),
            (
                "clf",
                KNeighborsClassifier(
                    **best_knn_parameters,
                ),
            ),
        ]
    )

    svm_scores = cross_val_score(
        svm_pca_pipeline,
        X_sw,
        y_sw,
        cv=CV,
        scoring="f1",
        n_jobs=N_JOBS,
    )

    knn_scores = cross_val_score(
        knn_pca_pipeline,
        X_sw,
        y_sw,
        cv=CV,
        scoring="f1",
        n_jobs=N_JOBS,
    )

    pca_scores["SVM"].append(svm_scores.mean())
    pca_score_stds["SVM"].append(svm_scores.std())

    pca_scores["k-NN"].append(knn_scores.mean())
    pca_score_stds["k-NN"].append(knn_scores.std())

pca_sweep_table = pd.DataFrame(
    {
        "PCA components": pca_component_values,
        "SVM mean CV F1": pca_scores["SVM"],
        "SVM CV F1 standard deviation": pca_score_stds[
            "SVM"
        ],
        "k-NN mean CV F1": pca_scores["k-NN"],
        "k-NN CV F1 standard deviation": pca_score_stds[
            "k-NN"
        ],
    }
)

pca_sweep_table.to_csv(
    "pca_component_sweep_results.csv",
    index=False,
)

plt.figure(figsize=(8, 5))

plt.plot(
    pca_component_values,
    pca_scores["SVM"],
    "o-",
    label=f"{selected_svm_kernel.upper()} SVM",
)

plt.plot(
    pca_component_values,
    pca_scores["k-NN"],
    "s-",
    label="k-NN",
)

plt.axvline(
    N_COMP,
    linestyle="--",
    color="gray",
    label=(
        f"{PCA_VARIANCE_THRESHOLD * 100:.0f}% "
        f"variance threshold ({N_COMP})"
    ),
)

plt.xticks(pca_component_values)
plt.xlabel("Number of PCA components")
plt.ylabel("Mean cross-validation F1")
plt.title("Effect of PCA dimensionality on classification")
plt.grid(alpha=0.25)
plt.legend()

save_current_figure("pca_component_sweep.png")


# ---------------------------------------------------------------- 11. Final four configurations

configurations = {
    "SVM - without PCA": make_model(
        SVC(
            kernel=selected_svm_kernel,
            **selected_svm_parameters,
        ),
        False,
        N_COMP,
    ),
    "SVM - with PCA": make_model(
        SVC(
            kernel=selected_svm_kernel,
            **selected_svm_parameters,
        ),
        True,
        N_COMP,
    ),
    "k-NN - without PCA": make_model(
        KNeighborsClassifier(
            **best_knn_parameters,
        ),
        False,
        N_COMP,
    ),
    "k-NN - with PCA": make_model(
        KNeighborsClassifier(
            **best_knn_parameters,
        ),
        True,
        N_COMP,
    ),
}

summary_rows = []

fig, axes = plt.subplots(
    1,
    4,
    figsize=(18, 4.5),
)

for axis, configuration in zip(
    axes,
    configurations.items(),
):
    configuration_name, model = configuration

    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = (
        time.perf_counter() - training_start
    )

    prediction_start = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_time = (
        time.perf_counter() - prediction_start
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    summary_rows.append(
        {
            "Configuration": configuration_name,
            "SVM kernel": (
                selected_svm_kernel
                if configuration_name.startswith("SVM")
                else "Not applicable"
            ),
            "No. of features": (
                N_COMP
                if "with PCA" in configuration_name
                else N_FEATURES
            ),
            "Accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "Precision": precision_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            ),
            "F1": f1_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            ),
            "True stable": int(matrix[0, 0]),
            "False unstable": int(matrix[0, 1]),
            "False stable": int(matrix[1, 0]),
            "True unstable": int(matrix[1, 1]),
            "Train time (s)": training_time,
            "Predict time (s)": prediction_time,
        }
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["stable", "unstable"],
    )

    display.plot(
        ax=axis,
        colorbar=False,
        cmap="Blues",
        values_format="d",
    )

    axis.set_title(configuration_name)

save_current_figure("confusion_matrices.png")


# ---------------------------------------------------------------- 12. Save and print final results

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    "summary_table_full_precision.csv",
    index=False,
)

rounded_summary = summary.copy()

numeric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Train time (s)",
    "Predict time (s)",
]

rounded_summary[numeric_columns] = rounded_summary[
    numeric_columns
].round(4)

rounded_summary.to_csv(
    "summary_table.csv",
    index=False,
)

print("\nFinal test-set results:")
print(rounded_summary.to_string(index=False))

best_configuration_index = summary["F1"].idxmax()
best_configuration = summary.loc[
    best_configuration_index
]

print("\nBest final configuration by test F1:")
print(
    best_configuration[
        [
            "Configuration",
            "SVM kernel",
            "No. of features",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Train time (s)",
            "Predict time (s)",
        ]
    ].to_string()
)

print("\nGenerated files:")
print("pca_explained_variance.csv")
print("pca_variance.png")
print("knn_grid_search_results.csv")
print("knn_k_sweep.png")
print("svm_rbf_grid_search_results.csv")
print("svm_rbf_c_gamma_heatmap.png")
print("svm_polynomial_grid_search_results.csv")
print("svm_polynomial_sweep.png")
print("svm_kernel_comparison.csv")
print("svm_kernel_comparison.png")
print("selected_model_cv_summary.csv")
print("pca_component_sweep_results.csv")
print("pca_component_sweep.png")
print("confusion_matrices.png")
print("summary_table.csv")
print("summary_table_full_precision.csv")