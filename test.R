# ==============================================================================
# Smart Grid Stability for Industry 4.0:
# SVM vs k-NN, with and without PCA
#
# Dataset:
# https://www.kaggle.com/datasets/pcbreviglieri/smart-grid-stability
#
# File used:
# smart_grid_stability_augmented.csv
#
# Place smart_grid_stability_augmented.csv next to this script.
#
# The classification target is stabf:
#     0 = stable
#     1 = unstable
#
# The continuous column stab is removed because stabf is derived from it.
# Using stab as an input would cause target leakage.
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(e1071)
  library(kknn)
  library(future)
  library(future.apply)
})

# ---------------------------------------------------------------- Configuration

SEED <- 42L
TEST_SIZE <- 0.20
SWEEP_SIZE <- 3000L
PCA_VARIANCE_THRESHOLD <- 0.95
N_SPLITS <- 3L
N_WORKERS <- max(1L, parallel::detectCores(logical = TRUE) - 1L)

K_VALUES <- seq(1L, 31L, by = 2L)
KNN_METRICS <- c("euclidean", "manhattan")
KNN_WEIGHTS <- c("uniform", "distance")

SVM_C_VALUES <- c(0.01, 0.1, 1, 10, 100)
RBF_GAMMA_VALUES <- c(0.001, 0.01, 0.1, 1)

POLY_C_VALUES <- c(0.1, 1, 10, 100)
POLY_DEGREES <- c(2L, 3L, 4L)
POLY_GAMMA_VALUES <- c("scale", "0.01", "0.1")
POLY_COEF0_VALUES <- c(0.0, 1.0)

plan(multisession, workers = N_WORKERS)
on.exit(plan(sequential), add = TRUE)

# ---------------------------------------------------------------- Helper functions

save_plot <- function(plot_obj, filename, width, height, dpi = 150) {
  ggsave(
    filename = filename,
    plot = plot_obj,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white"
  )
}

pop_sd <- function(x) {
  sqrt(mean((x - mean(x))^2))
}

stratified_split <- function(y, test_size = NULL, train_size = NULL, seed = SEED) {
  set.seed(seed)
  n_total <- length(y)

  if (!is.null(train_size)) {
    n_train_target <- if (train_size < 1) round(n_total * train_size) else as.integer(train_size)
    prop_train <- n_train_target / n_total
  } else {
    prop_train <- 1 - test_size
  }

  classes <- sort(unique(y))
  train_idx <- integer(0)
  test_idx <- integer(0)

  for (cls in classes) {
    cls_idx <- sample(which(y == cls))
    n_cls_train <- max(1L, min(length(cls_idx) - 1L, round(length(cls_idx) * prop_train)))
    train_idx <- c(train_idx, cls_idx[seq_len(n_cls_train)])
    test_idx <- c(test_idx, cls_idx[-seq_len(n_cls_train)])
  }

  list(train = sort(train_idx), test = sort(test_idx))
}

create_stratified_folds <- function(y, k = N_SPLITS, seed = SEED) {
  set.seed(seed)
  fold_ids <- integer(length(y))

  for (cls in sort(unique(y))) {
    cls_idx <- sample(which(y == cls))
    fold_ids[cls_idx] <- rep(seq_len(k), length.out = length(cls_idx))
  }

  fold_ids
}

fit_preprocessor <- function(X_df, use_pca = FALSE, n_components = NULL) {
  X_mat <- as.matrix(X_df)
  storage.mode(X_mat) <- "double"

  centers <- colMeans(X_mat)
  scales <- apply(X_mat, 2, pop_sd)
  scales[scales == 0] <- 1.0

  X_scaled <- scale(X_mat, center = centers, scale = scales)
  attr(X_scaled, "scaled:center") <- NULL
  attr(X_scaled, "scaled:scale") <- NULL

  rotation <- NULL
  if (use_pca) {
    pca_fit <- prcomp(X_scaled, center = FALSE, scale. = FALSE)
    rotation <- pca_fit$rotation[, seq_len(n_components), drop = FALSE]
  }

  list(
    center = centers,
    scale = scales,
    use_pca = use_pca,
    n_components = n_components,
    rotation = rotation
  )
}

transform_preprocessor <- function(prep, X_df) {
  X_mat <- as.matrix(X_df)
  storage.mode(X_mat) <- "double"

  X_scaled <- scale(X_mat, center = prep$center, scale = prep$scale)
  attr(X_scaled, "scaled:center") <- NULL
  attr(X_scaled, "scaled:scale") <- NULL

  if (prep$use_pca) {
    X_out <- X_scaled %*% prep$rotation
    colnames(X_out) <- paste0("PC", seq_len(ncol(X_out)))
    return(X_out)
  }

  X_scaled
}

resolve_gamma <- function(gamma_val, X_mat) {
  if (identical(as.character(gamma_val), "scale")) {
    flat_var <- mean((X_mat - mean(X_mat))^2)
    if (flat_var == 0) {
      return(1 / ncol(X_mat))
    }
    return(1 / (ncol(X_mat) * flat_var))
  }
  as.numeric(gamma_val)
}

compute_metrics <- function(y_true, y_pred) {
  tn <- sum(y_true == 0L & y_pred == 0L)
  fp <- sum(y_true == 0L & y_pred == 1L)
  fn <- sum(y_true == 1L & y_pred == 0L)
  tp <- sum(y_true == 1L & y_pred == 1L)

  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  precision <- if ((tp + fp) == 0L) 0.0 else tp / (tp + fp)
  recall <- if ((tp + fn) == 0L) 0.0 else tp / (tp + fn)
  f1 <- if ((precision + recall) == 0.0) 0.0 else 2 * precision * recall / (precision + recall)

  list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1,
    tn = as.integer(tn),
    fp = as.integer(fp),
    fn = as.integer(fn),
    tp = as.integer(tp)
  )
}

fit_pipeline <- function(X_train_df, y_train, model_type, params,
                         use_pca = FALSE, n_components = NULL) {
  prep <- fit_preprocessor(X_train_df, use_pca = use_pca, n_components = n_components)
  X_tr <- transform_preprocessor(prep, X_train_df)
  y_fac <- factor(y_train, levels = c(0L, 1L))

  if (model_type == "svm") {
    e1071_kernel <- if (params$kernel == "rbf") "radial" else "polynomial"
    gamma_num <- resolve_gamma(params$gamma, X_tr)

    if (e1071_kernel == "radial") {
      clf <- e1071::svm(
        x = X_tr,
        y = y_fac,
        type = "C-classification",
        kernel = "radial",
        cost = as.numeric(params$C),
        gamma = gamma_num,
        scale = FALSE
      )
    } else {
      clf <- e1071::svm(
        x = X_tr,
        y = y_fac,
        type = "C-classification",
        kernel = "polynomial",
        cost = as.numeric(params$C),
        degree = as.integer(params$degree),
        gamma = gamma_num,
        coef0 = as.numeric(params$coef0),
        scale = FALSE
      )
    }

    return(list(model_type = "svm", prep = prep, clf = clf))
  }

  if (model_type == "knn") {
    train_df <- as.data.frame(X_tr)
    train_df$y <- y_fac
    return(list(model_type = "knn", prep = prep, train_df = train_df, params = params))
  }

  stop("Unknown model_type: ", model_type)
}

predict_pipeline <- function(model_obj, X_new_df) {
  X_te <- transform_preprocessor(model_obj$prep, X_new_df)

  if (model_obj$model_type == "svm") {
    preds <- predict(model_obj$clf, X_te)
    return(as.integer(as.character(preds)))
  }

  if (model_obj$model_type == "knn") {
    test_df <- as.data.frame(X_te)
    dist_p <- if (model_obj$params$metric == "euclidean") 2 else 1
    wt_kernel <- if (model_obj$params$weights == "uniform") "rectangular" else "inv"

    knn_fit <- kknn::kknn(
      formula = y ~ .,
      train = model_obj$train_df,
      test = test_df,
      k = as.integer(model_obj$params$k),
      distance = dist_p,
      kernel = wt_kernel,
      scale = FALSE
    )

    preds <- fitted(knn_fit)
    return(as.integer(as.character(preds)))
  }

  stop("Unknown model_type: ", model_obj$model_type)
}

cv_f1_scores <- function(X_df, y_vec, fold_ids, model_type, params,
                         use_pca = FALSE, n_components = NULL) {
  k_folds <- max(fold_ids)
  scores <- numeric(k_folds)

  for (f in seq_len(k_folds)) {
    val_idx <- which(fold_ids == f)
    tr_idx <- which(fold_ids != f)

    fitted_mod <- fit_pipeline(
      X_df[tr_idx, , drop = FALSE],
      y_vec[tr_idx],
      model_type = model_type,
      params = params,
      use_pca = use_pca,
      n_components = n_components
    )

    preds <- predict_pipeline(fitted_mod, X_df[val_idx, , drop = FALSE])
    scores[f] <- compute_metrics(y_vec[val_idx], preds)$f1
  }

  scores
}

evaluate_cv <- function(X_df, y_vec, fold_ids, model_type, params, name,
                        use_pca = FALSE, n_components = NULL) {
  scores <- cv_f1_scores(
    X_df, y_vec, fold_ids,
    model_type = model_type,
    params = params,
    use_pca = use_pca,
    n_components = n_components
  )

  mean_score <- mean(scores)
  std_score <- pop_sd(scores)

  cat(sprintf("%s CV F1: %.4f +/- %.4f\n", name, mean_score, std_score))
  list(mean = mean_score, std = std_score)
}

format_params <- function(param_list) {
  paste(
    vapply(
      names(param_list),
      function(k) paste0(k, "=", param_list[[k]]),
      character(1)
    ),
    collapse = ", "
  )
}

# ---------------------------------------------------------------- 1. Load and validate

DATA_FILE <- "smart_grid_stability_augmented.csv"
df <- as.data.frame(readr::read_csv(DATA_FILE, show_col_types = FALSE))

required_columns <- c(
  "tau1", "tau2", "tau3", "tau4",
  "p1", "p2", "p3", "p4",
  "g1", "g2", "g3", "g4",
  "stab", "stabf"
)

missing_columns <- sort(setdiff(required_columns, colnames(df)))
if (length(missing_columns) > 0L) {
  stop("Dataset is missing required columns: ", paste(missing_columns, collapse = ", "))
}

feature_columns <- c(
  "tau1", "tau2", "tau3", "tau4",
  "p1", "p2", "p3", "p4",
  "g1", "g2", "g3", "g4"
)

if (nrow(df) == 0L) {
  stop("The dataset is empty.")
}

if (anyNA(df[, feature_columns, drop = FALSE])) {
  stop("The dataset contains missing feature values.")
}

if (anyNA(df$stabf)) {
  stop("The target column stabf contains missing values.")
}

unexpected_labels <- sort(setdiff(unique(df$stabf), c("stable", "unstable")))
if (length(unexpected_labels) > 0L) {
  stop("Unexpected values in stabf: ", paste(unexpected_labels, collapse = ", "))
}

cat(sprintf("Original dataset shape: (%d, %d)\n", nrow(df), ncol(df)))
cat("\nOriginal class counts:\n")
print(table(df$stabf))
cat("\nOriginal class proportions:\n")
print(round(prop.table(table(df$stabf)), 4))

total_exact_duplicates <- sum(duplicated(df))
duplicate_feature_vectors <- sum(
  duplicated(df[, feature_columns, drop = FALSE]) |
    duplicated(df[, feature_columns, drop = FALSE], fromLast = TRUE)
)

conflicting_feature_vectors <- df |>
  group_by(across(all_of(feature_columns))) |>
  summarise(n_labels = n_distinct(stabf), .groups = "drop") |>
  summarise(n_conflicts = sum(n_labels > 1L)) |>
  pull(n_conflicts)

cat("\nDuplicate analysis:\n")
cat("Total exact duplicate rows:", total_exact_duplicates, "\n")
cat("Rows belonging to duplicated feature vectors:", duplicate_feature_vectors, "\n")
cat("Feature vectors with conflicting labels:", conflicting_feature_vectors, "\n")

if (conflicting_feature_vectors > 0L) {
  stop(
    "Identical feature vectors have conflicting stable and unstable ",
    "labels. Resolve these conflicts before training."
  )
}

rows_before_duplicate_removal <- nrow(df)

df <- df |>
  distinct(across(all_of(c(feature_columns, "stabf"))), .keep_all = TRUE) |>
  as.data.frame()

rows_removed <- rows_before_duplicate_removal - nrow(df)

cat("Duplicate feature-label rows removed:", rows_removed, "\n")
cat(sprintf("Dataset shape after duplicate removal: (%d, %d)\n", nrow(df), ncol(df)))

if (nrow(df) < 10L) {
  stop("Too few observations remain after duplicate removal.")
}

cat("\nClass counts after duplicate removal:\n")
print(table(df$stabf))
cat("\nClass proportions after duplicate removal:\n")
print(round(prop.table(table(df$stabf)), 4))

# ---------------------------------------------------------------- 2. Features, target, and split

# stab is excluded because stabf is derived from stab.
X <- df[, feature_columns, drop = FALSE]
y <- as.integer(df$stabf == "unstable")

if (length(unique(y)) != 2L) {
  stop("The target must contain both stable and unstable classes.")
}

split_idx <- stratified_split(y, test_size = TEST_SIZE, seed = SEED)
X_train <- X[split_idx$train, , drop = FALSE]
X_test <- X[split_idx$test, , drop = FALSE]
y_train <- y[split_idx$train]
y_test <- y[split_idx$test]

cat(sprintf("\nTraining set shape: (%d, %d)\n", nrow(X_train), ncol(X_train)))
cat(sprintf("Test set shape: (%d, %d)\n", nrow(X_test), ncol(X_test)))

cat("\nTraining class proportions:\n")
print(round(prop.table(table(y_train)), 4))

cat("\nTest class proportions:\n")
print(round(prop.table(table(y_test)), 4))

actual_sweep_size <- min(SWEEP_SIZE, nrow(X_train))

if (actual_sweep_size < nrow(X_train)) {
  sw_idx <- stratified_split(y_train, train_size = actual_sweep_size, seed = SEED)
  X_sw <- X_train[sw_idx$train, , drop = FALSE]
  y_sw <- y_train[sw_idx$train]
} else {
  X_sw <- X_train
  y_sw <- y_train
}

cat(sprintf("\nSweep dataset shape: (%d, %d)\n", nrow(X_sw), ncol(X_sw)))
cat("Sweep observations:", nrow(X_sw), "\n")

cv_folds_sw <- create_stratified_folds(y_sw, k = N_SPLITS, seed = SEED)
cv_folds_train <- create_stratified_folds(y_train, k = N_SPLITS, seed = SEED)

# ---------------------------------------------------------------- 3. PCA explained variance

prep_train_only <- fit_preprocessor(X_train, use_pca = FALSE)
X_train_scaled <- transform_preprocessor(prep_train_only, X_train)

pca_full <- prcomp(X_train_scaled, center = FALSE, scale. = FALSE)
explained_variance <- (pca_full$sdev^2) / sum(pca_full$sdev^2)
cumulative_variance <- cumsum(explained_variance)

threshold_positions <- which(cumulative_variance >= PCA_VARIANCE_THRESHOLD)
if (length(threshold_positions) == 0L) {
  stop("PCA did not reach the requested explained-variance threshold.")
}

N_COMP <- as.integer(threshold_positions[1])
N_FEATURES <- ncol(X)

cat("\nExplained variance per principal component:\n")
print(round(explained_variance, 4))
cat("Cumulative explained variance:\n")
print(round(cumulative_variance, 4))
cat(sprintf(
  "Components needed for %.0f%% variance: %d of %d\n",
  PCA_VARIANCE_THRESHOLD * 100, N_COMP, N_FEATURES
))

pca_variance_table <- data.frame(
  `Principal component` = seq_len(N_FEATURES),
  `Explained variance ratio` = explained_variance,
  `Cumulative explained variance` = cumulative_variance,
  check.names = FALSE
)

write.csv(pca_variance_table, "pca_explained_variance.csv", row.names = FALSE)

p_pca_var <- ggplot(pca_variance_table, aes(x = `Principal component`)) +
  geom_col(aes(y = `Explained variance ratio`, fill = "Per component"), width = 0.7) +
  geom_line(aes(y = `Cumulative explained variance`, color = "Cumulative"), linewidth = 0.9) +
  geom_point(aes(y = `Cumulative explained variance`, color = "Cumulative"), size = 2.2) +
  geom_hline(
    aes(yintercept = PCA_VARIANCE_THRESHOLD, linetype = sprintf("%.0f%% threshold", PCA_VARIANCE_THRESHOLD * 100)),
    color = "gray40"
  ) +
  geom_vline(
    aes(xintercept = N_COMP, linetype = sprintf("Selected components = %d", N_COMP)),
    color = "black"
  ) +
  scale_x_continuous(breaks = seq_len(N_FEATURES)) +
  scale_fill_manual(name = NULL, values = c("Per component" = "#1f77b4")) +
  scale_color_manual(name = NULL, values = c("Cumulative" = "#d62728")) +
  scale_linetype_manual(
    name = NULL,
    values = setNames(
      c("dashed", "dotted"),
      c(
        sprintf("%.0f%% threshold", PCA_VARIANCE_THRESHOLD * 100),
        sprintf("Selected components = %d", N_COMP)
      )
    )
  ) +
  labs(
    title = "PCA explained variance",
    x = "Principal component",
    y = "Explained variance ratio"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "right")

save_plot(p_pca_var, "pca_variance.png", width = 7, height = 4.5)

# ---------------------------------------------------------------- 4. k-NN parameter sweep

cat("\nRunning k-NN grid search...\n")

knn_grid_df <- expand.grid(
  param_knn__n_neighbors = K_VALUES,
  param_knn__metric = KNN_METRICS,
  param_knn__weights = KNN_WEIGHTS,
  stringsAsFactors = FALSE
)

knn_cv_list <- future_lapply(
  seq_len(nrow(knn_grid_df)),
  function(i) {
    row <- knn_grid_df[i, ]
    scores <- cv_f1_scores(
      X_sw, y_sw, cv_folds_sw,
      model_type = "knn",
      params = list(
        k = row$param_knn__n_neighbors,
        metric = row$param_knn__metric,
        weights = row$param_knn__weights
      )
    )
    c(mean_test_score = mean(scores), std_test_score = pop_sd(scores))
  },
  future.seed = TRUE,
  future.packages = c("kknn")
)

knn_results <- cbind(knn_grid_df, as.data.frame(do.call(rbind, knn_cv_list)))
knn_results$rank_test_score <- rank(-knn_results$mean_test_score, ties.method = "min")

best_knn_idx <- which.max(knn_results$mean_test_score)
best_knn_row <- knn_results[best_knn_idx, ]

best_knn_parameters <- list(
  k = as.integer(best_knn_row$param_knn__n_neighbors),
  metric = best_knn_row$param_knn__metric,
  weights = best_knn_row$param_knn__weights
)

cat("\nBest k-NN parameters:\n")
print(best_knn_parameters)
cat(sprintf("Best k-NN CV F1: %.4f\n", best_knn_row$mean_test_score))

write.csv(knn_results, "knn_grid_search_results.csv", row.names = FALSE)

knn_results$combo <- paste(knn_results$param_knn__metric, knn_results$param_knn__weights, sep = ", ")
knn_combo_levels <- c(
  "euclidean, uniform",
  "euclidean, distance",
  "manhattan, uniform",
  "manhattan, distance"
)
knn_results$combo <- factor(knn_results$combo, levels = knn_combo_levels)

p_knn <- ggplot(
  knn_results,
  aes(
    x = param_knn__n_neighbors,
    y = mean_test_score,
    color = combo,
    linetype = combo,
    shape = combo
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.2) +
  scale_x_continuous(breaks = K_VALUES) +
  scale_linetype_manual(values = c("solid", "dashed", "dotdash", "dotted")) +
  scale_shape_manual(values = c(16, 15, 17, 18)) +
  labs(
    title = "k-NN parameter sweep",
    x = "Number of neighbours, k",
    y = "Mean cross-validation F1",
    color = NULL,
    linetype = NULL,
    shape = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    panel.grid.minor = element_blank()
  )

save_plot(p_knn, "knn_k_sweep.png", width = 8, height = 5)

# ---------------------------------------------------------------- 5. RBF SVM parameter sweep

cat("\nRunning RBF SVM grid search...\n")

rbf_grid_df <- expand.grid(
  param_svm__C = SVM_C_VALUES,
  param_svm__gamma = RBF_GAMMA_VALUES,
  stringsAsFactors = FALSE
)

rbf_cv_list <- future_lapply(
  seq_len(nrow(rbf_grid_df)),
  function(i) {
    row <- rbf_grid_df[i, ]
    scores <- cv_f1_scores(
      X_sw, y_sw, cv_folds_sw,
      model_type = "svm",
      params = list(
        kernel = "rbf",
        C = row$param_svm__C,
        gamma = row$param_svm__gamma
      )
    )
    c(mean_test_score = mean(scores), std_test_score = pop_sd(scores))
  },
  future.seed = TRUE,
  future.packages = c("e1071")
)

rbf_results <- cbind(rbf_grid_df, as.data.frame(do.call(rbind, rbf_cv_list)))
rbf_results$rank_test_score <- rank(-rbf_results$mean_test_score, ties.method = "min")

best_rbf_idx <- which.max(rbf_results$mean_test_score)
best_rbf_row <- rbf_results[best_rbf_idx, ]

best_rbf_parameters <- list(
  C = as.numeric(best_rbf_row$param_svm__C),
  gamma = as.numeric(best_rbf_row$param_svm__gamma)
)

cat("\nBest RBF SVM parameters:\n")
print(best_rbf_parameters)
cat(sprintf("Best RBF SVM CV F1: %.4f\n", best_rbf_row$mean_test_score))

write.csv(rbf_results, "svm_rbf_grid_search_results.csv", row.names = FALSE)

rbf_mid <- (min(rbf_results$mean_test_score) + max(rbf_results$mean_test_score)) / 2
rbf_plot_df <- rbf_results
rbf_plot_df$gamma_f <- factor(rbf_plot_df$param_svm__gamma, levels = RBF_GAMMA_VALUES)
rbf_plot_df$C_f <- factor(rbf_plot_df$param_svm__C, levels = rev(SVM_C_VALUES))
rbf_plot_df$text_color <- ifelse(rbf_plot_df$mean_test_score < rbf_mid, "white", "black")

p_rbf <- ggplot(rbf_plot_df, aes(x = gamma_f, y = C_f, fill = mean_test_score)) +
  geom_tile() +
  geom_text(
    aes(label = sprintf("%.3f", mean_test_score), color = text_color),
    size = 3.8,
    show.legend = FALSE
  ) +
  scale_fill_viridis_c(name = "Mean CV F1") +
  scale_color_identity() +
  labs(
    title = "RBF SVM: mean cross-validation F1",
    x = "Gamma",
    y = "C"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid = element_blank())

save_plot(p_rbf, "svm_rbf_c_gamma_heatmap.png", width = 7, height = 5)

# ---------------------------------------------------------------- 6. Polynomial SVM parameter sweep

cat("\nRunning Polynomial SVM randomized search (24 combinations)...\n")

poly_full_grid <- expand.grid(
  param_svm__C = POLY_C_VALUES,
  param_svm__degree = POLY_DEGREES,
  param_svm__gamma = POLY_GAMMA_VALUES,
  param_svm__coef0 = POLY_COEF0_VALUES,
  stringsAsFactors = FALSE
)

set.seed(SEED)
poly_sample_idx <- sample(seq_len(nrow(poly_full_grid)), size = min(24L, nrow(poly_full_grid)))
poly_grid_df <- poly_full_grid[poly_sample_idx, , drop = FALSE]
rownames(poly_grid_df) <- NULL

poly_cv_list <- future_lapply(
  seq_len(nrow(poly_grid_df)),
  function(i) {
    row <- poly_grid_df[i, ]
    scores <- cv_f1_scores(
      X_sw, y_sw, cv_folds_sw,
      model_type = "svm",
      params = list(
        kernel = "poly",
        C = row$param_svm__C,
        degree = row$param_svm__degree,
        gamma = row$param_svm__gamma,
        coef0 = row$param_svm__coef0
      )
    )
    c(mean_test_score = mean(scores), std_test_score = pop_sd(scores))
  },
  future.seed = TRUE,
  future.packages = c("e1071")
)

poly_results <- cbind(poly_grid_df, as.data.frame(do.call(rbind, poly_cv_list)))
poly_results$rank_test_score <- rank(-poly_results$mean_test_score, ties.method = "min")

best_poly_idx <- which.max(poly_results$mean_test_score)
best_poly_row <- poly_results[best_poly_idx, ]

best_poly_parameters <- list(
  C = as.numeric(best_poly_row$param_svm__C),
  degree = as.integer(best_poly_row$param_svm__degree),
  gamma = best_poly_row$param_svm__gamma,
  coef0 = as.numeric(best_poly_row$param_svm__coef0)
)

cat("\nBest polynomial SVM parameters:\n")
print(best_poly_parameters)
cat(sprintf("Best polynomial SVM CV F1: %.4f\n", best_poly_row$mean_test_score))

write.csv(poly_results, "svm_polynomial_grid_search_results.csv", row.names = FALSE)

poly_plot_data <- poly_results |>
  group_by(param_svm__degree, param_svm__C) |>
  summarise(mean_test_score = max(mean_test_score), .groups = "drop") |>
  mutate(Degree = factor(paste("Degree", param_svm__degree), levels = paste("Degree", POLY_DEGREES)))

p_poly <- ggplot(
  poly_plot_data,
  aes(x = param_svm__C, y = mean_test_score, color = Degree, group = Degree)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  scale_x_log10(breaks = POLY_C_VALUES) +
  labs(
    title = "Polynomial SVM: effect of C and polynomial degree",
    x = "C",
    y = "Best mean cross-validation F1",
    color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())

save_plot(p_poly, "svm_polynomial_sweep.png", width = 8, height = 5)

# ---------------------------------------------------------------- 7. RBF versus polynomial SVM comparison

kernel_comparison <- data.frame(
  Kernel = c("RBF", "Polynomial"),
  `Best CV F1` = c(best_rbf_row$mean_test_score, best_poly_row$mean_test_score),
  `Best parameters` = c(format_params(best_rbf_parameters), format_params(best_poly_parameters)),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

write.csv(kernel_comparison, "svm_kernel_comparison.csv", row.names = FALSE)

cat("\nSVM kernel comparison:\n")
print(kernel_comparison, row.names = FALSE)

lower_limit <- max(0.0, min(kernel_comparison$`Best CV F1`) - 0.05)
upper_limit <- min(1.0, max(kernel_comparison$`Best CV F1`) + 0.02)
if (upper_limit <= lower_limit) {
  lower_limit <- 0.0
  upper_limit <- 1.0
}

p_kernel_comp <- ggplot(kernel_comparison, aes(x = Kernel, y = `Best CV F1`, fill = Kernel)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(
    aes(label = sprintf("%.4f", `Best CV F1`)),
    vjust = -0.4,
    size = 3.8
  ) +
  scale_fill_manual(values = c("RBF" = "#1f77b4", "Polynomial" = "#ff7f0e")) +
  coord_cartesian(ylim = c(lower_limit, upper_limit)) +
  labs(
    title = "SVM kernel comparison",
    x = NULL,
    y = "Best mean cross-validation F1"
  ) +
  theme_minimal(base_size = 11)

save_plot(p_kernel_comp, "svm_kernel_comparison.png", width = 6, height = 4.5)

# ---------------------------------------------------------------- 8. Select the SVM kernel from CV only

if (best_rbf_row$mean_test_score >= best_poly_row$mean_test_score) {
  selected_svm_kernel <- "rbf"
  selected_svm_parameters <- c(list(kernel = "rbf"), best_rbf_parameters)
} else {
  selected_svm_kernel <- "poly"
  selected_svm_parameters <- c(list(kernel = "poly"), best_poly_parameters)
}

cat("\nSelected SVM kernel:", selected_svm_kernel, "\n")
cat("Selected SVM parameters:", format_params(selected_svm_parameters), "\n")
cat("Selected k-NN parameters:", format_params(best_knn_parameters), "\n")

# ---------------------------------------------------------------- 9. Cross-validation stability of selected models

cat("\nEvaluating CV stability on full training set...\n")

svm_cv <- evaluate_cv(
  X_train, y_train, cv_folds_train,
  model_type = "svm",
  params = selected_svm_parameters,
  name = sprintf("Selected %s SVM", toupper(selected_svm_kernel))
)

knn_cv <- evaluate_cv(
  X_train, y_train, cv_folds_train,
  model_type = "knn",
  params = best_knn_parameters,
  name = "Selected k-NN"
)

selected_model_cv_summary <- data.frame(
  Model = c(sprintf("%s SVM", toupper(selected_svm_kernel)), "k-NN"),
  `Mean CV F1` = c(svm_cv$mean, knn_cv$mean),
  `CV F1 standard deviation` = c(svm_cv$std, knn_cv$std),
  check.names = FALSE
)

write.csv(selected_model_cv_summary, "selected_model_cv_summary.csv", row.names = FALSE)

# ---------------------------------------------------------------- 10. PCA component sweep

cat("\nRunning PCA component sweep (1 to", N_FEATURES, "components)...\n")

pca_component_values <- seq_len(N_FEATURES)

pca_sweep_list <- future_lapply(
  pca_component_values,
  function(n_comp) {
    svm_s <- cv_f1_scores(
      X_sw, y_sw, cv_folds_sw,
      model_type = "svm",
      params = selected_svm_parameters,
      use_pca = TRUE,
      n_components = n_comp
    )
    knn_s <- cv_f1_scores(
      X_sw, y_sw, cv_folds_sw,
      model_type = "knn",
      params = best_knn_parameters,
      use_pca = TRUE,
      n_components = n_comp
    )
    c(
      svm_mean = mean(svm_s),
      svm_std = pop_sd(svm_s),
      knn_mean = mean(knn_s),
      knn_std = pop_sd(knn_s)
    )
  },
  future.seed = TRUE,
  future.packages = c("e1071", "kknn")
)

pca_sweep_mat <- do.call(rbind, pca_sweep_list)

pca_sweep_table <- data.frame(
  `PCA components` = pca_component_values,
  `SVM mean CV F1` = pca_sweep_mat[, "svm_mean"],
  `SVM CV F1 standard deviation` = pca_sweep_mat[, "svm_std"],
  `k-NN mean CV F1` = pca_sweep_mat[, "knn_mean"],
  `k-NN CV F1 standard deviation` = pca_sweep_mat[, "knn_std"],
  check.names = FALSE
)

write.csv(pca_sweep_table, "pca_component_sweep_results.csv", row.names = FALSE)

svm_label <- sprintf("%s SVM", toupper(selected_svm_kernel))
pca_plot_long <- rbind(
  data.frame(
    Components = pca_component_values,
    F1 = pca_sweep_table$`SVM mean CV F1`,
    Model = svm_label
  ),
  data.frame(
    Components = pca_component_values,
    F1 = pca_sweep_table$`k-NN mean CV F1`,
    Model = "k-NN"
  )
)

threshold_label <- sprintf("%.0f%% variance threshold (%d)", PCA_VARIANCE_THRESHOLD * 100, N_COMP)

p_pca_sweep <- ggplot(pca_plot_long, aes(x = Components, y = F1, color = Model, shape = Model)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.4) +
  geom_vline(aes(xintercept = N_COMP, linetype = threshold_label), color = "gray40") +
  scale_x_continuous(breaks = pca_component_values) +
  scale_linetype_manual(name = NULL, values = setNames("dashed", threshold_label)) +
  labs(
    title = "Effect of PCA dimensionality on classification",
    x = "Number of PCA components",
    y = "Mean cross-validation F1",
    color = NULL,
    shape = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())

save_plot(p_pca_sweep, "pca_component_sweep.png", width = 8, height = 5)

# ---------------------------------------------------------------- 11. Final four configurations

cat("\nTraining and evaluating final four configurations...\n")

configurations <- list(
  "SVM - without PCA" = list(model_type = "svm", params = selected_svm_parameters, use_pca = FALSE),
  "SVM - with PCA"    = list(model_type = "svm", params = selected_svm_parameters, use_pca = TRUE),
  "k-NN - without PCA" = list(model_type = "knn", params = best_knn_parameters, use_pca = FALSE),
  "k-NN - with PCA"    = list(model_type = "knn", params = best_knn_parameters, use_pca = TRUE)
)

summary_rows <- list()
cm_plots <- list()

for (config_name in names(configurations)) {
  cfg <- configurations[[config_name]]

  t_train_start <- proc.time()[["elapsed"]]
  fitted_model <- fit_pipeline(
    X_train, y_train,
    model_type = cfg$model_type,
    params = cfg$params,
    use_pca = cfg$use_pca,
    n_components = N_COMP
  )
  training_time <- proc.time()[["elapsed"]] - t_train_start

  t_pred_start <- proc.time()[["elapsed"]]
  predictions <- predict_pipeline(fitted_model, X_test)
  prediction_time <- proc.time()[["elapsed"]] - t_pred_start

  m <- compute_metrics(y_test, predictions)

  summary_rows[[config_name]] <- data.frame(
    Configuration = config_name,
    `SVM kernel` = if (startsWith(config_name, "SVM")) selected_svm_kernel else "Not applicable",
    `No. of features` = if (cfg$use_pca) N_COMP else N_FEATURES,
    Accuracy = m$accuracy,
    Precision = m$precision,
    Recall = m$recall,
    F1 = m$f1,
    `True stable` = m$tn,
    `False unstable` = m$fp,
    `False stable` = m$fn,
    `True unstable` = m$tp,
    `Train time (s)` = training_time,
    `Predict time (s)` = prediction_time,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  cm_df <- data.frame(
    Actual = factor(c("stable", "stable", "unstable", "unstable"), levels = c("unstable", "stable")),
    Predicted = factor(c("stable", "unstable", "stable", "unstable"), levels = c("stable", "unstable")),
    Count = c(m$tn, m$fp, m$fn, m$tp)
  )
  cm_mid <- (min(cm_df$Count) + max(cm_df$Count)) / 2
  cm_df$text_color <- ifelse(cm_df$Count > cm_mid, "white", "black")

  cm_plots[[config_name]] <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(
      aes(label = Count, color = text_color),
      size = 4.5,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_fill_gradient(low = "#f7fbff", high = "#084594", guide = "none") +
    scale_color_identity() +
    labs(
      title = config_name,
      x = "Predicted label",
      y = "True label"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5)
    )
}

p_cm_all <- wrap_plots(cm_plots, nrow = 1)
save_plot(p_cm_all, "confusion_matrices.png", width = 18, height = 4.5)

# ---------------------------------------------------------------- 12. Save and print final results

summary_df <- do.call(rbind, summary_rows)
rownames(summary_df) <- NULL

write.csv(summary_df, "summary_table_full_precision.csv", row.names = FALSE)

rounded_summary <- summary_df
numeric_columns <- c(
  "Accuracy", "Precision", "Recall", "F1",
  "Train time (s)", "Predict time (s)"
)
rounded_summary[numeric_columns] <- lapply(rounded_summary[numeric_columns], round, digits = 4)

write.csv(rounded_summary, "summary_table.csv", row.names = FALSE)

cat("\nFinal test-set results:\n")
print(rounded_summary, row.names = FALSE)

best_configuration_index <- which.max(summary_df$F1)
best_configuration <- summary_df[best_configuration_index, c(
  "Configuration", "SVM kernel", "No. of features",
  "Accuracy", "Precision", "Recall", "F1",
  "Train time (s)", "Predict time (s)"
)]

cat("\nBest final configuration by test F1:\n")
print(t(best_configuration), quote = FALSE)

cat("\nGenerated files:\n")
cat(paste(
  c(
    "pca_explained_variance.csv",
    "pca_variance.png",
    "knn_grid_search_results.csv",
    "knn_k_sweep.png",
    "svm_rbf_grid_search_results.csv",
    "svm_rbf_c_gamma_heatmap.png",
    "svm_polynomial_grid_search_results.csv",
    "svm_polynomial_sweep.png",
    "svm_kernel_comparison.csv",
    "svm_kernel_comparison.png",
    "selected_model_cv_summary.csv",
    "pca_component_sweep_results.csv",
    "pca_component_sweep.png",
    "confusion_matrices.png",
    "summary_table.csv",
    "summary_table_full_precision.csv"
  ),
  collapse = "\n"
), "\n")