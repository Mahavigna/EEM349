# Predictive Maintenance of Hydraulic Systems Using k-NN and Polynomial SVM with PCA Dimensionality Reduction

This README is internal reference for our though process and final execution packaged with the code

**Course:** EEM349 Computational Intelligence (Academic Session 2026/2027, Semester I)  
**Theme:** Manufacturing & Industry 4.0  
**Group:** Group 16  
**Authors:**
- Edmund Chin You Sheng (-)
- Mahavigna Jyothi Eswera A/L K Murthy (-)

**Supervisor:** -

---

## Problem Statement

Modern Industry 4.0 facilities depend on stable hydraulic systems to operate automated production lines, industrial presses, heavy machinery, programmable logic controllers, and robotic actuators. Gradual degradation across coolers, valves, internal pumps, and hydraulic accumulators can destabilize system dynamics well before catastrophic failure occurs, leading to costly unplanned downtime.

This project applies leakage-free supervised machine learning pipelines in Python to classify 60-second hydraulic operating cycles as **stable (`0`)** or **unstable (`1`)**, evaluating compact, edge-ready classical algorithms alongside Principal Component Analysis (PCA) dimensionality reduction.

---

## Task & Dataset

- **Task:** Supervised Binary Classification (`stable_flag`: `0` = stable, `1` = unstable; unstable is treated as the positive class to prioritize early fault-risk detection).
- **Dataset:** [Condition Monitoring of Hydraulic Systems — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) (Helwig et al., 2015, 2018).
- **Data Scale:** 2,205 repeated 60-second load cycles recorded across 17 multi-rate physical and virtual sensors (1–100 Hz): pressures (`PS1–PS6`), motor power (`EPS1`), volume flows (`FS1–FS2`), oil temperatures (`TS1–TS4`), vibration (`VS1`), and efficiency/cooling indicators (`CE`, `CP`, `SE`).
- **Feature Engineering & Preprocessing:**
  - Computing cycle-level arithmetic mean and standard deviation across all 17 sensors yields **34 numerical input features**.
  - Component degradation columns (cols 1–4 of `profile.txt`) are strictly excluded to prevent target leakage.
  - Automated validation checks confirm 0 missing values, 0 conflicting labels, and deduplication prior to splitting.
  - Partitioned using an **80/20 stratified train–test split** (1,764 training cycles / 441 unseen test cycles; `random_state = 42`).
  - `StandardScaler` and `PCA` are strictly encapsulated inside `scikit-learn` `Pipeline` objects so all statistics and eigenvectors are fitted on training data only.

---

## Objectives

1. Extract, validate, and preprocess 34 cycle-level statistical features from 17 multi-rate hydraulic process sensors free of data leakage using an 80/20 stratified split and training-fitted Z-score standardization.
2. Implement and systematically tune two distinct supervised classification algorithms—**k-Nearest Neighbours (k-NN)** and **Support Vector Machine (SVC)**—via 3-fold stratified cross-validation (optimizing positive-class F1-score) to achieve at least 85% test F1-score on unstable cycle detection.
3. Apply **Principal Component Analysis (PCA)** at a 95% cumulative explained variance threshold to compress the 34 correlated sensor features into an orthogonal lower-dimensional subspace (8 principal components) and run a full 1–34 component sweep.
4. Compare all four model configurations (k-NN and SVM, with and without PCA) using Accuracy, Precision, Recall, F1-score, Confusion Matrices, training time, and prediction time to identify the best trade-off for Industry 4.0 and TinyML edge deployment.

---

## Algorithms & Experimental Protocol

- **k-Nearest Neighbours (`knn.py`):** Non-parametric, instance-based baseline. Tuned via `GridSearchCV` across odd $K \in \{1, 3, \dots, 31\}$, distance metrics (`euclidean`, `manhattan`), and weighting (`uniform`, `distance`).
  - *Best baseline:* $K = 1$, Manhattan ($L_1$) distance, uniform weighting (3-fold CV F1 = `0.9396`).
- **Support Vector Machine (`svm.py`):** Discriminative maximum-margin kernel classifier. Compares an exhaustive `GridSearchCV` for the **RBF kernel** ($C, \gamma$) against a 24-iteration `RandomizedSearchCV` for the **Polynomial kernel** ($C$, degree $d$, $\gamma$, $\text{coef0}$), selecting the winning kernel strictly by training CV F1-score.
  - *Best baseline:* Polynomial kernel ($d = 2$, $C = 10$, $\gamma = 0.1$, $\text{coef0} = 1.0$; 3-fold CV F1 = `0.9468`).
- **Controlled PCA Evaluation:** Winning baseline hyperparameters are locked constant while adding `PCA(n_components=8)` (95.70% cumulative variance, 76.47% feature reduction) to isolate the true effect of dimensionality reduction.

---

## Summary of Test-Set Results ($N_{\text{test}} = 441$)

| Configuration | Features | Accuracy | Precision | Recall | F1-Score | Train Time (s) | Predict Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **k-NN (without PCA)** | 34 | 0.9773 | 0.9862 | 0.9470 | 0.9662 | 0.0028 | 0.0209 |
| **k-NN (with PCA)** | 8 | 0.9229 | 0.9034 | 0.8675 | 0.8851 | 0.0036 | 0.0037 |
| **Polynomial SVM (without PCA)** | 34 | **0.9819** | **1.0000** | **0.9470** | **0.9728** | 0.0279 | **0.0021** |
| **Polynomial SVM (with PCA)** | 8 | 0.9252 | 0.8882 | 0.8940 | 0.8911 | 0.0432 | 0.0029 |

**Key Takeaways:**
- **Best Overall:** Polynomial SVM without PCA achieved the highest accuracy (`98.19%`), highest F1-score (`0.9728`), zero false alarms (`0 FP`), and fastest inference (`4.72 µs/cycle`).
- **Effect of 95% PCA:** Compressing 34 features to 8 components accelerated k-NN prediction by **5.65×**, but reduced F1-score by ~8.1 percentage points for both models because subtle fault signatures reside in lower-variance components (`PC9–PC21`).

---

## Project Structure

```text
.
├── common_data.py        # Shared data ingestion, validation, 80/20 split, PCA, and evaluation helpers
├── knn.py                # k-NN GridSearchCV, PCA dimensionality sweep (1–34), and test evaluation
├── svm.py                # RBF vs. Polynomial SVM sweeps, PCA sweep (1–34), and test evaluation
├── flowchart.py          # Generates the experimental workflow diagram (Figure_4_1_Workflow_Flowchart.png)
├── README.md             # Project documentation
└── *.txt                 # UCI Hydraulic Systems raw sensor files (PS1–PS6, EPS1, FS1–FS2, TS1–TS4, VS1, CE, CP, SE, profile.txt)
```

---

## Installation & Usage

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2. Place Dataset Files

Ensure `profile.txt` and all 17 sensor `.txt` files from the UCI Hydraulic Systems dataset are in the same directory as the scripts (or update `DATA_DIR` in `common_data.py`).

### 3. Run the Experiments

```bash
# Run k-NN hyperparameter sweep, PCA sweep, and test-set evaluation
python knn.py

# Run RBF vs. Polynomial SVM sweeps, PCA sweep, and test-set evaluation
python svm.py

# Regenerate the experimental workflow flowchart
python flowchart.py
```

Running `knn.py` and `svm.py` automatically outputs all CSV metric summaries, PCA explained variance tables, sweep curves, and confusion matrix figures into the working directory.

---

## References

- Helwig, N., Pignanelli, E., & Schütze, A. (2015). Condition monitoring of a complex hydraulic system using multivariate statistics. *IEEE I2MTC Proceedings*, 210–215.
- Helwig, N., Pignanelli, E., & Schütze, A. (2018). *Condition Monitoring of Hydraulic Systems* [Dataset]. UCI Machine Learning Repository.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.