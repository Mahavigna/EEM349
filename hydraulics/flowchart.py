import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10.5, 12.4), dpi=220)
ax.set_xlim(0, 10)
ax.set_ylim(0, 13.5)
ax.axis("off")

C_PURPLE = "#4B1D6B"
C_PURPLE_BG = "#F4EFF8"
C_BLUE = "#1D4E89"
C_BLUE_BG = "#EDF4FB"
C_TEAL = "#1B6B6B"
C_TEAL_BG = "#EAF6F6"
C_SLATE = "#2C3E50"
C_SLATE_BG = "#F4F6F8"
C_GOLD = "#8A5A00"
C_GOLD_BG = "#FFF8E7"
C_ARROW = "#34495E"


def draw_box(
    x,
    y,
    w,
    h,
    title,
    body="",
    edge_color=C_SLATE,
    face_color=C_SLATE_BG,
    title_size=10,
    body_size=8.6,
):
    box = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.14",
        linewidth=1.6,
        edgecolor=edge_color,
        facecolor=face_color,
        zorder=3,
    )
    ax.add_patch(box)
    if body:
        ax.text(
            x,
            y + h * 0.21,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=edge_color,
            zorder=4,
        )
        ax.text(
            x,
            y - h * 0.16,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color="#212529",
            linespacing=1.32,
            zorder=4,
        )
    else:
        ax.text(
            x,
            y,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=edge_color,
            zorder=4,
        )


def draw_arrow(x1, y1, x2, y2):
    arrow = patches.FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>,head_length=6.5,head_width=4.2",
        linewidth=1.55,
        color=C_ARROW,
        zorder=5,
    )
    ax.add_patch(arrow)


def draw_line(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=C_ARROW, linewidth=1.55, zorder=5)


def draw_stage_badge(x, y, text):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.30,rounding_size=0.12",
            facecolor="#495057",
            edgecolor="none",
        ),
        zorder=6,
    )


# Title Banner
ax.text(
    5.0,
    13.15,
    "Experimental Implementation & Controlled PCA Evaluation Workflow",
    ha="center",
    va="center",
    fontsize=13.5,
    fontweight="bold",
    color=C_PURPLE,
)
ax.text(
    5.0,
    12.80,
    "UCI Hydraulic Systems (2,205 cycles, 17 sensors -> 34 features) | Pipeline-Encapsulated Leakage Prevention",
    ha="center",
    va="center",
    fontsize=9,
    color="#555555",
)

# Step 1
draw_box(
    5.0,
    11.95,
    8.2,
    0.96,
    "1. Data Ingestion & Statistical Feature Extraction (common_data.py)",
    "Load 17 multi-rate sensor files (1-100 Hz) + profile.txt (exclude cols 1-4; target = stable_flag)\n"
    "Compute cycle mean & std -> 34 features | Validate: 0 NaNs, 0 label conflicts, deduplicate",
    edge_color=C_PURPLE,
    face_color=C_PURPLE_BG,
)

# Arrow 1 -> 2
draw_arrow(5.0, 11.42, 5.0, 11.00)

# Step 2
draw_box(
    5.0,
    10.48,
    8.2,
    0.92,
    "2. Stratified Train / Test Partition (seed = 42, stratify = y)",
    "Training Set: 1,764 cycles (80%)  |  Unseen Test Set: 441 cycles (20%: 290 stable, 151 unstable)\n"
    "Shared 3-Fold StratifiedKFold CV (scoring = F1) + PCA fit on X_train only -> 95% variance = 8 PCs",
    edge_color=C_SLATE,
    face_color=C_SLATE_BG,
)

# Split orthogonal connector routed to x=1.5 and x=8.5, cleanly clear of centered badge!
draw_line(5.0, 9.97, 5.0, 9.70)
draw_line(1.55, 9.70, 8.45, 9.70)
draw_arrow(1.55, 9.70, 1.55, 8.95)
draw_arrow(8.45, 9.70, 8.45, 8.95)

# Stage 1 Group Box
stage1_bg = patches.FancyBboxPatch(
    (0.32, 6.48),
    9.36,
    2.86,
    boxstyle="round,pad=0.05,rounding_size=0.16",
    linewidth=1.2,
    linestyle="--",
    edgecolor="#868E96",
    facecolor="#FAFBFD",
    zorder=1,
)
ax.add_patch(stage1_bg)
draw_stage_badge(
    5.0,
    9.34,
    "STAGE 1: Baseline Hyperparameter Sweeps on Full 34 Features (Training Set Only)",
)

# 3A & 3B
draw_box(
    2.65,
    8.20,
    4.25,
    1.38,
    "3A. k-NN GridSearchCV (knn.py)",
    "Pipeline: StandardScaler -> k-NN\n"
    "Sweep: K in {1, 3, ..., 31}, L1/L2, uniform/distance\n"
    "Winner: K=1, Manhattan, uniform (CV F1 = 0.9396)",
    edge_color=C_BLUE,
    face_color=C_BLUE_BG,
)

draw_box(
    7.35,
    8.20,
    4.25,
    1.38,
    "3B. SVM Kernel Selection (svm.py)",
    "Pipeline: StandardScaler -> SVC\n"
    "RBF GridSearch (0.9426) vs. Poly RandSearch (0.9468)\n"
    "Winner: Poly (d=2, C=10, gamma=0.1, coef0=1.0)",
    edge_color=C_TEAL,
    face_color=C_TEAL_BG,
)

# Straight vertical arrows from 3A & 3B into Gold Lock Bar
draw_arrow(2.65, 7.46, 2.65, 7.12)
draw_arrow(7.35, 7.46, 7.35, 7.12)

# Gold Lock Bar
draw_box(
    5.0,
    6.84,
    8.2,
    0.46,
    "Lock Winning Baseline Hyperparameters (Isolate True Effect of PCA Reduction)",
    edge_color=C_GOLD,
    face_color=C_GOLD_BG,
    title_size=9.2,
)

# Vertical arrows routed at x=1.55 and x=8.45 cleanly clear of centered Stage 2 badge!
draw_arrow(1.55, 6.56, 1.55, 5.66)
draw_arrow(8.45, 6.56, 8.45, 5.66)

# Stage 2 Group Box
stage2_bg = patches.FancyBboxPatch(
    (0.32, 3.55),
    9.36,
    2.52,
    boxstyle="round,pad=0.05,rounding_size=0.16",
    linewidth=1.2,
    linestyle="--",
    edgecolor="#868E96",
    facecolor="#FAFBFD",
    zorder=1,
)
ax.add_patch(stage2_bg)
draw_stage_badge(
    5.0,
    6.07,
    "STAGE 2: Controlled PCA Dimensionality Sweep & 4-Configuration Model Fitting",
)

# 4A & 4B
draw_box(
    2.65,
    4.92,
    4.25,
    1.38,
    "4A. k-NN Controlled Evaluation",
    "PCA Component Sweep: k = 1 to 34 (3-fold CV)\n"
    "Config 1: StandardScaler -> k-NN (34 features)\n"
    "Config 2: StandardScaler -> PCA(8) -> k-NN (8 PCs)",
    edge_color=C_BLUE,
    face_color=C_BLUE_BG,
)

draw_box(
    7.35,
    4.92,
    4.25,
    1.38,
    "4B. SVM Controlled Evaluation",
    "PCA Component Sweep: k = 1 to 34 (3-fold CV)\n"
    "Config 3: StandardScaler -> Poly SVM (34 features)\n"
    "Config 4: StandardScaler -> PCA(8) -> Poly SVM (8 PCs)",
    edge_color=C_TEAL,
    face_color=C_TEAL_BG,
)

# Leakage Note centered cleanly below 4A/4B inside Stage 2 box
ax.text(
    5.0,
    3.85,
    "All Pipelines fit StandardScaler & PCA strictly on X_train (zero test data leakage)",
    ha="center",
    va="center",
    fontsize=8.5,
    style="italic",
    fontweight="bold",
    color=C_PURPLE,
    zorder=4,
)

# Orthogonal merge from 4A & 4B at x=1.55 and x=8.45 (completely clear of the centered italic text!)
draw_line(1.55, 4.18, 1.55, 3.30)
draw_line(8.45, 4.18, 8.45, 3.30)
draw_line(1.55, 3.30, 8.45, 3.30)
draw_arrow(5.0, 3.30, 5.0, 2.94)

# Step 5
draw_box(
    5.0,
    2.40,
    8.2,
    0.96,
    "5. Single-Pass Unseen Test Evaluation (N_test = 441) & End-to-End Timing",
    "Transform X_test via fitted Pipeline  |  Record Train & Predict latency via time.perf_counter()\n"
    "Compute Accuracy, Precision, Recall, F1-Score, and Confusion Matrices (TP, TN, FP, FN)",
    edge_color=C_PURPLE,
    face_color=C_PURPLE_BG,
)

# Arrow 5 -> 6
draw_arrow(5.0, 1.87, 5.0, 1.48)

# Step 6
draw_box(
    5.0,
    0.88,
    8.2,
    1.08,
    "6. Four-Configuration Comparison & Engineering Meta-Analysis",
    "Best Overall: Polynomial SVM without PCA (Acc = 98.19%, F1 = 0.9728, 0 FP, 4.72 us/cycle)\n"
    "Largest PCA Speedup: k-NN prediction 5.65x faster (47.36 -> 8.29 us/cycle; F1 = 0.8851)\n"
    "Outputs: CSV summaries, sweep curves, confusion matrices, and edge-deployment recommendation",
    edge_color=C_SLATE,
    face_color=C_SLATE_BG,
)

plt.tight_layout()
plt.savefig(
    "Figure_4_1_Workflow_Flowchart.png", dpi=220, bbox_inches="tight", facecolor="white"
)
plt.close()
print("Saved final spotless Figure_4_1_Workflow_Flowchart.png")
