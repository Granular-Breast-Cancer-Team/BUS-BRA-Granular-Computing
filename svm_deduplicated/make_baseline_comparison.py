
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

# Original private experiment outputs
# Locate the repository root automatically.
SOURCE = Path(__file__).resolve().parent.parent

BASELINE = SOURCE / "submissions" / "baseline" / "predictions.csv"
DEDUP = SOURCE / "submissions" / "svm_deduplicated" / "predictions.csv"

# GitHub figure destination
REPO = Path(__file__).resolve().parent.parent
OUTPUT = (
    REPO / "svm_deduplicated" / "figures"
    / "svm_288_vs_246_comparison.png"
)

def calculate_means(path):
    pred = pd.read_csv(path)
    records = []

    for fold in range(1, 6):
        subset = pred[pred["outer_fold"] == fold]
        y_true = subset["y_true"]
        y_pred = subset["y_pred"]

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        records.append({
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "macro_f1": f1_score(
                y_true, y_pred, average="macro"
            ),
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
        })

    return pd.DataFrame(records).mean()


original = calculate_means(BASELINE)
deduplicated = calculate_means(DEDUP)

metrics = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "sensitivity",
    "specificity",
]

labels = [
    "Accuracy",
    "Balanced\naccuracy",
    "Macro-F1",
    "Malignant\nsensitivity",
    "Benign\nspecificity",
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    x - width / 2,
    [original[m] for m in metrics],
    width,
    color="#663399",
    label="Original SVM (288 features)",
)

ax.bar(
    x + width / 2,
    [deduplicated[m] for m in metrics],
    width,
    color="#006D77",
    label="Deduplicated SVM (246 features)",
)

ax.set_title(
    "SVM Classification Performance\n"
    "Original 288 Features vs. Deduplicated 246 Features"
)
ax.set_ylabel("Five-fold mean score")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=2,
    frameon=False,
)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n288-FEATURE BASELINE")
print(original.round(4).to_string())

print("\n246-FEATURE DEDUPLICATED SVM")
print(deduplicated.round(4).to_string())

print("\nFigure saved:", OUTPUT)
