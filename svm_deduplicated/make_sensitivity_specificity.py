
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

# Original saved results (private, not uploaded to GitHub).
# Locate the repository root automatically.
SOURCE = Path(__file__).resolve().parent.parent
PREDICTIONS = (
    SOURCE / "submissions" / "svm_deduplicated" / "predictions.csv"
)

# GitHub repository output folder.
REPO = Path(__file__).resolve().parent.parent
OUTPUT = (
    REPO / "svm_deduplicated" / "figures"
    / "svm_deduplicated_sensitivity_specificity.png"
)

pred = pd.read_csv(PREDICTIONS)
records = []

for fold in range(1, 6):
    subset = pred[pred["outer_fold"] == fold]

    tn, fp, fn, tp = confusion_matrix(
        subset["y_true"],
        subset["y_pred"],
        labels=[0, 1],
    ).ravel()

    records.append({
        "fold": fold,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    })

results = pd.DataFrame(records)

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    results["fold"],
    results["sensitivity"],
    marker="o",
    linewidth=2.5,
    markersize=8,
    color="#006D77",
    label="Malignant sensitivity",
)

ax.plot(
    results["fold"],
    results["specificity"],
    marker="s",
    linewidth=2.5,
    markersize=8,
    color="#49A69C",
    label="Benign specificity",
)

ax.set_title(
    "Deduplicated SVM (246 Features)\n"
    "Malignant Sensitivity and Benign Specificity"
)
ax.set_xlabel("Outer fold")
ax.set_ylabel("Score")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_ylim(0.60, 1.0)
ax.grid(alpha=0.2)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=2,
    frameon=False,
)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print(results.round(4).to_string(index=False))
print("Figure saved:", OUTPUT)
