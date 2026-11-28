
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

# Read the original saved 246-feature predictions.
# Repository root
PROJECT = Path(__file__).resolve().parent.parent

# Private predictions must be supplied locally.
PREDICTIONS = (
    PROJECT / "submissions" / "svm_deduplicated" / "predictions.csv"
)

REPO = PROJECT

OUTPUT = REPO / "svm_deduplicated" / "figures" / "svm_deduplicated_fold_metrics_updated.png"
pred = pd.read_csv(PREDICTIONS)

records = []

for fold in range(1, 6):
    test = pred[pred["outer_fold"] == fold]

    records.append({
        "fold": fold,
        "accuracy": accuracy_score(test["y_true"], test["y_pred"]),
        "balanced_accuracy": balanced_accuracy_score(
            test["y_true"], test["y_pred"]
        ),
        "macro_f1": f1_score(
            test["y_true"], test["y_pred"], average="macro"
        ),
    })

results = pd.DataFrame(records)

x = np.arange(5)
width = 0.24

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.bar(
    x - width,
    results["accuracy"],
    width,
    label="Accuracy",
    color="#006D77",
)
ax.bar(
    x,
    results["balanced_accuracy"],
    width,
    label="Balanced accuracy",
    color="#49A69C",
)
ax.bar(
    x + width,
    results["macro_f1"],
    width,
    label="Macro-F1",
    color="#A1D6C5",
)

ax.set_title("Deduplicated SVM (246 Features)\nFive-Fold Classification Performance")
ax.set_xlabel("Outer fold")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {i}" for i in range(1, 6)])
ax.set_ylim(0, 1.0)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=3,
    frameon=False
)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print(results.round(4).to_string(index=False))
print("Figure saved:", OUTPUT)
