from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Repository root
PROJECT = Path(__file__).resolve().parent.parent

# Private predictions are supplied locally, not committed to GitHub.
OUT = PROJECT / "svm_deduplicated" / "figures"
PREDICTIONS = PROJECT / "submissions" / "svm_deduplicated" / "predictions.csv"
OUTPUT_FILE = OUT / "svm_deduplicated_confusion_matrix.png"

pred = pd.read_csv(PREDICTIONS)

cm = confusion_matrix(
    pred["y_true"],
    pred["y_pred"],
    labels=[0, 1],
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"],
)

fig, ax = plt.subplots(figsize=(7, 6))
display.plot(ax=ax, cmap="BuGn", colorbar=False, values_format="d")

ax.set_title(
    "Deduplicated SVM (246 Features)\nAggregated Out-of-Fold Confusion Matrix"
)
ax.set_xlabel("Predicted diagnosis")
ax.set_ylabel("Actual diagnosis")

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Confusion matrix:")
print(cm)
print("Figure saved:", OUTPUT_FILE)