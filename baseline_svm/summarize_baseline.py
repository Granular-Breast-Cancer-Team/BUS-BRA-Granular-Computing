from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

PROJECT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT / "submissions" / "baseline"
predictions = pd.read_csv(OUTPUTS / "predictions.csv")

records = []

for fold, rows in predictions.groupby("outer_fold"):
    actual = rows["y_true"]
    predicted = rows["y_pred"]

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    records.append({
        "outer_fold": fold,
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro"),
        "malignant_sensitivity": tp / (tp + fn),
        "benign_specificity": tn / (tn + fp),
    })

scores = pd.DataFrame(records)
scores.to_csv(OUTPUTS / "baseline_fold_metrics.csv", index=False)

print("TEST SCORES PER FOLD")
print(scores.round(4).to_string(index=False))

print("\nMEAN AND STANDARD DEVIATION ACROSS FOLDS")
summary = scores.drop(columns="outer_fold").agg(["mean", "std"])
print(summary.round(4).to_string())
summary.to_csv(OUTPUTS / "baseline_metrics_summary.csv")