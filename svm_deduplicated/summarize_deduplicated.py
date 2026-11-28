from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT / "submissions" / "svm_deduplicated"

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

pred = pd.read_csv(OUTPUTS / "predictions.csv")

records = []

for fold in range(1, 6):
    test = pred.loc[pred["outer_fold"] == fold]
    y_true = test["y_true"]
    y_pred = test["y_pred"]

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    records.append({
        "fold": fold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro"
        ),
        "malignant_sensitivity": tp / (tp + fn),
        "benign_specificity": tn / (tn + fp),
    })

results = pd.DataFrame(records)
metrics = results.columns.drop("fold")

print("\nPER-FOLD RESULTS")
print(results.round(4).to_string(index=False))

print("\nMEAN RESULTS")
print(results[metrics].mean().round(4).to_string())

print("\nSAMPLE STANDARD DEVIATION")
print(results[metrics].std(ddof=1).round(4).to_string())

print("\nTOTAL RECORDED RUNTIME")
timings = pd.read_csv(OUTPUTS / "timings.csv")

print(round(timings["total_seconds"].sum(), 2), "seconds")
