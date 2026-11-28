from pathlib import Path
from time import perf_counter
import json

import platform
from importlib.metadata import version

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from duplicate_selector import DuplicateFeatureSelector


# The experiment script lives inside svm_deduplicated/.
# Shared project files are located one directory above it.
PROJECT = Path(__file__).resolve().parent.parent
INPUTS = PROJECT / "inputs"

data = pd.read_csv(INPUTS / "BUSBRA_features_ready.csv")
feature_names = [
    line.strip()
    for line in (INPUTS / "allowed_feature_columns.txt").read_text().splitlines()
    if line.strip()
]

X = data[feature_names]
y = data["diagnosis"]
groups = data["patient_id"]
outer_folds = data["fold5_kfold"]

print("Baseline data loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Patient groups:", groups.nunique())

# This first loop displays counts only; it does not train models.
for fold in range(1, 6):
    test_mask = outer_folds == fold
    train_mask = outer_folds != fold
    print(
        f"Fold {fold}: {train_mask.sum()} training images, "
        f"{test_mask.sum()} testing images"
    )

baseline_pipeline = Pipeline([
    ("deduplicate", DuplicateFeatureSelector()),
    ("scaler", StandardScaler()),
    ("svm", SVC()),
])
print("\nBaseline pipeline:")
print(baseline_pipeline)

contract = json.loads(
    (PROJECT / "evaluation" / "evaluation_contract.json").read_text()
)

grid = contract["common_svm_grid"]
parameter_grid = [
    {
        "svm__kernel": ["linear"],
        "svm__C": grid["C"],
        "svm__class_weight": grid["class_weight"],
    },
    {
        "svm__kernel": ["rbf"],
        "svm__C": grid["C"],
        "svm__gamma": grid["rbf_gamma"],
        "svm__class_weight": grid["class_weight"],
    },
]
print("\nSVM parameter combinations:", len(ParameterGrid(parameter_grid)))

OUTPUTS = PROJECT / "submissions" / "svm_deduplicated"
OUTPUTS.mkdir(parents=True, exist_ok=True)
prediction_tables = []
parameter_records = []
timing_records = []
feature_records = []

# Every operation in this loop belongs to one outer evaluation round.
for outer_fold in range(1, 6):
    fold_start = perf_counter()
    train_mask = outer_folds != outer_fold
    test_mask = outer_folds == outer_fold

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    groups_train = groups.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]
    assert set(groups_train).isdisjoint(set(groups.loc[test_mask])), (
        "Patient overlap between outer training and testing"
    )

    inner_cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42 + outer_fold,
    )
    inner_splits = list(
        inner_cv.split(X_train, y_train, groups=groups_train)
    )
    print(f"\nInner validation for outer fold {outer_fold}:")
    for number, (fit_positions, val_positions) in enumerate(
        inner_splits, start=1
    ):
        fit_patients = set(groups_train.iloc[fit_positions])
        val_patients = set(groups_train.iloc[val_positions])
        overlap = fit_patients & val_patients
        assert not overlap, "Patient overlap detected"
        print(f"Inner split {number}: {len(overlap)} shared patients")

    search = GridSearchCV(
        estimator=baseline_pipeline,
        param_grid=parameter_grid,
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
        cv=inner_splits,
        refit="f1_macro",
        n_jobs=1,
         verbose=2,
        error_score="raise",
    )
    print(f"\nStarting inner validation for outer fold {outer_fold}...")
    search_start = perf_counter()
    search.fit(X_train, y_train)
    search_seconds = perf_counter() - search_start
    chosen = search.best_params_

    parameter_records.append({
        "outer_fold": outer_fold,
        "kernel": chosen["svm__kernel"],
        "C": chosen["svm__C"],
        "gamma": chosen.get("svm__gamma", ""),
        "class_weight": chosen["svm__class_weight"],
        "inner_macro_f1": search.best_score_,
        "inner_accuracy": search.cv_results_["mean_test_accuracy"][
            search.best_index_
        ],
    })
    pd.DataFrame(parameter_records).to_csv(
        OUTPUTS / "best_params.csv", index=False
    )
    print("\nBest SVM settings:", chosen)
    print("Mean inner-validation macro-F1:", round(search.best_score_, 4))

    predict_start = perf_counter()
    test_predictions = search.predict(X_test)
    predict_seconds = perf_counter() - predict_start
    total_seconds = perf_counter() - fold_start

    # Search time includes refitting; subtract it to avoid double-counting.
    timing_records.append({
        "outer_fold": outer_fold,
        "selector_seconds": float("nan"),
        "svm_tuning_seconds": max(0.0, search_seconds - search.refit_time_),
        "refit_seconds": search.refit_time_,
        "predict_seconds": predict_seconds,
        "total_seconds": total_seconds,
    })
    pd.DataFrame(timing_records).to_csv(
        OUTPUTS / "timings.csv", index=False
    )

    
    # Get the features retained by the trained duplicate selector.
    selector = search.best_estimator_.named_steps["deduplicate"]

    selected_names = [
        feature_names[i]
        for i in selector.keep_indices_
    ]

    print("Features retained:", len(selected_names))
    print("Redundant features removed:", len(feature_names) - len(selected_names))

    for feature_name in selected_names:
        feature_records.append({
            "outer_fold": outer_fold,
            "feature_name": feature_name,
        })

    pd.DataFrame(feature_records).to_csv(
        OUTPUTS / "selected_features.csv", index=False
    )


    fold_predictions = pd.DataFrame({
        "image_id": data.loc[test_mask, "image_id"].to_numpy(),
        "patient_id": groups.loc[test_mask].to_numpy(),
        "outer_fold": outer_fold,
        "y_true": y_test.to_numpy(),
        "y_pred": test_predictions,
    })
    prediction_tables.append(fold_predictions)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        OUTPUTS / "predictions.csv", index=False
    )

    print(f"\nOuter fold {outer_fold} test results:")
    print("Accuracy:", round(accuracy_score(y_test, test_predictions), 4))
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y_test, test_predictions), 4),
    )
    print(
        "Macro-F1:",
        round(f1_score(y_test, test_predictions, average="macro"), 4),
    )
    print("Confusion matrix: rows = actual, columns = predicted")
    print("Class order: 0 = benign, 1 = malignant")
    print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

print("\nAll five folds finished. Predictions saved to:", OUTPUTS / "predictions.csv")

# Record completion only after all five folds finish successfully.
run_info = {
    "protocol_id": contract["protocol_id"],
    "dataset_sha256": contract["dataset_sha256"],
    "method": "svm_deduplicated",
    "seed": 42,
    "outer_folds": [1, 2, 3, 4, 5],
    "inner_cv": "StratifiedGroupKFold(3,shuffle=True,random_state=42+outer_fold)",
    "scoring": "f1_macro",
    "status": "COMPLETED",
    
    "implementation_note": (
        "Exact duplicate features removed using training data only; "
        "DuplicateFeatureSelector, StandardScaler and SVC pipeline; "
        "patient-grouped nested cross-validation."
    ),

    "python_version": platform.python_version(),
    "package_versions": {
        name: version(name)
        for name in ["pandas", "numpy", "scipy", "scikit-learn"]
    },
}

with (OUTPUTS / "run_info.json").open("w", encoding="utf-8") as file:
    json.dump(run_info, file, indent=2)

print("Run information saved.")

