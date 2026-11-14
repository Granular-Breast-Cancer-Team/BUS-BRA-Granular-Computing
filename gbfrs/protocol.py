"""Shared BUS-BRA evaluation protocol utilities.

This module prepares the exact outer/inner grouped splits and common SVM grid.
It does not execute the full experiment by itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
)

from .pipeline import make_gbfrs_svm_pipeline


PROTOCOL_ID = "BUSBRA-V1-OUTER5-INNER3-MACROF1"
OUTER_FOLD_COLUMN = "fold5_kfold"
GROUP_COLUMN = "patient_id"
TARGET_COLUMN = "diagnosis"
RANDOM_SEED = 42


def frozen_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return exactly the original radiomic predictors in frame order."""

    columns = [
        column
        for column in frame.columns
        if column.startswith("original_") or column.startswith("wavelet-")
    ]
    if len(columns) != 288:
        raise ValueError(f"expected exactly 288 frozen predictors, found {len(columns)}")
    return columns


def assert_group_disjoint(
    groups_left: Sequence[Any], groups_right: Sequence[Any]
) -> None:
    overlap = set(groups_left).intersection(set(groups_right))
    if overlap:
        raise AssertionError(f"patient groups overlap: {sorted(overlap)[:5]}")


def outer_split(
    frame: pd.DataFrame, outer_fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return row positions for one patient-disjoint outer split."""

    if outer_fold not in (1, 2, 3, 4, 5):
        raise ValueError("outer_fold must be one of 1, 2, 3, 4, 5")
    test_mask = frame[OUTER_FOLD_COLUMN].to_numpy() == outer_fold
    train_positions = np.flatnonzero(~test_mask)
    test_positions = np.flatnonzero(test_mask)
    assert_group_disjoint(
        frame.iloc[train_positions][GROUP_COLUMN],
        frame.iloc[test_positions][GROUP_COLUMN],
    )
    return train_positions, test_positions


def inner_cv(outer_fold: int) -> StratifiedGroupKFold:
    """Create the prescribed 3-fold grouped inner splitter."""

    if outer_fold not in (1, 2, 3, 4, 5):
        raise ValueError("outer_fold must be one of 1, 2, 3, 4, 5")
    return StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_SEED + outer_fold,
    )


def common_svm_grid() -> list[dict[str, list[Any]]]:
    """Return the 32 valid candidates from the shared SVM grid."""

    grid: list[dict[str, list[Any]]] = []
    for C in (0.1, 1, 10, 100):
        for class_weight in (None, "balanced"):
            grid.append(
                {
                    "svc__C": [C],
                    "svc__kernel": ["linear"],
                    "svc__class_weight": [class_weight],
                }
            )
            for gamma in ("scale", 0.01, 0.1):
                grid.append(
                    {
                        "svc__C": [C],
                        "svc__kernel": ["rbf"],
                        "svc__gamma": [gamma],
                        "svc__class_weight": [class_weight],
                    }
                )
    return grid


def make_inner_search(
    feature_names: Sequence[str],
    outer_fold: int,
    *,
    selector_params: dict[str, Any] | None = None,
    memory: Any = None,
    n_jobs: int | None = 1,
) -> GridSearchCV:
    """Prepare, but do not fit, the prescribed inner SVM search."""

    pipeline = make_gbfrs_svm_pipeline(
        feature_names,
        selector_params=selector_params,
        memory=memory,
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=common_svm_grid(),
        scoring="f1_macro",
        cv=inner_cv(outer_fold),
        refit=True,
        n_jobs=n_jobs,
        return_train_score=False,
    )


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Calculate the protocol's metrics after both classes are represented."""

    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if set(np.unique(true).tolist()) != {0, 1}:
        raise ValueError("metric scoring requires both 0 and 1 in y_true")
    if set(np.unique(pred).tolist()) - {0, 1}:
        raise ValueError("predictions must be encoded as 0 and 1")
    matrix = confusion_matrix(true, pred, labels=[0, 1])
    true_negative, false_positive, false_negative, _ = matrix.ravel()
    malignant_recall = float(true_positive := matrix[1, 1]) / max(
        1, int(matrix[1].sum())
    )
    benign_specificity = float(true_negative) / max(1, int(matrix[0].sum()))
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "malignant_recall": malignant_recall,
        "benign_specificity": benign_specificity,
    }


def run_nested_evaluation(
    frame: pd.DataFrame,
    *,
    selector_params: dict[str, Any] | None = None,
    memory: Any = None,
    n_jobs: int | None = 1,
) -> dict[str, Any]:
    """Run the future five-fold nested protocol when explicitly invoked.

    This function is intentionally not called by Phase 4 setup or tests. Its
    return value is an in-memory research record; private handoff templates
    must be supplied before any export is implemented.
    """

    features = frozen_feature_columns(frame)
    X = frame[features]
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)
    groups = frame[GROUP_COLUMN].to_numpy()
    records: list[dict[str, Any]] = []
    for outer_fold in (1, 2, 3, 4, 5):
        train_positions, test_positions = outer_split(frame, outer_fold)
        search = make_inner_search(
            features,
            outer_fold,
            selector_params=selector_params,
            memory=memory,
            n_jobs=n_jobs,
        )
        search.fit(X.iloc[train_positions], y[train_positions], groups=groups[train_positions])
        fitted = search.best_estimator_
        predictions = fitted.predict(X.iloc[test_positions])
        records.append(
            {
                "outer_fold": outer_fold,
                "test_positions": test_positions,
                "y_true": y[test_positions],
                "y_pred": predictions,
                "metrics": binary_metrics(y[test_positions], predictions),
                "best_params": search.best_params_,
                "selected_features": fitted.named_steps["gbfrs"].get_feature_names_out(),
            }
        )
    return {"protocol_id": PROTOCOL_ID, "folds": records}
