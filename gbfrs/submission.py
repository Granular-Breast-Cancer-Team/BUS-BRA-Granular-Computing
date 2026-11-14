"""Compatibility checks for the private BUS-BRA handoff templates.

This module validates the supplied schema but is not invoked by Phase 4's
smoke test and does not create a submission.  Full export should happen only
after a real five-fold experiment and review of the private handoff package.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PREDICTION_COLUMNS = ["image_id", "patient_id", "outer_fold", "y_true", "y_pred"]
SELECTED_FEATURE_COLUMNS = ["outer_fold", "feature_name"]
BEST_PARAMETER_COLUMNS = [
    "outer_fold",
    "kernel",
    "C",
    "gamma",
    "class_weight",
    "inner_macro_f1",
    "inner_accuracy",
]
TIMING_COLUMNS = [
    "outer_fold",
    "selector_seconds",
    "svm_tuning_seconds",
    "refit_seconds",
    "predict_seconds",
    "total_seconds",
]
RUN_INFO_REQUIRED_KEYS = [
    "protocol_id",
    "dataset_sha256",
    "method",
    "seed",
    "outer_folds",
    "inner_cv",
    "scoring",
    "status",
    "implementation_note",
]


def validate_submission_templates(template_dir: str | Path) -> dict[str, object]:
    """Validate exact CSV/JSON headers from the supplied handoff package."""

    directory = Path(template_dir)
    expected = {
        "predictions.csv": PREDICTION_COLUMNS,
        "selected_features.csv": SELECTED_FEATURE_COLUMNS,
        "best_params.csv": BEST_PARAMETER_COLUMNS,
        "timings.csv": TIMING_COLUMNS,
    }
    for filename, columns in expected.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = list(pd.read_csv(path, nrows=0).columns)
        if actual != columns:
            raise ValueError(f"{path} has {actual}; expected {columns}")

    run_info_path = directory / "run_info.json"
    if not run_info_path.is_file():
        raise FileNotFoundError(run_info_path)
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    missing = [key for key in RUN_INFO_REQUIRED_KEYS if key not in run_info]
    if missing:
        raise ValueError(f"{run_info_path} is missing keys: {missing}")
    return {
        "directory": str(directory),
        "csv_columns": expected,
        "run_info_keys": RUN_INFO_REQUIRED_KEYS,
        "template_status": run_info.get("status"),
    }
