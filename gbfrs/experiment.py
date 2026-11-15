"""Controlled outer-fold experiment runner for GBFRS + SVM.

The runner is deliberately serial and checkpointed.  It creates one cache
directory per inner split, so repeated GBFRS transformations can be reused
across SVM candidates in that split without sharing fitted steps across folds.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from joblib import Memory
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid

from .pipeline import make_gbfrs_svm_pipeline, selected_feature_names
from .protocol import (
    PROTOCOL_ID,
    TARGET_COLUMN,
    binary_metrics,
    common_svm_grid,
    frozen_feature_columns,
    inner_cv,
    outer_split,
)


def _candidate_params() -> list[dict[str, Any]]:
    return list(ParameterGrid(common_svm_grid()))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def run_outer_fold(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    selector_params: dict[str, Any] | None = None,
    dataset_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one complete outer fold without touching any other outer fold."""

    if outer_fold not in (1, 2, 3, 4, 5):
        raise ValueError("outer_fold must be one of 1, 2, 3, 4, 5")
    features = frozen_feature_columns(frame)
    train_positions, test_positions = outer_split(frame, outer_fold)
    X = frame[features]
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)
    groups = frame["patient_id"].to_numpy()
    selector_kwargs = dict(selector_params or {})
    selector_kwargs.setdefault("random_state", 42)
    selector_kwargs.setdefault("purity_threshold", 0.95)
    selector_kwargs.setdefault("min_samples", 1)
    selector_kwargs.setdefault("n_init", 10)
    selector_kwargs.setdefault("resolve_overlaps", True)
    selector_kwargs.setdefault("improvement_tol", 0.0)

    cache_root = Path(cache_root)
    checkpoint_path = Path(checkpoint_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    candidate_params = _candidate_params()
    checkpoint_config = {
        "protocol_id": PROTOCOL_ID,
        "outer_fold": outer_fold,
        "dataset_sha256": dataset_sha256,
        "n_rows": int(len(frame)),
        "feature_names": features,
        "selector_params": selector_kwargs,
    }
    if checkpoint_path.exists():
        existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if existing.get("checkpoint_config") not in (None, checkpoint_config):
            raise ValueError("checkpoint configuration does not match this run")
        if existing.get("status") == "completed":
            return existing
        if existing.get("status") == "inner_progress":
            inner_records = list(existing.get("inner_records", []))
        else:
            inner_records = []
    else:
        inner_records = []
    inner_splits = list(inner_cv(outer_fold).split(
        X.iloc[train_positions], y[train_positions], groups[train_positions]
    ))
    if len(inner_records) > len(inner_splits):
        raise ValueError("checkpoint contains more inner splits than this protocol")
    selector_seconds = sum(
        float(record.get("selector_seconds") or 0.0) for record in inner_records
    )
    tuning_started = time.perf_counter()

    for split_number, (inner_train_relative, inner_valid_relative) in enumerate(
        inner_splits, start=1
    ):
        if split_number <= len(inner_records):
            continue
        inner_train = train_positions[inner_train_relative]
        inner_valid = train_positions[inner_valid_relative]
        cache = Memory(cache_root / f"inner_{split_number}", verbose=0)
        split_scores: list[dict[str, Any]] = []
        first_selector_runtime: float | None = None

        for candidate_number, params in enumerate(candidate_params, start=1):
            pipeline = make_gbfrs_svm_pipeline(
                features,
                selector_params=selector_kwargs,
                memory=cache,
            )
            pipeline.set_params(**params)
            pipeline.fit(X.iloc[inner_train], y[inner_train])
            selector_runtime = getattr(
                pipeline.named_steps["gbfrs"], "fit_runtime_seconds_", None
            )
            if candidate_number == 1 and selector_runtime is not None:
                first_selector_runtime = float(selector_runtime)
            predictions = pipeline.predict(X.iloc[inner_valid])
            split_scores.append(
                {
                    "candidate_number": candidate_number,
                    "params": params,
                    "macro_f1": float(
                        f1_score(
                            y[inner_valid],
                            predictions,
                            average="macro",
                            labels=[0, 1],
                            zero_division=0,
                        )
                    ),
                    "accuracy": float(accuracy_score(y[inner_valid], predictions)),
                }
            )

        selector_seconds += first_selector_runtime or 0.0
        inner_records.append(
            {
                "split": split_number,
                "n_train": int(len(inner_train)),
                "n_validation": int(len(inner_valid)),
                "train_patients": int(np.unique(groups[inner_train]).size),
                "validation_patients": int(np.unique(groups[inner_valid]).size),
                "selector_seconds": first_selector_runtime,
                "scores": split_scores,
            }
        )
        _write_checkpoint(
            checkpoint_path,
            {
                "status": "inner_progress",
                "checkpoint_config": checkpoint_config,
                "outer_fold": outer_fold,
                "completed_inner_splits": split_number,
                "inner_records": inner_records,
            },
        )

    tuning_seconds = time.perf_counter() - tuning_started
    aggregate: list[dict[str, Any]] = []
    for candidate_number, params in enumerate(candidate_params, start=1):
        rows = [
            record["scores"][candidate_number - 1]
            for record in inner_records
        ]
        aggregate.append(
            {
                "candidate_number": candidate_number,
                "params": params,
                "inner_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
                "inner_accuracy": float(np.mean([row["accuracy"] for row in rows])),
                "per_split_macro_f1": [row["macro_f1"] for row in rows],
            }
        )
    best = max(aggregate, key=lambda row: (row["inner_macro_f1"], -row["candidate_number"]))

    refit_cache = Memory(cache_root / "outer_refit", verbose=0)
    refit_pipeline = make_gbfrs_svm_pipeline(
        features,
        selector_params=selector_kwargs,
        memory=refit_cache,
    )
    refit_pipeline.set_params(**best["params"])
    refit_started = time.perf_counter()
    refit_pipeline.fit(X.iloc[train_positions], y[train_positions])
    refit_seconds = time.perf_counter() - refit_started
    selector_seconds += float(
        getattr(refit_pipeline.named_steps["gbfrs"], "fit_runtime_seconds_", 0.0)
    )

    predict_started = time.perf_counter()
    predictions = refit_pipeline.predict(X.iloc[test_positions])
    predict_seconds = time.perf_counter() - predict_started
    y_true = y[test_positions]
    selected = selected_feature_names(refit_pipeline).tolist()
    total_seconds = tuning_seconds + refit_seconds + predict_seconds
    result = {
        "status": "completed",
        "checkpoint_config": checkpoint_config,
        "protocol_id": PROTOCOL_ID,
        "outer_fold": outer_fold,
        "train_positions": train_positions.tolist(),
        "test_positions": test_positions.tolist(),
        "n_train": int(len(train_positions)),
        "n_test": int(len(test_positions)),
        "n_train_patients": int(frame.iloc[train_positions].patient_id.nunique()),
        "n_test_patients": int(frame.iloc[test_positions].patient_id.nunique()),
        "best_params": best["params"],
        "inner_macro_f1": best["inner_macro_f1"],
        "inner_accuracy": best["inner_accuracy"],
        "inner_records": inner_records,
        "aggregate_scores": aggregate,
        "selected_features": selected,
        "n_selected_features": len(selected),
        "initial_balls": refit_pipeline.named_steps["gbfrs"].initial_ball_count_,
        "final_balls": len(refit_pipeline.named_steps["gbfrs"].balls_),
        "selector_dependency": refit_pipeline.named_steps["gbfrs"].dependency_,
        "selector_seconds": selector_seconds,
        "tuning_seconds": tuning_seconds,
        "refit_seconds": refit_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": total_seconds,
        "y_true": y_true.tolist(),
        "y_pred": predictions.tolist(),
        "metrics": binary_metrics(y_true, predictions),
        "feature_names": features,
    }
    _write_checkpoint(checkpoint_path, result)
    return result


def write_private_fold_outputs(
    frame: pd.DataFrame,
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    dataset_sha256: str,
) -> Path:
    """Write the completed fold-1 records using the private template schema."""

    if result.get("status") != "completed":
        raise ValueError("refusing to write outputs for an incomplete run")
    output = Path(output_dir)
    filenames = [
        "predictions.csv",
        "selected_features.csv",
        "best_params.csv",
        "timings.csv",
        "run_info.json",
    ]
    if output.exists():
        if any((output / filename).exists() for filename in filenames):
            raise FileExistsError(
                f"refusing to overwrite existing submission files in {output}"
            )
        raise FileExistsError(f"refusing to use existing output directory: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{__import__('os').getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output directory already exists: {temporary}")
    temporary.mkdir(parents=True, exist_ok=False)
    output = temporary
    outer_fold = int(result["outer_fold"])
    test_positions = np.asarray(result["test_positions"], dtype=int)
    test_frame = frame.iloc[test_positions]
    predictions = pd.DataFrame(
        {
            "image_id": test_frame["image_id"].astype(str).to_numpy(),
            "patient_id": test_frame["patient_id"].astype(str).to_numpy(),
            "outer_fold": outer_fold,
            "y_true": test_frame["diagnosis"].astype(int).to_numpy(),
            "y_pred": np.asarray(result["y_pred"], dtype=int),
        }
    )
    if len(predictions) != len(test_frame) or predictions.image_id.duplicated().any():
        raise ValueError("prediction identity validation failed")
    predictions.to_csv(output / "predictions.csv", index=False)

    pd.DataFrame(
        {"outer_fold": outer_fold, "feature_name": result["selected_features"]}
    ).to_csv(output / "selected_features.csv", index=False)
    best = result["best_params"]
    svm = {key.removeprefix("svc_"): value for key, value in best.items()}
    pd.DataFrame(
        {
            "outer_fold": [outer_fold],
            "kernel": [svm.get("_kernel")],
            "C": [svm.get("_C")],
            "gamma": [svm.get("_gamma")],
            "class_weight": [svm.get("_class_weight")],
            "inner_macro_f1": [result["inner_macro_f1"]],
            "inner_accuracy": [result["inner_accuracy"]],
        }
    ).to_csv(output / "best_params.csv", index=False)
    pd.DataFrame(
        {
            "outer_fold": [outer_fold],
            "selector_seconds": [result["selector_seconds"]],
            "svm_tuning_seconds": [result["tuning_seconds"]],
            "refit_seconds": [result["refit_seconds"]],
            "predict_seconds": [result["predict_seconds"]],
            "total_seconds": [result["total_seconds"]],
        }
    ).to_csv(output / "timings.csv", index=False)
    run_info = {
        "protocol_id": PROTOCOL_ID,
        "dataset_sha256": dataset_sha256,
        "method": "gbfrs",
        "seed": 42,
        "outer_folds": [outer_fold],
        "inner_cv": (
            "StratifiedGroupKFold(3,shuffle=True,"
            f"random_state={42 + outer_fold})"
        ),
        "scoring": "f1_macro",
        "status": f"COMPLETED_OUTER_FOLD_{outer_fold}_ONLY",
        "implementation_note": (
            "Paper-based independent GBFRS with direct strict-improvement stopping; "
            f"scikit-learn {sklearn.__version__}, Python {platform.python_version()}, "
            f"platform {platform.platform()}"
        ),
    }
    (output / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    temporary.replace(Path(output_dir))
    return Path(output_dir)


def write_private_fold1_outputs(
    frame: pd.DataFrame,
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    dataset_sha256: str,
) -> Path:
    """Backward-compatible alias for the former fold-1-only writer."""

    return write_private_fold_outputs(
        frame,
        result,
        output_dir,
        dataset_sha256=dataset_sha256,
    )
