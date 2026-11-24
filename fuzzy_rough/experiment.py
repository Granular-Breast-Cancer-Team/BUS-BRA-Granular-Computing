"""Run the locked five-fold BUS-BRA evaluation and write the offline handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gbfrs.protocol import (
    GROUP_COLUMN,
    OUTER_FOLD_COLUMN,
    PROTOCOL_ID,
    RANDOM_SEED,
    TARGET_COLUMN,
    assert_group_disjoint,
    binary_metrics,
    common_svm_grid,
    frozen_feature_columns,
    inner_cv,
    outer_split,
)
from gbfrs.submission import (
    BEST_PARAMETER_COLUMNS,
    PREDICTION_COLUMNS,
    SELECTED_FEATURE_COLUMNS,
    TIMING_COLUMNS,
)

from .selector import DEPENDENCY_TOLERANCE, PAPER_DOI, Reduct, select_features


METHOD = "fuzzy_rough"
FOLDS = (1, 2, 3, 4, 5)
DATASET_SHA256 = "7f22dccf4be1f8920a0e9fbd46ae31ed814f1be138e55dc936b00b76d67324ef"


@dataclass
class FoldResult:
    predictions: pd.DataFrame
    features: pd.DataFrame
    params: dict
    timings: dict
    audit: dict


def load_frozen_data(path: Path) -> tuple[pd.DataFrame, list[str]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATASET_SHA256:
        raise ValueError(f"Frozen BUS-BRA CSV SHA-256 mismatch: {digest}")
    frame = pd.read_csv(path, dtype={"image_id": str, GROUP_COLUMN: str})
    features = frozen_feature_columns(frame)
    allowlist = path.with_name("allowed_feature_columns.txt")
    if allowlist.is_file():
        allowed = allowlist.read_text(encoding="utf-8").splitlines()
        if allowed != features:
            raise ValueError("Feature allowlist does not match the frozen CSV columns")
    if len(frame) != 1875 or frame[GROUP_COLUMN].nunique() != 1064:
        raise ValueError("Unexpected BUS-BRA row or patient count")
    if frame[["image_id", GROUP_COLUMN]].isna().any().any() or not frame.image_id.is_unique:
        raise ValueError("Missing group IDs or missing/duplicate image IDs")
    if set(frame[OUTER_FOLD_COLUMN]) != set(FOLDS) or set(frame[TARGET_COLUMN]) != {0, 1}:
        raise ValueError("Unexpected BUS-BRA fold or diagnosis values")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Frozen predictor values must be finite")
    if not frame.groupby(GROUP_COLUMN)[OUTER_FOLD_COLUMN].nunique().eq(1).all():
        raise ValueError("A patient occurs in more than one official outer fold")
    return frame, features


def _prepare_training(
    X: np.ndarray, y: np.ndarray
) -> tuple[StandardScaler, Reduct, np.ndarray, float]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    started = perf_counter()
    reduct = select_features(scaled, y)
    selector_seconds = perf_counter() - started
    if not reduct.indices:
        raise ValueError("QuickReduct selected zero features for this training split")
    return scaler, reduct, scaled[:, reduct.indices], selector_seconds


def _svm_candidates() -> list[dict]:
    return [
        {name.removeprefix("svc__"): value for name, value in params.items()}
        for params in ParameterGrid(common_svm_grid())
    ]


def _best_candidate(scores: list[dict]) -> dict:
    return max(
        scores,
        key=lambda row: (
            row["inner_macro_f1"],
            row["inner_accuracy"],
            -row["mean_inner_feature_count"],
            -row["grid_position"],
        ),
    )


def evaluate_fold(frame: pd.DataFrame, features: list[str], fold: int) -> FoldResult:
    started = perf_counter()
    train, test = outer_split(frame, fold)
    X = frame[features].to_numpy(dtype=float)
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)
    groups = frame[GROUP_COLUMN].to_numpy()
    candidates = _svm_candidates()
    split_scores = np.empty((3, len(candidates), 2))
    inner_records = []
    tuning_seconds = 0.0

    for split, (train_relative, valid_relative) in enumerate(
        inner_cv(fold).split(X[train], y[train], groups[train])
    ):
        inner_train, inner_valid = train[train_relative], train[valid_relative]
        assert_group_disjoint(groups[inner_train], groups[inner_valid])
        scaler, reduct, train_reduced, selection_time = _prepare_training(
            X[inner_train], y[inner_train]
        )
        valid_reduced = scaler.transform(X[inner_valid])[:, reduct.indices]
        inner_records.append({
            "split": split + 1,
            "training_images": len(inner_train),
            "validation_images": len(inner_valid),
            "training_patients": int(np.unique(groups[inner_train]).size),
            "validation_patients": int(np.unique(groups[inner_valid]).size),
            "selected_features": [features[index] for index in reduct.indices],
            "dependency_path": list(reduct.dependencies),
            "full_dependency": reduct.full_dependency,
            "stop_reason": reduct.stop_reason,
            "selector_seconds": selection_time,
        })

        # Selection is independent of SVC parameters. Reuse the same reduced
        # training and validation matrices only across this split's candidates.
        tuning_started = perf_counter()
        for candidate, params in enumerate(candidates):
            model = SVC(**params, random_state=RANDOM_SEED).fit(train_reduced, y[inner_train])
            predicted = model.predict(valid_reduced)
            split_scores[split, candidate] = (
                f1_score(y[inner_valid], predicted, labels=[0, 1], average="macro", zero_division=0),
                accuracy_score(y[inner_valid], predicted),
            )
        tuning_seconds += perf_counter() - tuning_started

    mean_scores = split_scores.mean(axis=0)
    feature_count = float(np.mean([len(row["selected_features"]) for row in inner_records]))
    candidate_scores = [
        {
            "grid_position": index + 1,
            "parameters": params,
            "inner_macro_f1": float(mean_scores[index, 0]),
            "inner_accuracy": float(mean_scores[index, 1]),
            "mean_inner_feature_count": feature_count,
            "per_split_macro_f1": split_scores[:, index, 0].tolist(),
            "per_split_accuracy": split_scores[:, index, 1].tolist(),
        }
        for index, params in enumerate(candidates)
    ]
    winner = _best_candidate(candidate_scores)
    params = winner["parameters"]

    refit_started = perf_counter()
    scaler, reduct, train_reduced, outer_selection_time = _prepare_training(X[train], y[train])
    model = SVC(**params, random_state=RANDOM_SEED).fit(train_reduced, y[train])
    refit_seconds = perf_counter() - refit_started

    predict_started = perf_counter()
    predicted = model.predict(scaler.transform(X[test])[:, reduct.indices])
    predict_seconds = perf_counter() - predict_started
    predictions = frame.iloc[test][["image_id", GROUP_COLUMN]].assign(
        outer_fold=fold, y_true=y[test], y_pred=predicted
    )
    selected = [features[index] for index in reduct.indices]
    return FoldResult(
        predictions=predictions,
        features=pd.DataFrame({"outer_fold": fold, "feature_name": selected}),
        params={
            "outer_fold": fold,
            "kernel": params["kernel"],
            "C": params["C"],
            "gamma": params.get("gamma"),
            "class_weight": params["class_weight"],
            "inner_macro_f1": winner["inner_macro_f1"],
            "inner_accuracy": winner["inner_accuracy"],
        },
        timings={
            "outer_fold": fold,
            "selector_seconds": outer_selection_time + sum(
                row["selector_seconds"] for row in inner_records
            ),
            "svm_tuning_seconds": tuning_seconds,
            "refit_seconds": refit_seconds,
            "predict_seconds": predict_seconds,
            "total_seconds": perf_counter() - started,
        },
        audit={
            "outer_fold": fold,
            "test_images": len(test),
            "test_patients": int(np.unique(groups[test]).size),
            "selected_features": selected,
            "dependency_path": list(reduct.dependencies),
            "full_dependency": reduct.full_dependency,
            "stop_reason": reduct.stop_reason,
            "outer_selector_seconds": outer_selection_time,
            "inner_splits": inner_records,
            "candidate_scores": candidate_scores,
            "winning_grid_position": winner["grid_position"],
            "metrics": binary_metrics(y[test], predicted),
        },
    )


def write_submission(output: Path, frame: pd.DataFrame, results: list[FoldResult]) -> None:
    predictions = pd.concat([result.predictions for result in results]).sort_index()
    if not predictions.image_id.equals(frame.image_id):
        raise ValueError("Expected one held-out prediction for each original image")
    info = {
        "protocol_id": PROTOCOL_ID,
        "dataset_sha256": DATASET_SHA256,
        "method": METHOD,
        "seed": RANDOM_SEED,
        "outer_folds": list(FOLDS),
        "inner_cv": "StratifiedGroupKFold(3,shuffle=True,random_state=42+outer_fold)",
        "scoring": "f1_macro",
        "status": "COMPLETE",
        "implementation_note": (
            "Jensen and Shen (2009), Section III-A, Equation 26, Łukasiewicz operators, "
            "sample-standard-deviation similarity and forward QuickReduct."
        ),
        "selector_doi": PAPER_DOI,
        "dependency_tolerance": DEPENDENCY_TOLERANCE,
        "environment": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                package: version(package)
                for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")
            },
        },
        "pooled_metrics": binary_metrics(predictions.y_true, predictions.y_pred),
        "folds": [result.audit for result in results],
    }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as directory:
        staging = Path(directory)
        tables = (
            ("predictions.csv", predictions, PREDICTION_COLUMNS),
            (
                "selected_features.csv",
                pd.concat([result.features for result in results]),
                SELECTED_FEATURE_COLUMNS,
            ),
            ("best_params.csv", pd.DataFrame([result.params for result in results]), BEST_PARAMETER_COLUMNS),
            ("timings.csv", pd.DataFrame([result.timings for result in results]), TIMING_COLUMNS),
        )
        for name, table, columns in tables:
            table.to_csv(staging / name, columns=columns, index=False)
        (staging / "run_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name, _, _ in tables:
            (staging / name).replace(output / name)
        (staging / "run_info.json").replace(output / "run_info.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("inputs/BUSBRA_features_ready.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("submissions/fuzzy_rough"))
    args = parser.parse_args()
    frame, features = load_frozen_data(args.dataset)
    results = []
    for fold in FOLDS:
        result = evaluate_fold(frame, features, fold)
        results.append(result)
        print(f"Fold {fold}/5 complete, {len(result.features)} features selected", flush=True)
    write_submission(args.output_dir, frame, results)
    print(f"Saved {len(frame)} held-out predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
