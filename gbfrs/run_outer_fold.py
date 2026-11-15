"""Terminal entry point for one resumable BUS-BRA outer fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .experiment import run_outer_fold, write_private_fold_outputs
from .protocol import frozen_feature_columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-fold", type=int, required=True, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--dataset", type=Path, default=Path("inputs/BUSBRA_features_ready.csv"))
    parser.add_argument("--contract", type=Path, default=Path("evaluation/evaluation_contract.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if digest != contract["dataset_sha256"]:
        raise SystemExit("dataset SHA-256 does not match evaluation_contract.json")
    frame = pd.read_csv(args.dataset)
    if len(frame) != 1875 or len(frozen_feature_columns(frame)) != 288:
        raise SystemExit("frozen dataset shape or predictor count is invalid")
    if args.output_dir.exists():
        final_files = [
            args.output_dir / filename
            for filename in (
                "predictions.csv",
                "selected_features.csv",
                "best_params.csv",
                "timings.csv",
                "run_info.json",
            )
        ]
        if any(path.exists() for path in final_files):
            print(f"Completed output exists; refusing to rerun or overwrite: {args.output_dir}")
            return 0
        raise SystemExit(f"output directory exists but is not a complete result: {args.output_dir}")

    result = run_outer_fold(
        frame,
        outer_fold=args.outer_fold,
        cache_root=args.cache_dir,
        checkpoint_path=args.checkpoint,
        dataset_sha256=digest,
        selector_params={
            "purity_threshold": 0.95,
            "min_samples": 1,
            "random_state": 42,
            "n_init": 10,
            "resolve_overlaps": True,
            "improvement_tol": 0.0,
        },
    )
    output = write_private_fold_outputs(
        frame,
        result,
        args.output_dir,
        dataset_sha256=digest,
    )
    print(json.dumps({
        "status": result["status"],
        "outer_fold": args.outer_fold,
        "output_dir": str(output),
        "selected_features": result["n_selected_features"],
        "total_seconds": result["total_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
