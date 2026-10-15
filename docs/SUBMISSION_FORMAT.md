# Output format — MUST MATCH the distributed BUSBRA_Team_Handoff.zip

All teams return a folder called `submissions/<method>/` with `<method>` equal to `baseline`, `fuzzy_rough`, or `gbfrs`. Copy the **header-only templates** from the original ZIP, rather than creating your own schema.

- `predictions.csv`: columns `image_id,patient_id,outer_fold,y_true,y_pred`; **1,875 rows**, exact held-out predictions, one per image.
- `selected_features.csv`: columns `outer_fold,feature_name`; selected *existing 288* feature names per outer fold (baseline returns all 288).
- `best_params.csv`: use the supplied template schema; include inner-selected SVM settings and selector settings per outer fold.
- `timings.csv`: use the supplied template schema; **real measured** running times for each fold.
- `run_info.json`: use the supplied template, including correct `protocol_id = BUSBRA-V1-OUTER5-INNER3-MACROF1`, `dataset_sha256`, method, hardware/software versions, random seeds and run status.
- Runnable source code, reproducibility command, paper/code citation, and explanation of train/validation boundaries.

The comparison tool is already in the team's offline ZIP at `tools/compare_results.py`. Execute from the ZIP root:
```bash
python -m pip install -r tools/comparison_requirements.txt
python tools/compare_results.py
```

It checks exact row IDs, assigned folds, truth labels, allowed feature names and dataset hash, and writes comparison tables. **The script cannot verify model code correctness or no-leakage on its own.** Do not enter fabricated values.
