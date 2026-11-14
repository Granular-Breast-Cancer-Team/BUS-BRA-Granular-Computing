# GBFRS + SVM final five-fold results

## Scope and status

This report summarizes the completed GBFRS + SVM run across all five patient-disjoint outer folds. No model was retrained while preparing these deliverables. The original per-fold outputs under `outputs/gbfrs/fold1/` through `fold5/` were read without modification.

The results are GBFRS-only results under the locked protocol. They do not establish that GBFRS outperformed the baseline or conventional fuzzy-rough method. The private three-way comparison remains pending until those two submissions are available.

## Dataset and protocol

- Dataset: `inputs/BUSBRA_features_ready.csv`
- SHA-256: `7f22dccf4be1f8920a0e9fbd46ae31ed814f1be138e55dc936b00b76d67324ef`
- Records: 1,875 images from 1,064 patients
- Predictors: the frozen 288 original `original_*` and `wavelet-*` columns only
- Labels: `0 = benign`, `1 = malignant`
- Outer evaluation: supplied patient-disjoint `fold5_kfold` partitions
- Inner tuning: three-fold `StratifiedGroupKFold`, grouped by `patient_id`, with random state `42 + outer_fold`
- Primary tuning metric: macro-F1
- Pipeline: training-fold `StandardScaler` → GBFRS → SVC
- SVM grid: the shared 32-candidate grid over `C`, kernel, class weight, and RBF gamma
- GBFRS: independent paper-based implementation, optimized `scipy.spatial.distance.cdist` calculations, and the documented direct strict-improvement stopping rule

The final preflight checks found no duplicate image IDs, no patient crossing outer folds, and no mismatch between submitted and source patient IDs, fold assignments, or ground-truth labels.

## Per-fold results

| Outer fold | Test images | Selected features | Accuracy | Macro-F1 | Balanced accuracy | Malignant recall | Benign specificity | Selected SVM |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 376 | 23 | 0.8644 | 0.8487 | 0.8570 | 0.8361 | 0.8780 | linear, C=0.1, balanced |
| 2 | 385 | 23 | 0.9091 | 0.8961 | 0.8953 | 0.8560 | 0.9346 | RBF, C=1, gamma=0.01, balanced |
| 3 | 366 | 23 | 0.8743 | 0.8562 | 0.8532 | 0.7917 | 0.9146 | linear, C=10, balanced |
| 4 | 365 | 22 | 0.8082 | 0.7799 | 0.7775 | 0.6891 | 0.8659 | RBF, C=1, gamma=0.01, balanced |
| 5 | 383 | 26 | 0.8564 | 0.8370 | 0.8439 | 0.8099 | 0.8779 | RBF, C=1, gamma=0.01, balanced |

### Five-fold summary

| Metric | Mean ± sample SD | Pooled across 1,875 predictions |
|---|---:|---:|
| Accuracy | 0.8625 ± 0.0364 | 0.8629 |
| Macro-F1 | 0.8436 ± 0.0420 | 0.8442 |
| Balanced accuracy | 0.8454 ± 0.0427 | 0.8458 |
| Malignant recall | 0.7965 ± 0.0649 | 0.7974 |
| Benign specificity | 0.8942 ± 0.0291 | 0.8943 |

These are descriptive estimates under this single locked evaluation protocol. They are not a claim of superiority over another method.

## Feature reduction and stability

GBFRS selected 23.4 features on average, with a range of 22–26 from the 288 available predictors. This is a mean reduction of 91.9%, leaving 8.1% of the original feature count.

Across folds, 41 distinct feature names were selected. Twenty-seven were selected in at least two folds, and nine were selected in all five folds:

- `original_gldm_SmallDependenceLowGrayLevelEmphasis`
- `original_shape2D_Elongation`
- `original_shape2D_Sphericity`
- `wavelet-H_firstorder_Mean`
- `wavelet-H_firstorder_Median`
- `wavelet-H_firstorder_Skewness`
- `wavelet-H_glcm_MCC`
- `wavelet-H_gldm_DependenceVariance`
- `wavelet-L_gldm_SmallDependenceLowGrayLevelEmphasis`

Pairwise selected-set Jaccard similarity had mean 0.525, minimum 0.353, and maximum 0.714. These values describe fold-to-fold selection overlap; they should not be interpreted as a general stability guarantee outside this dataset and protocol.

## Runtime

Measured end-to-end outer-fold runtimes were:

| Outer fold | Total seconds |
|---:|---:|
| 1 | 302.87 |
| 2 | 284.03 |
| 3 | 268.64 |
| 4 | 257.80 |
| 5 | 267.37 |

The summed recorded runtime was 1,380.71 seconds, approximately 23.0 minutes. The fold timings include GBFRS selection, inner SVM tuning, final outer-training refit, and prediction. The detailed measured timings are preserved in the private submission bundle.

## Validation and comparison status

Independent validation confirmed:

- Dataset hash, row count, patient count, label encoding, and 288-feature inventory match the contract.
- The combined submission contains exactly 1,875 predictions, 1,875 unique image IDs, and 1,064 unique patient IDs.
- Each image appears in its supplied outer fold exactly once.
- Submitted patient IDs and ground-truth labels match the frozen source.
- All selected names belong to the frozen 288-feature inventory.
- All five fold records have completed status, finite timing values, one best-parameter row, and one timing row.
- The exact private handoff headers and required run-information keys validate successfully.

The original private `tools/compare_results.py` checker was run in an isolated temporary root containing the assembled GBFRS submission. It stopped with a missing `baseline` submission before producing a three-way scoreboard. This is expected when the other methods are not available. GBFRS-alone validation passed; the full baseline/fuzzy-rough/GBFRS comparison is pending.

The complete local test suite also passed: 32 tests passed. The warnings were environment-level core-count detection and an expected duplicate-point clustering warning in a synthetic edge-case test.

## Deliverables

The aggregate-only figures are stored under `gbfrs/figures/`:

- `phase5_fold_metrics.svg`
- `phase5_selected_features.svg`
- `phase5_metric_variability.svg`
- `phase5_feature_stability.svg`

The exact private handoff bundle and validation tables are under the ignored `outputs/gbfrs/final_submission/` directory. The bundle contains only the five required `submissions/gbfrs/` files. It must remain private because `predictions.csv` contains patient-level identifiers.

## Limitations

The report does not compare methods, test statistical significance, or claim external validity. Outer-fold performance is reported only after the complete locked run, but the comparison checker cannot prove preprocessing or selector nesting by itself; those claims depend on the reviewed implementation and protocol tests. The private prediction files contain patient-level metadata and must remain outside public Git history.

## References

1. Xia et al., “Granular-Ball Fuzzy-Rough Feature Selection,” arXiv:2501.18413, 2025.
2. Official author implementation repository: `https://github.com/lianxiaoyu724/GBFRS`.
3. Project evaluation contract: `evaluation/evaluation_contract.json`.
4. Project protocol and submission schema: `docs/EVALUATION_PROTOCOL.md` and `docs/SUBMISSION_FORMAT.md`.
