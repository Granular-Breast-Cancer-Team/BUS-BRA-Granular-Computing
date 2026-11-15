# GBFRS Phase 5 — Outer Fold 1 Results

> **Historical milestone:** This report records the state at that phase. The complete five-fold evaluation is documented in [PHASE5_FINAL_RESULTS.md](PHASE5_FINAL_RESULTS.md).

Status: **outer fold 1 complete; folds 2–5 intentionally not run**.

This is one locked-protocol outer-fold result. It is not a five-fold estimate,
and it must not be used to claim general performance or compare methods.

## 1. Environment and preflight

- Branch: `feature/gbfrs`.
- Dataset: `inputs/BUSBRA_features_ready.csv`.
- SHA-256: `7f22dccf4be1f8920a0e9fbd46ae31ed814f1be138e55dc936b00b76d67324ef`.
- Shape: 1,875 rows, 1,064 patients, 288 frozen numerical predictors.
- Labels: `{0: benign, 1: malignant}`.
- Architecture: Apple Silicon `arm64` Mac.
- Python: 3.11.15 in `.venv-busbra`.
- NumPy 1.26.4, pandas 2.2.3, SciPy 1.14.1, scikit-learn 1.5.2,
  pytest 8.3.5.
- Complete test suite after the run: **32 passed**, with one expected
  duplicate-point KMeans `ConvergenceWarning` regression-test warning.

The feature matrix was constructed only from the 288 columns beginning with
`original_` or `wavelet-`. IDs, patient metadata, labels, pathology fields,
and fold columns were excluded.

The exact private handoff templates were validated read-only. Their headers
match the established submission format. The frozen CSV and evaluation
contract were not modified.

## 2. Full-feature feasibility check

Before nested tuning, one GBFRS-only fit used 512 rows from 278 whole patient
groups selected deterministically from outer-fold-1 training patients. The
first 512 eligible rows were not used; patient groups were sorted and included
greedily only when the complete group fit the target. No outer-fold-1 test
patient was included.

Settings were the documented values:

```text
purity_threshold=0.95
min_samples=1
random_state=42
n_init=10
resolve_overlaps=True
improvement_tol=0.0
```

| Quantity | Result |
|---|---:|
| Samples | 512 |
| Patient groups | 278 |
| Starting features | 288 |
| Initial balls | 22 |
| Final balls | 325 |
| Selected features | 21 |
| Final weighted dependency | 0.678268 |
| Lower-approximation range | 0.337514–2.726597 |
| All numerical values finite | Yes |
| Runtime | 21.400 s |
| Peak RSS | 229,195,776 bytes |
| Numerical warnings/failures | None observed |

The direct stopping progression was:

| Step | Feature | Dependency | Direct change | Eq. 19 diagnostic | Result |
|---:|---|---:|---:|---:|---|
| 1 | `original_glrlm_RunVariance` | 0.081892 | — | — | accept |
| 2 | `original_shape2D_MinorAxisLength` | 0.136956 | +0.055064 | +0.079050 | accept |
| 3 | `original_shape2D_Sphericity` | 0.240552 | +0.103596 | +0.128728 | accept |
| 4 | `original_shape2D_Elongation` | 0.329023 | +0.088470 | +0.120698 | accept |
| 5 | `wavelet-H_firstorder_Range` | 0.409690 | +0.080667 | +0.115403 | accept |
| 6 | `original_glcm_Imc1` | 0.476751 | +0.067061 | +0.102757 | accept |
| 7 | `wavelet-L_firstorder_Minimum` | 0.529543 | +0.052792 | +0.088157 | accept |
| 8 | `original_glcm_MaximumProbability` | 0.569280 | +0.039738 | +0.073938 | accept |
| 9 | `wavelet-L_gldm_SmallDependenceLowGrayLevelEmphasis` | 0.592027 | +0.022746 | +0.055304 | accept |
| 10 | `original_glcm_MCC` | 0.611601 | +0.019575 | +0.049955 | accept |
| 11 | `original_ngtdm_Coarseness` | 0.626186 | +0.014585 | +0.043047 | accept |
| 12 | `wavelet-H_glcm_MCC` | 0.638714 | +0.012528 | +0.039186 | accept |
| 13 | `original_gldm_SmallDependenceLowGrayLevelEmphasis` | 0.650074 | +0.011360 | +0.036417 | accept |
| 14 | `wavelet-H_firstorder_Skewness` | 0.657358 | +0.007284 | +0.030931 | accept |
| 15 | `wavelet-L_glcm_Imc1` | 0.662752 | +0.005394 | +0.027684 | accept |
| 16 | `wavelet-H_gldm_DependenceVariance` | 0.666892 | +0.004140 | +0.025185 | accept |
| 17 | `wavelet-H_firstorder_Median` | 0.671144 | +0.004252 | +0.024164 | accept |
| 18 | `original_ngtdm_Contrast` | 0.675470 | +0.004326 | +0.023235 | accept |
| 19 | `wavelet-L_glcm_ClusterShade` | 0.676372 | +0.000903 | +0.018918 | accept |
| 20 | `original_gldm_LargeDependenceHighGrayLevelEmphasis` | 0.676881 | +0.000509 | +0.017635 | accept |
| 21 | `wavelet-L_glrlm_GrayLevelNonUniformity` | 0.678268 | +0.001387 | +0.017700 | accept |
| stop | `wavelet-L_glrlm_ShortRunLowGrayLevelEmphasis` | 0.677494 | −0.000775 | +0.014820 | reject |

The negative direct change at the stopping candidate confirms that the
approved direct strict-improvement rule was applied. Equation 19 remained
diagnostic only.

## 3. Outer-fold-1 protocol

- Outer test: `fold5_kfold == 1`.
- Outer training: all rows with `fold5_kfold != 1`.
- Training/test rows: 1,499 / 376.
- Training/test patients: 852 / 212.
- Patient overlap: none.
- Inner CV: `StratifiedGroupKFold(n_splits=3, shuffle=True,
  random_state=43)` grouped by `patient_id`.
- Tuning metric: macro-F1.
- Pipeline: training-only `StandardScaler → GBFRSSelector → SVC`.
- SVM candidates: all 32 shared candidates.
- GBFRS selector settings: the documented fixed settings above.

Each inner split used its own cache directory. Cached transformations were
shared only between SVM candidates using the same inner-training rows and
selector settings. The outer refit used a separate cache and was fitted on
all outer-training rows only.

## 4. Inner-CV tuning results

The best candidate in each split was:

| Split | Train rows / patients | Validation rows / patients | Best candidate | Macro-F1 | Accuracy |
|---:|---:|---:|---|---:|---:|
| 1 | 999 / 568 | 500 / 284 | linear, C=0.1, balanced | 0.807265 | 0.828000 |
| 2 | 1,000 / 569 | 499 / 283 | RBF, C=10, gamma=0.01, balanced | 0.839964 | 0.861723 |
| 3 | 999 / 567 | 500 / 285 | linear, C=0.1, balanced | 0.863887 | 0.878000 |

The aggregate winner was linear SVM with `C=0.1` and
`class_weight="balanced"`:

| Candidate | Mean inner macro-F1 | Mean inner accuracy |
|---|---:|---:|
| linear, C=0.1, balanced | 0.834199 | 0.852568 |
| RBF, C=10, gamma=0.01, balanced | 0.831201 | 0.851241 |
| linear, C=100, balanced | 0.826673 | 0.845233 |
| linear, C=10, balanced | 0.826535 | 0.845234 |
| linear, C=10, unweighted | 0.826404 | 0.853905 |

All 32 candidates were evaluated; only the aggregate inner results selected
the final configuration. No outer-test labels were passed to tuning.

## 5. Outer-fold-1 refit and evaluation

The best pipeline was refit once on all 1,499 outer-training rows and then
predicted once on the untouched 376-row outer test partition.

| Metric | Outer-fold-1 value |
|---|---:|
| Accuracy | 0.864362 |
| Macro-F1 | 0.848692 |
| Balanced accuracy | 0.857009 |
| Malignant sensitivity/recall | 0.836066 |
| Benign specificity | 0.877953 |

These are **single-fold diagnostic results only**, not final five-fold
performance estimates.

Final outer-training refit details:

- Best SVM: linear kernel, `C=0.1`, `class_weight="balanced"`.
- GBFRS selected features: 23.
- Initial balls: 38.
- Final balls: 1,001.
- Selector runtime across the three unique inner selector fits and final
  refit: 288.739 s.
- Inner tuning wall time: 191.317 s.
- Final refit time: 111.550 s.
- Prediction time: 0.0068 s.
- Total measured pipeline time: 302.874 s.
- Peak process RSS: 353,124,352 bytes.

Selected feature names:

```text
wavelet-L_glcm_Imc2
original_shape2D_Sphericity
original_shape2D_Elongation
wavelet-H_firstorder_Maximum
wavelet-L_glrlm_RunEntropy
wavelet-H_glcm_MCC
original_firstorder_Minimum
original_shape2D_PerimeterSurfaceRatio
original_gldm_SmallDependenceLowGrayLevelEmphasis
original_glcm_MCC
wavelet-H_gldm_DependenceVariance
wavelet-H_firstorder_Skewness
wavelet-H_firstorder_Median
wavelet-L_glrlm_GrayLevelNonUniformity
wavelet-L_ngtdm_Contrast
wavelet-L_gldm_SmallDependenceLowGrayLevelEmphasis
original_glcm_ClusterShade
wavelet-H_firstorder_Mean
wavelet-H_glrlm_LongRunHighGrayLevelEmphasis
wavelet-L_ngtdm_Strength
original_glcm_MaximumProbability
wavelet-H_glcm_Imc2
wavelet-H_glrlm_GrayLevelNonUniformityNormalized
```

## 6. Leakage and output verification

The private output directory is:

```text
/private/tmp/gbfrs_phase5_fold1/submissions/gbfrs/
```

It contains the established `predictions.csv`, `selected_features.csv`,
`best_params.csv`, `timings.csv`, and `run_info.json` schemas. The prediction
checks passed:

- 376 predictions for 376 outer-test rows;
- exact image-ID set match;
- no duplicate image IDs;
- patient IDs match the frozen dataset;
- all predictions assigned to outer fold 1;
- selected names are all among the original 288 predictors;
- run metadata contains the correct protocol ID and dataset hash.

Patient-level predictions and cache files remain outside the public repository.
No prediction files were copied into `submissions/` in the repository.

## 7. Bottlenecks and five-fold feasibility

The main cost is GBFRS selection, not prediction. The outer refit produced
1,001 granular balls and consumed 111.550 s; selector work accounted for
288.739 s across the cached inner selectors and final refit. The peak RSS of
353 MB remained well below the 6 GiB watchdog threshold.

The fold completed under the 900-second wall safeguard. On this run, all five
folds would require substantially more than the current 303-second fold time
because each fold repeats the full three-way grouped tuning and refit; a
rough linear planning estimate is 25–30 minutes serially, with fold-to-fold
variation expected. Caching is safe within matching inner splits, but the
selector still has to be fitted for each distinct training partition. Bounded
serial execution is recommended for the remaining folds to control memory.

The initial macOS `RLIMIT_AS` attempt was rejected by the operating system
before computation began. The successful run therefore used a 900-second
wall-clock safeguard and a 6 GiB peak-RSS watchdog instead; neither was
triggered.

## 8. Next step

Outer fold 1 is complete and documented. Outer folds 2–5 have not been run.
Approval is required before continuing the remaining four outer folds.
