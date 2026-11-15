# GBFRS Phase 4 — Pipeline Integration and Validation

> **Historical milestone:** This report records the state at that phase. The complete five-fold evaluation is documented in [PHASE5_FINAL_RESULTS.md](PHASE5_FINAL_RESULTS.md).

Status: **integration and smoke validation complete; full experiment not run**.

No full 288-feature GBFRS run, nested cross-validation, final accuracy
calculation, commit, or push was performed.

## 1. Files created or modified

- `gbfrs/pipeline.py` — name-preserving `StandardScaler`, the
  `StandardScaler -> GBFRSSelector -> SVC` factory, and selected-name helper.
- `gbfrs/protocol.py` — frozen predictor validation, patient-disjoint outer
  split, grouped inner split, exact 32-candidate SVM grid, metric helper, and
  a configurable future nested runner.
- `gbfrs/submission.py` — read-only validation of the private handoff template
  headers and required metadata keys.
- `gbfrs/__init__.py` — public exports for the integration helpers.
- `tests/test_pipeline.py` — synthetic pipeline, protocol, reproducibility,
  class-balance, and leakage-prevention tests.
- `gbfrs/PHASE4_INTEGRATION.md` — this report.

`gbfrs/selector.py` was not changed mathematically. It still uses the
optimized `scipy.spatial.distance.cdist` implementation and the approved
direct strict-improvement stopping rule.

## 2. Pipeline structure

```text
NamedStandardScaler
        |
        v
GBFRSSelector
        |
        v
SVC(kernel="linear" or "rbf")
```

`NamedStandardScaler` fits only on the rows supplied to `Pipeline.fit` and
returns a DataFrame carrying the original 288 CSV names. Consequently,
`GBFRSSelector.feature_names_in_` and the selected-feature helper return the
original names rather than generated `x0`, `x1`, ... names.

The pipeline accepts selector and SVM parameter dictionaries, fixed random
seeds, and optional scikit-learn `Pipeline(memory=...)` caching. It does not
fit any step during `transform` or `predict`.

## 3. Evaluation protocol prepared

The implementation follows `BUSBRA-V1-OUTER5-INNER3-MACROF1`:

- all 288 `original_*` and `wavelet-*` predictors are validated in frozen CSV
  order;
- outer test rows are selected by `fold5_kfold`;
- `outer_split()` asserts patient disjointness using `patient_id`;
- inner CV is `StratifiedGroupKFold(n_splits=3, shuffle=True,
  random_state=42 + outer_fold)`;
- `GridSearchCV` uses `f1_macro`;
- the grid contains exactly 32 valid candidates: four `C` values, two class
  weights, one linear kernel choice, and three RBF gamma choices;
- future scoring supports accuracy, macro-F1, balanced accuracy, malignant
  recall, and benign specificity.

The future runner begins each outer fold with all 288 predictors and returns
an in-memory fold record. It was not invoked in Phase 4.

## 4. Leakage-prevention tests

The complete suite result is:

```text
32 passed, 1 expected scikit-learn ConvergenceWarning
```

The new integration tests verify:

1. outer train/test patient groups are disjoint;
2. inner train/validation patient groups are disjoint;
3. scaler means equal the current training partition only;
4. GBFRS records only the current training-row count;
5. each inner fold receives a distinct fitted selector;
6. held-out rows and labels are never passed to fitting and cannot change
   fitted support;
7. prediction shape and binary encoding are valid;
8. selected names are a subset of the original input names;
9. fixed-seed fits reproduce support and predictions;
10. binary metric scoring and selector fitting require both classes where
    required.

The existing 23 GBFRS mathematical/robustness tests also remain passing.

## 5. Controlled smoke test

The smoke test used the frozen CSV, only `fold5_kfold != 1` patients, the
deterministic whole-patient rule used in Phase 3.5, and the first 24 existing
predictors in CSV order. It used 128 rows from 76 patient groups. The scaler,
GBFRS selector, and linear SVM were fitted sequentially through the pipeline.

| Item | Result |
|---|---:|
| Outer-fold-1 test patients included | No |
| Starting features | 24 |
| Initial granular balls | 11 |
| Final granular balls | 80 |
| Selected features | 8 |
| Fit runtime | 0.2879 s |
| Prediction runtime | 0.0011 s |
| Peak process RSS | 217,300,992 bytes |
| Prediction shape | `(128,)` |
| Prediction labels | valid `{0, 1}` |
| Classification score calculated | No |

Selected smoke-test names were:

```text
original_firstorder_Uniformity
original_firstorder_Minimum
original_glcm_Contrast
original_glcm_Correlation
original_firstorder_Range
original_firstorder_Energy
original_firstorder_Skewness
original_glcm_Autocorrelation
```

These predictions are technical smoke-test outputs only and are not research
results.

## 6. Submission-template compatibility

The exact private package was found at:

```text
/Users/zanouba/Desktop/BUSBRA_Team_Handoff.zip
```

Its `submissions/gbfrs/` templates were inspected and validated read-only.
The exact required headers are:

- `predictions.csv`: `image_id,patient_id,outer_fold,y_true,y_pred`;
- `selected_features.csv`: `outer_fold,feature_name`;
- `best_params.csv`: `outer_fold,kernel,C,gamma,class_weight,inner_macro_f1,inner_accuracy`;
- `timings.csv`: `outer_fold,selector_seconds,svm_tuning_seconds,refit_seconds,predict_seconds,total_seconds`;
- `run_info.json`: protocol, dataset hash, method, seed, fold, CV, scoring,
  status, and implementation-note fields.

`gbfrs/submission.py` validates those exact headers. No template files were
copied into the public repository and no prediction output was created. The
private package remains marked `TEMPLATE_ONLY_NOT_TRAINED`.

## 7. Computational planning

The locked protocol has five outer folds, three inner folds, and 32 SVM
candidates: **480 candidate SVM fits**, plus one final outer-training refit
per fold. If the pipeline is used without caching, each candidate fit also
repeats scaling and GBFRS fitting, creating up to roughly 485 selector fits.

Repeated selector fitting is safe to reuse across SVM candidates when all of
the following are identical: the inner-training rows, training labels,
feature names/order, selector parameters, scaler parameters, and random seed.
It is not safe to reuse a fitted selector across different inner or outer
folds.

The prepared pipeline supports scikit-learn `Memory` caching. A future run
should use a fresh cache directory tied to the dataset hash, protocol ID, and
run configuration. Pipeline cache keys include the training data and
transformer parameters, so validation/test rows are transformed by a cached
training-fitted step rather than included in its fit. With correct caching,
the expected GBFRS fit count is approximately 15 inner-fold fits plus 5
outer-training refits, while SVC still evaluates all 480 candidates.

The Phase 3.5 measurements provide only a directional estimate:

- 256×24 selector: 0.805 s;
- 512×24 selector: 1.466 s;
- 128×24 pipeline smoke fit: 0.288 s.

The final experiment has 1,875 rows and 288 predictors. GBFRS cost can grow
faster than linearly because the number of granular balls and the number of
feature-subset evaluations both grow; the `m²` ball-distance component is the
main memory/runtime risk. The full run should therefore begin with measured
per-fold timing and RSS, use bounded parallelism, and stop for review if ball
counts or memory grow unexpectedly. Caching reduces repeated selector work but
does not make the full experiment risk-free.

## 8. Remaining issues before full model training

1. Obtain approval before invoking the future nested runner.
2. Keep the frozen CSV checksum and 288-column assertion as preflight gates.
3. Use the exact private template directory only for a real, reviewed run;
   do not fabricate rows or metadata.
4. Record actual selector, tuning, refit, prediction, and total timings per
   outer fold.
5. Review memory and granular-ball counts before allowing all five folds to
   continue.
6. Report classification metrics only after the approved nested experiment;
   the smoke test above intentionally reports none.
