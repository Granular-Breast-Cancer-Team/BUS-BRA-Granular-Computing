# Shared evaluation protocol — BUSBRA-V1-OUTER5-INNER3-MACROF1

**Frozen from the team's original handoff ZIP.** This repository deliberately follows the same hyperparameter grid, fold definitions, and submission schema as the ZIP, to avoid misleading comparisons.

## Data and labels
- Only the 288 original numeric radiomic features with prefixes `original_` or `wavelet-`.
- `diagnosis` is target: 0 benign, 1 malignant. Exclude every ID, pathology/metadata field and fold column from X.
- Every method starts from 288 possible columns. Do not pre-delete the 42 redundant columns globally or create new ones.

## Outer evaluation
Use the existing `fold5_kfold` column, values 1,2,3,4,5. For each outer fold f, only `fold5_kfold == f` is test; remaining rows train. Assert that **patient_id groups do not overlap** across train/test. Never tune or perform supervised feature selection on the held-out patients.

## Inner model selection
Within each outer-training set, use `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + f)`, grouped by `patient_id`. This includes selector fitting and preprocessing INSIDE each training portion:
- `StandardScaler` fitted on training portion only.
- For fuzzy-rough and GBFRS: feature selector fitted on inner-training only, then transform inner-validation.
- Select parameters by **mean macro-F1**.
- Fixed SVM grid from the original package: **C = [0.1, 1, 10, 100]**, **kernel = linear or RBF**, **class_weight = None or balanced**, and **RBF gamma = scale, 0.01 or 0.1**. Linear SVM does not use gamma. That is **32 candidates** (8 linear + 24 RBF), before tuning selector-specific settings.
- Record the selector's hyperparameters and selection budgets. Selector settings must be tuned using outer training only, without unequal test access.
- Do not use PCA, augmentation, resampling, extra features, global scaling or test-driven threshold tuning.

## Refit and report
After inner tuning, fit the chosen complete pipeline on all current outer-training rows, produce **one prediction** per outer-test image, and save them using the package's submission schema.

Report accuracy, macro-F1, balanced accuracy, malignant sensitivity, benign specificity, selected features (names/count and reduction), and measured runtime, for each outer fold plus mean ± SD. Hardware and package versions must accompany runtime; cross-device speed comparisons are descriptive rather than controlled.

The coordinator computes the final comparison from the **same image IDs and outer folds**. The checker verifies identity and prediction schemas, but **cannot prove** that algorithms and scaling were leakage-free; source review is still mandatory. Do not promise 90%+ accuracy.
