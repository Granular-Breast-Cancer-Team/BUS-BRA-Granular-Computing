# BUS-BRA Granular Computing

Three support vector machine (SVM) pipelines for classifying benign and malignant breast ultrasound findings from precomputed BUS-BRA radiomic features. The comparison covers **1,875 images from 1,064 patients** across five patient-disjoint test folds.

[Results](#results) · [Quickstart](#quickstart) · [Reproduce the experiments](#reproduce-the-experiments) · [Evaluation design](#evaluation-design)

## Results

Conventional fuzzy-rough selection retained **7.6 of 288 features** on average, a **97.36% reduction**, and came within **0.33 percentage points** of the all-feature baseline on mean macro-F1.

![Held-out macro-F1 and malignant recall by method with fold standard deviations, alongside mean feature counts](evaluation/final_comparison/comparison.png)

Scores and feature counts are means across the five outer test folds.

| Method | Accuracy | Macro-F1 | Malignant recall | Features retained |
| --- | ---: | ---: | ---: | ---: |
| All-feature SVM | 87.14% | 85.34% | 80.63% | 288 |
| Conventional fuzzy-rough + SVM | 86.73% | 85.01% | 81.96% | 7.6 |
| GBFRS + SVM | 86.25% | 84.36% | 79.65% | 23.4 |

The full comparison includes fold variability, pooled metrics, and the conditions under which runtimes were measured. [Final report](evaluation/final_comparison/README.md) · [Summary CSV](evaluation/final_comparison/comparison_summary.csv) · [Per-fold CSV](evaluation/final_comparison/per_fold_metrics.csv)

## Methods

| Approach | Feature selection | Code and results |
| --- | --- | --- |
| All-feature SVM | All 288 original predictors | [Baseline runner](baseline_svm/baseline.py) · [Baseline results](baseline_svm/README.md#original-baseline-results) |
| Conventional fuzzy-rough + SVM | Lower-approximation QuickReduct | [Method and runner](fuzzy_rough/README.md) · [Five-fold results](fuzzy_rough/RESULTS.md) |
| GBFRS + SVM | Granular-ball fuzzy-rough selection | [Method and runner](gbfrs/README.md) · [Five-fold results](gbfrs/PHASE5_FINAL_RESULTS.md) |

## Quickstart

Run the synthetic tests from the default branch with Python 3.11. The tests use generated fixtures and run without the frozen dataset.

~~~bash
git clone https://github.com/Granular-Breast-Cancer-Team/BUS-BRA-Granular-Computing.git
cd BUS-BRA-Granular-Computing
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -p no:cacheprovider -q
~~~

GitHub Actions also runs the synthetic tests. [Workflow runs](https://github.com/Granular-Breast-Cancer-Team/BUS-BRA-Granular-Computing/actions/workflows/tests.yml)

In Windows PowerShell, use `py -3.11 -m venv .venv` to create the environment and `.\.venv\Scripts\Activate.ps1` to activate it.

## Reproduce the experiments

1. Copy `BUSBRA_features_ready.csv` and `allowed_feature_columns.txt` from the team's private handoff into `inputs/`. Verify the CSV fingerprint before training. [Input instructions](inputs/README.md) · [Dataset contract](docs/DATASET.md)
2. Use the shared [pinned environment](requirements.txt) for the paired comparison. The method pages give the commands for the [baseline](baseline_svm/README.md#reproduce-the-baseline), [conventional fuzzy-rough method](fuzzy_rough/README.md#run-from-the-repository-root), and [GBFRS](gbfrs/README.md#reproducibility).
3. Assemble the three submissions in the offline handoff and run its `tools/compare_results.py`. [Submission format](docs/SUBMISSION_FORMAT.md)

The repository publishes source code, aggregate metrics, and figures. The frozen input table, image-level predictions, and run artifacts stay in the private handoff. The project is for research use. [Team contribution workflow](CONTRIBUTING.md)

## Evaluation design

- **Inputs:** 288 numeric radiomic predictors. Diagnosis is the binary target. Patient IDs define groups and are excluded from the predictor matrix.
- **Outer evaluation:** five frozen patient-disjoint folds in `fold5_kfold`, with one held-out prediction per image.
- **Inner tuning:** three patient-grouped splits per outer training fold. The 32-candidate SVM grid is ranked by mean macro-F1.
- **Leakage controls:** scaling and feature selection are fitted within each training partition.

[Evaluation protocol](docs/EVALUATION_PROTOCOL.md) · [Machine-readable contract](evaluation/evaluation_contract.json)

## References

- W. Gómez-Flores, M. J. Gregorio-Calas, and W. Coelho de Albuquerque Pereira. “BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems.” *Medical Physics*, 51(4), 3110–3123, 2024. [Paper](https://doi.org/10.1002/mp.16812) · [Dataset record](https://doi.org/10.5281/zenodo.8231412)
- R. Jensen and Q. Shen. “New Approaches to Fuzzy-Rough Feature Selection.” *IEEE Transactions on Fuzzy Systems*, 17(4), 824–838, 2009. [Paper](https://doi.org/10.1109/TFUZZ.2008.924209)
- X. Lian, S. Xia, B. Sang, G. Wang, and X. Gao. “GBFRS: Robust Fuzzy Rough Sets via Granular Ball Computing.” *IEEE Transactions on Neural Networks and Learning Systems*, 37(10), 5271–5285, 2026. [Paper](https://doi.org/10.1109/TNNLS.2026.3681370) · [Technical note](gbfrs/technical_note.md)

The repository has no declared software license. See the [dataset provenance and reuse terms](docs/DATASET.md) for the BUS-BRA data.
