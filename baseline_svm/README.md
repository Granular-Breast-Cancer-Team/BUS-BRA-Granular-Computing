# All-feature SVM baseline

The baseline fits a standard-scaled support vector machine to all **288 BUS-BRA radiomic predictors**. It provides the reference for measuring what the feature-selection methods retain and how their held-out scores change. [Final three-method comparison](../evaluation/final_comparison/README.md)

## Evaluation

The frozen input contains 1,875 images from 1,064 patients, with 1,268 benign and 607 malignant images. The baseline uses the five predefined patient-disjoint test folds. Three-fold `StratifiedGroupKFold` tunes 32 SVC candidates within each outer training set using seed `42 + outer_fold`. Each candidate fits `StandardScaler` on its training partition. The winning pipeline is refitted on the complete outer training set, then predicts that fold's held-out images. [Evaluation protocol](../docs/EVALUATION_PROTOCOL.md)

## Original baseline results

These scores come from the original baseline environment. The final team comparison reruns the baseline with the shared scikit-learn version to align the inner patient-group assignments across all three methods.

| Outer fold | Accuracy | Balanced accuracy | Macro-F1 | Malignant recall | Benign specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8883 | 0.8896 | 0.8765 | 0.8934 | 0.8858 |
| 2 | 0.9169 | 0.9073 | 0.9056 | 0.8800 | 0.9346 |
| 3 | 0.8579 | 0.8089 | 0.8274 | 0.6667 | 0.9512 |
| 4 | 0.8110 | 0.7925 | 0.7880 | 0.7395 | 0.8455 |
| 5 | 0.8512 | 0.8445 | 0.8331 | 0.8264 | 0.8626 |
| **Mean** | **0.8650** | **0.8486** | **0.8461** | **0.8012** | **0.8960** |
| Sample SD | 0.0400 | 0.0497 | 0.0457 | 0.0965 | 0.0456 |

All five folds selected a linear kernel with `C=0.1`. Fold 3 used equal class weights. The other four used balanced weights. The recorded fold times summed to approximately 575 seconds on the original machine.

Malignant recall ranged from 66.67% to 89.34%. Sample standard deviations describe fold-to-fold variation. Evaluation was limited to the frozen BUS-BRA cohort.

## Reproduce the baseline

Use Python 3.11 and put the frozen CSV and feature allowlist in `inputs/`. Both are supplied in the team handoff. The original experiment used Python 3.11.9. [Input instructions](../inputs/README.md)

From the repository root in an activated virtual environment:

~~~bash
python -m pip install -r baseline_svm/requirements.txt
python baseline_svm/baseline.py
python baseline_svm/summarize_baseline.py
~~~

To reproduce the baseline used in the paired comparison, install the root `requirements.txt` in a separate environment and run the same scripts. Run each configuration in a separate checkout to preserve its outputs. The runner and summarizer write to `submissions/baseline/` for offline validation. [Shared requirements](../requirements.txt) · [Comparison provenance](../evaluation/final_comparison/README.md#reproduction-and-checks)
