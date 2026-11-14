# Coordinator — Published GBFRS + SVM

Study and reproduce the published **granular-ball fuzzy rough feature selection** method (Xia et al., 2025), using the paper and official released code as references. Verify granular-ball creation, fuzzy dependency, feature search, stopping and label encoding before claiming faithful reproduction. Any adaptation must be explicitly named.

Use exactly the frozen 288 starting radiomic predictors, 5 patient-disjoint outer folds, and training-only selection inside 3-fold grouped inner SVM tuning. Record runtime, chosen features and settings per outer fold and return all outer-test predictions in the agreed format.

Your folder is for your own implementation, not an inherited notebook.

Read ../docs/EVALUATION_PROTOCOL.md and ../docs/SUBMISSION_FORMAT.md.

## Phase 2 implementation status

`selector.py` contains an independent, paper-based implementation. It uses
the published granular-ball definitions and sample-mass weighted dependency,
not the alternative scoring expression in the authors' released research
script. `technical_note.md` records the overlap-resolution choice, distance
parameter convention, reproducibility settings, and unresolved paper/code
ambiguities.

Run the synthetic verification suite from the repository root with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
```

The suite does not read `inputs/BUSBRA_features_ready.csv`, run GBFRS on
BUS-BRA, train SVMs, or execute nested cross-validation.
