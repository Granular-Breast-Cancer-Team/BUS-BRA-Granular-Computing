# Coordinator — Published GBFRS + SVM

Study and reproduce the published **granular-ball fuzzy rough feature selection** method (Xia et al., 2025), using the paper and official released code as references. Verify granular-ball creation, fuzzy dependency, feature search, stopping and label encoding before claiming faithful reproduction. Any adaptation must be explicitly named.

Use exactly the frozen 288 starting radiomic predictors, 5 patient-disjoint outer folds, and training-only selection inside 3-fold grouped inner SVM tuning. Record runtime, chosen features and settings per outer fold and return all outer-test predictions in the agreed format.

Your folder is for your own implementation, not an inherited notebook.

Read ../docs/EVALUATION_PROTOCOL.md and ../docs/SUBMISSION_FORMAT.md.
