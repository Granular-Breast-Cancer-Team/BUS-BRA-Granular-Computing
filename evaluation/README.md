# Shared comparison

The exact machine-readable specification in `evaluation_contract.json` is copied from the previous team ZIP. Its primary model-selection metric is macro-F1. It mandates the same five patient-disjoint test folds, three grouped inner folds, 288 original features, and common SVM tuning grid.

The **working results checker** is currently included in the distributed `BUSBRA_Team_Handoff.zip` at `tools/compare_results.py`. Keep using that file for the final private comparison to avoid incompatible submission schemas. Nothing here claims model accuracy.

If the coordinator later decides to publish the checker, copy the same audited script rather than recreating a different contract.
