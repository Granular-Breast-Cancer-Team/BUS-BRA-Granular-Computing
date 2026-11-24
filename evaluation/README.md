# Evaluation and results

The BUS-BRA experiments share five frozen patient-disjoint outer folds, three patient-grouped inner folds, 288 candidate features, and a 32-candidate SVM search. Mean inner macro-F1 selects the configuration.

- [Final comparison](final_comparison/README.md) reports the paired three-method results, feature reduction, and fold variability.
- [Aggregate summary](final_comparison/comparison_summary.csv) includes fold means, sample standard deviations, and pooled scores.
- [Per-fold metrics](final_comparison/per_fold_metrics.csv) contain aggregate scores, image counts, and feature counts for all 15 method-fold combinations.
- [Evaluation contract](evaluation_contract.json) defines the frozen inputs, group assignments, and search grid.

The team's offline handoff contains `tools/compare_results.py`. Run it from the handoff root with all three completed submissions. The public CSVs contain aggregate results. [Submission format](../docs/SUBMISSION_FORMAT.md)
