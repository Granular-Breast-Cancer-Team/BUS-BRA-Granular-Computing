# GBFRS Phase 3.5 — Memory Optimization and Robustness Verification

> **Historical milestone:** This report records the state at that phase. The complete five-fold evaluation is documented in [PHASE5_FINAL_RESULTS.md](PHASE5_FINAL_RESULTS.md).

Status: **mathematical verification and controlled BUS-BRA pilots complete;
ready for SVM integration pending approval**.

This phase does not train an SVM, run nested cross-validation, or run the
full 288-feature selector.

## 1. Changes made

### Pairwise distances

`gbfrs/selector.py` now computes both types of pairwise Euclidean distances
with `scipy.spatial.distance.cdist`:

- heterogeneous-ball overlap detection;
- projected ball-center distances used by the fuzzy lower approximations.

The old expression was mathematically

```python
np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
```

The new expression computes the same Euclidean distance matrix without
materializing the intermediate `(m, m, d)` broadcast tensor. No decision
labels, feature columns, distance parameter, dependency formula, or feature
selection objective was changed.

### Robustness changes

- Initial KMeans construction now creates balls only for occupied cluster
  assignments. Coincident rows can produce fewer occupied clusters than the
  requested KMeans count; this must not create an empty ball.
- The selector records `ball_classes_` and rejects a final partition that does
  not contain both encoded decision classes `{0, 1}`.
- Fuzzy lower approximations likewise reject a ball partition missing one
  decision class.
- `heterogeneous_overlap_pairs()` exposes remaining cross-class geometric
  overlaps for diagnostics.

These are validation and representation safeguards. They do not add,
engineer, remove, or globally preprocess any input predictor.

## 2. Tests

Command used:

```text
PYTHONDONTWRITEBYTECODE=1 LOKY_MAX_CPU_COUNT=4 \
  /Users/zanouba/Downloads/breast+cancer+wisconsin+diagnostic/.venv-busbra/bin/python \
  -m pytest -p no:cacheprovider -q
```

Result: **23 passed, 1 expected scikit-learn `ConvergenceWarning`** on the
duplicate-point KMeans regression test.

The tests cover:

- old broadcast and new `cdist` distances agreeing to numerical tolerance;
- fuzzy lower approximations agreeing with a hand-computed broadcast
  reference;
- centers, radii, majority labels, and purity;
- recursive splitting, termination, and row-partition preservation;
- positive-radius coincident centers and deterministic overlap resolution;
- zero-radius coincident singleton balls and retained unsplittable overlaps;
- duplicate points not producing empty initial balls;
- explicit rejection when final balls contain only one decision class;
- dependency calculations on manually checked examples;
- dependency values exceeding 1 without silent clipping;
- monotonicity when the distance parameter and ball partition are fixed;
- the fact that the default changing `C=|B|` convention need not be
  monotonic;
- greedy strict-improvement behavior and tie handling;
- fixed-seed reproducibility;
- binary label encoding;
- preservation of fitted support, balls, and selection history during
  `transform`.

The original Phase 3 tests remain passing; the complete suite was rerun after
the optimization, robustness changes, stopping-rule documentation, and both
pilots.

## 3. Mathematical review

### Equations 9 and 13

For each ball and decision class, the implementation computes the minimum
cross-class projected-center distance divided by `sqrt(C)`, then takes the
maximum across decision classes for the positive-domain value. Equation 13 is
implemented as the sample-mass weighted sum of those values divided by the
represented sample count. The `cdist` replacement is only a computational
rewrite of the same Euclidean distance operation.

The implementation does **not** clip the result to `[0, 1]`. With the
paper's distance-form Equation 9, a center distance greater than `sqrt(C)` can
produce a value greater than 1. This is mathematically distinct from a
normalized fuzzy similarity; clipping would change the published formula and
was not added.

With a fixed ball partition and fixed positive `C`, adding coordinates cannot
decrease a Euclidean distance. Therefore each cross-class minimum, each
positive-domain value, and the weighted dependency are non-decreasing. The
selector's default `C=|B|`, however, changes when a feature is added. The
resulting dependency sequence is therefore not guaranteed to be monotonic;
this is demonstrated by a test.

### Edge cases and overlap termination

- Both final ball classes are checked explicitly. A missing class is rejected
  rather than allowing an undefined cross-class minimum.
- Strict overlap uses `distance < radius_i + radius_j`. Thus two coincident
  zero-radius singleton balls are not treated as having positive geometric
  overlap; their fuzzy lower approximation is zero.
- Coincident positive-radius balls are split deterministically when KMeans can
  produce two non-empty children.
- If a heterogeneous overlap contains only unsplittable balls, it is retained
  and can be reported through `heterogeneous_overlap_pairs()`. Rows are never
  dropped or duplicated.
- Fixed random seeds make the KMeans-based construction and overlap handling
  reproducible. The finite splitting loop and `max_overlap_rounds` bound
  protect termination.

## 4. Stopping-rule ambiguity requiring a decision

The current selector uses the operational rule represented by Algorithm 1's
strict-improvement condition: it accepts a candidate only when

```text
W(B ∪ {a}) > W(B).
```

This is a defensible reading of the pseudocode and matches the intended
greedy behavior used in the Phase 3 implementation. It is **not algebraically
equivalent to Equation 19 when `C` changes from `|B|` to `|B|+1`**.

The paper's Theorem 3 / Equation 19 adjustment implies comparing the candidate
against the scaled baseline

```text
sqrt(|B| / (|B| + 1)) * W(B),
```

or, equivalently, evaluating the corresponding adjusted significance. Direct
comparison to `W(B)` is stricter and can reject a feature that would be a
positive improvement under the Equation 19 normalization. The paper's
pseudocode and prose do not make this transition fully operational, and the
released implementation uses a different scoring expression rather than
resolving this ambiguity.

This is an unresolved mathematical distinction, not a test failure. No
stopping rule was silently changed. The project explicitly selected convention
1 for the pilots:

1. retain the current direct strict-improvement interpretation; **selected**;
2. implement the Equation 19/Theorem 3 adjusted comparison and update the
   corresponding tests and documentation; not selected.

The primary reference for this review is Xia et al.,
[arXiv:2501.18413](https://arxiv.org/html/2501.18413v1).

## 5. Distance microbenchmark

This is an isolated synthetic benchmark, not a BUS-BRA feature-selection
run. It used 512 projected centers with 24 coordinates in separate Python
processes and verified equal distance-matrix checksums.

| Calculation | Runtime | Peak RSS | Distance checksum |
|---|---:|---:|---:|
| Broadcast tensor | 0.05565 s | 139,001,856 bytes | 1,799,956.4028913593 |
| `scipy.spatial.distance.cdist` | 0.00300 s | 60,473,344 bytes | 1,799,956.4028913593 |

Peak RSS includes interpreter and library overhead, so this is directional
rather than a complete application profile. It nevertheless confirms the
expected memory benefit of removing the `(m, m, d)` temporary.

## 6. Controlled BUS-BRA pilots

The frozen input checksum was reverified as:

```text
7f22dccf4be1f8920a0e9fbd46ae31ed814f1be138e55dc936b00b76d67324ef
```

`inputs/BUSBRA_features_ready.csv` remains ignored by Git. The outer-fold-1
training partition contained 1,499 rows from 852 patients. Both pilots used
only patients with `fold5_kfold != 1`; the held-out outer-fold-1 patient set
was explicitly disjoint from each pilot.

Patient selection was deterministic and label-blind: sort eligible patient IDs
lexicographically, then greedily include a whole patient group whenever its
row count fits the remaining target. The first 24 frozen numerical predictors
in CSV order were used. `StandardScaler` was fitted separately on each pilot
subset only. No classifier, accuracy, or cross-validation was run.

| Pilot | Patient groups | Samples | Starting features | Initial balls | Final balls | Selected | Runtime / repeat | Peak process RSS | Unresolved overlaps | Reproducible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 256×24 | 145 | 256 | 24 | 16 | 135 | 11 | 0.805 / 0.627 s | 208,158,720 B | 0 | Yes |
| 512×24 | 278 | 512 | 24 | 22 | 325 | 12 | 1.466 / 1.357 s | 213,696,512 B | 0 | Yes |

Peak RSS includes the process imports and both same-seed fits. It is a
diagnostic upper bound rather than an isolated selector allocation. Both final
partitions contained ball labels `{0, 1}`. All dependencies and lower
approximations were finite.

### 256×24 selected features and progression

Selected features, in order:

1. `original_firstorder_Uniformity`
2. `original_firstorder_Minimum`
3. `original_glcm_Contrast`
4. `original_firstorder_Skewness`
5. `original_glcm_Correlation`
6. `original_glcm_Autocorrelation`
7. `original_firstorder_Range`
8. `original_firstorder_Energy`
9. `original_firstorder_InterquartileRange`
10. `original_firstorder_10Percentile`
11. `original_glcm_ClusterShade`

The best-candidate progression was:

| Step | Best candidate | Dependency | Direct delta | Eq. 19 significance | Accepted |
|---:|---|---:|---:|---:|---|
| 1 | `original_firstorder_Uniformity` | 0.084289 | — | — | yes |
| 2 | `original_firstorder_Minimum` | 0.228061 | +0.143772 | +0.168459 | yes |
| 3 | `original_glcm_Contrast` | 0.344939 | +0.116878 | +0.158728 | yes |
| 4 | `original_firstorder_Skewness` | 0.403339 | +0.058400 | +0.104613 | yes |
| 5 | `original_glcm_Correlation` | 0.433660 | +0.030321 | +0.072902 | yes |
| 6 | `original_glcm_Autocorrelation` | 0.468519 | +0.034859 | +0.072644 | yes |
| 7 | `original_firstorder_Range` | 0.483805 | +0.015287 | +0.050041 | yes |
| 8 | `original_firstorder_Energy` | 0.495590 | +0.011785 | +0.043032 | yes |
| 9 | `original_firstorder_InterquartileRange` | 0.498252 | +0.002661 | +0.031005 | yes |
| 10 | `original_firstorder_10Percentile` | 0.498696 | +0.000445 | +0.026013 | yes |
| 11 | `original_glcm_ClusterShade` | 0.499398 | +0.000702 | +0.023910 | yes |
| stop | `original_firstorder_Entropy` | 0.498853 | −0.000545 | +0.020716 | no |

The final lower-approximation range was **0.153907–1.700490**, with all
values finite.

### 512×24 selected features and progression

Selected features, in order:

1. `original_firstorder_Kurtosis`
2. `original_glcm_Correlation`
3. `original_glcm_Contrast`
4. `original_firstorder_Minimum`
5. `original_firstorder_Energy`
6. `original_glcm_Autocorrelation`
7. `original_firstorder_Range`
8. `original_firstorder_Uniformity`
9. `original_glcm_ClusterShade`
10. `original_firstorder_10Percentile`
11. `original_firstorder_Skewness`
12. `original_firstorder_InterquartileRange`

The best-candidate progression was:

| Step | Best candidate | Dependency | Direct delta | Eq. 19 significance | Accepted |
|---:|---|---:|---:|---:|---|
| 1 | `original_firstorder_Kurtosis` | 0.040050 | — | — | yes |
| 2 | `original_glcm_Correlation` | 0.117048 | +0.076998 | +0.088729 | yes |
| 3 | `original_glcm_Contrast` | 0.208643 | +0.091595 | +0.113074 | yes |
| 4 | `original_firstorder_Minimum` | 0.285800 | +0.077156 | +0.105109 | yes |
| 5 | `original_firstorder_Energy` | 0.325608 | +0.039808 | +0.069981 | yes |
| 6 | `original_glcm_Autocorrelation` | 0.350304 | +0.024696 | +0.053066 | yes |
| 7 | `original_firstorder_Range` | 0.364166 | +0.013862 | +0.039847 | yes |
| 8 | `original_firstorder_Uniformity` | 0.376716 | +0.012550 | +0.036070 | yes |
| 9 | `original_glcm_ClusterShade` | 0.386220 | +0.009503 | +0.031048 | yes |
| 10 | `original_firstorder_10Percentile` | 0.388374 | +0.002154 | +0.021973 | yes |
| 11 | `original_firstorder_Skewness` | 0.390858 | +0.002484 | +0.020558 | yes |
| 12 | `original_firstorder_InterquartileRange` | 0.391358 | +0.000500 | +0.017139 | yes |
| stop | `original_firstorder_TotalEnergy` | 0.387998 | −0.003360 | +0.011994 | no |

The final lower-approximation range was **0.100103–1.914795**, with all
values finite.

## 7. Readiness and next step

The optimized selector is **ready for SVM integration from an implementation
and pilot-readiness perspective**: its mathematical tests pass, pairwise
distance memory use is improved, the direct stopping rule is explicit, and
both controlled pilots were deterministic and training-only. The known
paper/code differences and the diagnostic-only Equation 19 value remain
documented in this report and `technical_note.md`.

No SVM training or full 288-feature selection has been performed. The next
step requires approval to place this selector inside the protocol's leakage-
safe SVM pipelines and inner grouped cross-validation.

With the stopping convention approved, the next safe steps are:

1. preserve the approved direct strict-improvement convention;
2. build the selector into the protocol's leakage-safe SVM pipelines;
3. request separate approval before any full nested cross-validation or
   288-feature experiment.
