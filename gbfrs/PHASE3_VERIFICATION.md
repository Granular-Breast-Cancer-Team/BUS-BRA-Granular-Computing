# Phase 3 GBFRS mathematical verification and controlled pilot

Date: 2026-10-15

This report covers the independent implementation in `selector.py`. The
published mathematical reference is Xia et al.,
[GBFRS: Robust Fuzzy Rough Sets via Granular-ball Computing](https://arxiv.org/html/2501.18413v1).
No SVM, classifier, accuracy calculation, nested cross-validation, or full
288-feature selection was run.

## 1. Mathematical verification

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 LOKY_MAX_CPU_COUNT=4 python -m pytest -p no:cacheprovider -q
```

Result: **14 passed in 2.22 seconds**.

The tests now verify:

- Equation 9 lower approximations use the minimum distance to balls outside a
  decision class, divided by `sqrt(C)`.
- Equation 13 uses each ball’s sample count divided by the total represented
  sample count as its weight.
- Dependency can exceed 1 under the paper’s un-clipped distance form. A
  hand-checked two-singleton example returns `3.0`.
- With a fixed ball partition and fixed `C`, dependency is monotonic
  non-decreasing as features are added.
- With the selector’s default `C=|B|`, monotonicity is not guaranteed because
  the denominator changes. A two-feature example decreases from `10.0` to
  `10/sqrt(2)`; the implementation does not hide this behavior.
- Greedy forward selection stops after a strict-improvement failure.
- Heterogeneous overlap resolution is deterministic with a fixed seed and
  terminates without duplicating or dropping row memberships.
- The ball membership structure is fixed during forward feature evaluation.
- `transform` does not refit the selector and preserves selected feature
  indices.
- Binary labels must be encoded exactly as `0` and `1`.

The core Equations 9 and 13 behavior is therefore supported by hand-checked
tests. This is not a claim of exact reproduction of the authors’ research
script or their UCI experiments.

## 2. Pilot protocol

The CSV was re-checked immediately before the pilots:

```text
SHA-256: 7f22dccf4be1f8920a0e9fbd46ae31ed814f1be138e55dc936b00b76d67324ef
```

Only rows with `fold5_kfold != 1` were eligible. This outer-training
partition contained 1,499 rows and 852 patients; no fold-1 test patient was
passed to the selector.

The patient subsets were deterministic and target-independent: sort patient
IDs, then use a first-found dynamic-programming subset of whole patient groups
whose row counts sum exactly to the requested sample size. The pilot features
were the first 12 or first 24 existing predictor columns in CSV order. No
correlations, labels, or performance values were used to choose them.

For each pilot, `StandardScaler` was fitted only on that pilot’s rows, and the
selector was run with:

```text
purity_threshold = 0.95
min_samples = 1
random_state = 42
n_init = 10
overlap resolution = enabled
```

## 3. Pilot results

### Pilot A: 64 samples, 12 predictors

```text
patient groups: 37
initial balls: 8
final balls: 41
runtime: 0.120001 s (repeat: 0.086918 s)
peak process RSS: 224,788,480 bytes (~214.4 MiB)
repeated run identical: yes
nonfinite values: none
```

Selected feature path and weighted dependency:

| Step | Feature | Weighted dependency |
|---:|---|---:|
| 1 | `original_firstorder_Energy` | 0.2163396846 |
| 2 | `original_firstorder_Mean` | 0.3038395222 |
| 3 | `original_firstorder_Entropy` | 0.3934287847 |
| 4 | `original_firstorder_Kurtosis` | 0.4468493178 |

The next best candidate was
`original_firstorder_Maximum` with dependency `0.4454857478`, so selection
stopped correctly.

Final selected lower-approximation range: **0.0487719036 to 1.2849447144**.
Across all candidate evaluations: **0.0 to 2.2027372414**.

### Pilot B: 128 samples, 24 predictors

```text
patient groups: 75
initial balls: 11
final balls: 81
runtime: 0.365313 s (repeat: 0.476795 s)
peak process RSS: 229,670,912 bytes (~219.0 MiB)
repeated run identical: yes
nonfinite values: none
```

Selected feature path and weighted dependency:

| Step | Feature | Weighted dependency |
|---:|---|---:|
| 1 | `original_glcm_Autocorrelation` | 0.1524730136 |
| 2 | `original_glcm_Contrast` | 0.2661039484 |
| 3 | `original_glcm_Correlation` | 0.3582454926 |
| 4 | `original_firstorder_Skewness` | 0.4284746506 |
| 5 | `original_firstorder_Minimum` | 0.4700155505 |
| 6 | `original_firstorder_Uniformity` | 0.4893453061 |
| 7 | `original_firstorder_Range` | 0.4988008575 |
| 8 | `original_firstorder_Energy` | 0.5008900939 |
| 9 | `original_glcm_ClusterShade` | 0.5013129063 |
| 10 | `original_firstorder_10Percentile` | 0.5019679344 |

The next best candidate was
`original_firstorder_InterquartileRange` with dependency `0.5018982050`, so
selection stopped correctly.

Final selected lower-approximation range: **0.1285274393 to 1.7195985861**.
Across all candidate evaluations: **0.0 to 4.2936956889**.

The lower-approximation values above 1 are mathematically expected under the
published distance form. Neither lower approximations nor dependency values
were clipped.

## 4. Remaining issues and computational assessment

- The paper does not operationally define every detail of heterogeneous-ball
  overlap removal. The implementation uses deterministic recursive splitting;
  unsplittable singleton overlaps are retained rather than losing samples.
- The paper’s fixed-`C` monotonicity argument and the selector’s default
  cardinality-dependent `C=|B|` are distinct. The report’s tests make that
  distinction explicit.
- Observed final ball counts grew from 41 to 81 when both sample and feature
  dimensions doubled. Runtime grew from 0.12 s to 0.37 s.
- Peak RSS was approximately 214–219 MiB, including the Python and
  scikit-learn process baseline; it is not an isolated GBFRS allocation.
- Feature evaluation repeatedly forms pairwise ball-distance matrices. The
  dominant cost is approximately quadratic in the number of final balls per
  candidate and can become expensive across hundreds of features.

The M4 Mac should have sufficient memory for the raw BUS-BRA matrix and a
single ball-distance matrix, but the present implementation should not yet be
assumed to make a comfortable full 1,499-row × 288-feature run. Depending on
how many balls survive the purity threshold and how many features are accepted,
the repeated candidate scoring could range from minutes to hours. An
intermediate 256/512-row diagnostic or a profiling pass should precede any
full selector run.

## 5. Recommended next step

After approval, integrate `StandardScaler` and `GBFRSSelector` into a
training-only scikit-learn pipeline, validate the 3-fold grouped inner-CV
boundaries on a small protocol fixture, and only then consider a controlled
full-feature GBFRS pilot. SVM integration should remain separate until that
selector pilot is accepted.
