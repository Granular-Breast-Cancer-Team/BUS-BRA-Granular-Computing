# GBFRS implementation technical note

This implementation follows the mathematical definitions in Xia et al.,
“GBFRS: Robust Fuzzy Rough Sets via Granular-ball Computing” (arXiv:2501.18413,
2025). It does not copy source from the authors’ public repository, which has
no explicit software license and whose archive was inspected only as a
reference.

## Deliberate mathematical choices

- Initial k-means uses `floor(sqrt(n))` balls, bounded to at least two when
  possible, as stated in the paper’s Algorithm 1.
- A ball’s center is the coordinate-wise mean; its radius is the mean
  Euclidean distance to that center; its label is the deterministic majority
  class; and its purity is majority count divided by ball size.
- Impure balls are recursively split into two k-means children until they meet
  the purity threshold or cannot be split. `min_samples=1` means singleton
  balls are terminal, matching the stopping behavior visible in the reference
  script.
- For a feature subset `B`, distances are computed between projected ball
  centers. The distance parameter is `C=|B|` by default, following the paper’s
  incremental-parameter theorem. The implementation does not clip the
  paper’s distance-form lower approximation.
- The weighted dependency is exactly the paper’s sample-mass weighted sum of
  granular-ball positive-domain values divided by the total represented
  sample count. It is not the authors’ released script’s alternative
  `sum_each_ball` score.
- With a fixed granular-ball partition and fixed positive `C`, adding a
  feature cannot decrease Euclidean center distances, so the Equation 9/13
  dependency is monotonic non-decreasing. The selector’s default convention
  uses `C=|B|`; because that denominator changes when features are added, the
  resulting sequence is not guaranteed to be monotonic. The greedy selector
  therefore correctly applies its explicit strict-improvement stopping rule.
- Greedy selection keeps a candidate only when dependency strictly increases.
  Candidate ties resolve by the original column order for reproducibility.
- The primary stopping rule is the direct operational rule in Algorithm 1 and
  its accompanying explanation: at each step, choose the remaining feature
  with the largest weighted dependency and accept it only when
  `W(B ∪ {a}) > W(B)`. Equation 19/Theorem 3 is retained as a diagnostic
  significance calculation, not as a replacement criterion. This is an
  explicit interpretation because the paper's changing-`C` normalization is
  not fully reconciled with the pseudocode's direct comparison. A regression
  test demonstrates that Equation 19 significance can be positive even when
  the direct dependency decreases; the selector still stops in that case.

## Ambiguities and known paper/code differences

1. The paper requires removal of heterogeneous-ball overlaps but does not give
   a fully operational procedure. This implementation recursively splits all
   splittable balls in each heterogeneous-overlap component, following the
   broad behavior of the reference script. Unsplittable singleton overlaps are
   retained because removing them would drop or duplicate samples.
2. The paper describes `sqrt(n)` initial balls. The reference script instead
   chooses approximately `sqrt(class_count)` random seeds per class, and its
   main program evaluates purity thresholds from 0.60 to 0.95. This
   implementation follows the paper’s stated initial-ball count; the selector
   default threshold is 0.95 but remains configurable.
3. The pseudocode’s while-condition is typographically inconsistent. The
   implementation uses the intended operational rule: continue while a
   remaining feature yields a strict dependency increase.
4. The paper states that the similarity is in `[0, 1]` but does not prescribe
   a universal `C` that guarantees this for arbitrary feature scales. Equation
   9 is a distance-form lower approximation, so the resulting dependency can
   exceed 1 when distances exceed `sqrt(C)`. No silent clipping or global
   normalization is performed here. The later BUS-BRA pipeline must fit
   `StandardScaler` inside each training partition before calling this
   selector, as required by the project protocol.
5. The tests below validate the formulas on hand-constructed balls and verify
   selector invariants. They do not constitute a complete reproduction of the
   authors’ UCI experiments or prove identical output to the unlicensed
   reference script.
