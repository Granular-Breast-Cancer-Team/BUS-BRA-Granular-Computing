# Conventional fuzzy-rough selection + SVM

This experiment tests whether lower-approximation fuzzy-rough feature selection can reduce the 288 BUS-BRA radiomic predictors while preserving benign-versus-malignant classification. Each image is predicted once, with all images from the same patient kept together. [Five-fold results](RESULTS.md)

## Feature selection

The selector implements the similarity-based lower-approximation method and forward QuickReduct search from Section III-A of Jensen and Shen. It uses the triangular relation in Equation 26 and the Łukasiewicz connectives from the worked example. For training rows $i,j$ and feature $a$, with training sample standard deviation $s_a$,

$$
R_a(i,j) = \max(0, 1 - |x_{ia}-x_{ja}|/s_a).
$$

A constant feature has similarity 1 for every pair. The relation is reflexive, symmetric and Łukasiewicz-transitive because its complementary distance is a truncated, scaled absolute difference. Feature relations are combined with $T(u,v)=\max(0,u+v-1)$, and lower approximations use $I(u,v)=\min(1,1-u+v)$.

For binary crisp decisions, a row's membership in the positive region comes from the lower approximation of its own class. The implementation computes the same dependency using only opposite-label pairs. With $D_a=1-R_a$,

$$
\gamma(P) = \frac{1}{n}\sum_i \min_{j:\,y_j \ne y_i}
  \min\left(1,\sum_{a\in P}D_a(i,j)\right).
$$

QuickReduct starts with the empty set and adds the feature yielding the largest training dependency at each step. It stops at the full-set dependency or when no remaining feature improves dependency, using a fixed tolerance of $10^{-12}$. Numerical ties follow the original CSV column order. The implementation uses sample standard deviation to reproduce the paper's six-object example, which selects $c$ then $b$ with dependencies approximately 0.4078 and 1.0. The regression test checks both values and their order.

The cross-class distance cache lives for one selector fit. All 288 columns enter each fit, including the handoff's exact-value redundancies. The Łukasiewicz t-norm can give additional weight to duplicate values when both columns are selected, so reported feature counts refer to original columns.

## Patient-grouped evaluation

- Use the five frozen `fold5_kfold` assignments. Hold out one complete fold at a time.
- Split each outer training set with three-fold `StratifiedGroupKFold`, grouped by `patient_id`, with `shuffle=True` and seed `42 + outer_fold`. Check patient disjointness at both levels.
- Fit `StandardScaler` and QuickReduct on each inner-training partition. Use their fitted transformations on the matching validation rows. Reuse the reduced matrices for all 32 SVC candidates within that split.
- Choose the highest mean inner macro-F1. Break exact ties by higher mean accuracy, fewer mean selected features, then the shared grid order in code. All SVC candidates within a fold share the same three inner reducts, so the feature-count tie-break does not distinguish candidates in this experiment.
- Refit scaling, QuickReduct and the winning SVC on the complete outer-training partition. Transform and predict the held-out partition with the refitted pipeline and the default SVC decision threshold.

The shared grid covers $C \in \{0.1,1,10,100\}$, linear and RBF kernels, `None` and `balanced` class weights, and RBF gamma values `scale`, `0.01` and `0.1`. See the [team evaluation protocol](../docs/EVALUATION_PROTOCOL.md).

## Run from the repository root

Use Python 3.11 and the pinned dependencies. Place the frozen CSV at the path shown below. The runner checks the original SHA-256 before fitting.

~~~bash
python3.11 -m venv .venv-busbra
source .venv-busbra/bin/activate
python -m pip install -r requirements.txt
python -m pytest -p no:cacheprovider -q
python -m fuzzy_rough.experiment \
  --dataset inputs/BUSBRA_features_ready.csv \
  --output-dir submissions/fuzzy_rough
~~~

The offline handoff contains the five team-format files. The public repository tracks the implementation and aggregate results. See the [submission format](../docs/SUBMISSION_FORMAT.md).

| File | Contents |
|---|---|
| `predictions.csv` | 1,875 held-out predictions with image IDs, patient groups, folds and labels |
| `selected_features.csv` | Original feature names per outer fold, in QuickReduct selection order |
| `best_params.csv` | Winning SVC settings and inner-CV scores, with empty cells for unused gamma and unweighted SVC |
| `timings.csv` | Combined inner and outer selector fit time, candidate scoring time, full outer refit time, prediction time and elapsed fold time. The outer selector fit also contributes to the full refit time. |
| `run_info.json` | Dataset fingerprint, versions, all inner candidate scores, dependency paths, and fold and pooled metrics |

## References

- R. Jensen and Q. Shen, “New Approaches to Fuzzy-Rough Feature Selection,” *IEEE Transactions on Fuzzy Systems*, 17(4), 824–838, 2009. [Paper](https://doi.org/10.1109/TFUZZ.2008.924209)
- W. Gómez-Flores, M. J. Gregorio-Calas and W. Coelho de Albuquerque Pereira, “BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems,” *Medical Physics*, 51(4), 3110–3123, 2024. [Paper](https://doi.org/10.1002/mp.16812) · [Dataset record](https://doi.org/10.5281/zenodo.8231412)
