"""Jensen and Shen's lower-approximation fuzzy-rough QuickReduct."""

from dataclasses import dataclass

import numpy as np


PAPER_DOI = "10.1109/TFUZZ.2008.924209"
DEPENDENCY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Reduct:
    """Feature indices in search order and their cumulative dependencies."""

    indices: tuple[int, ...]
    dependencies: tuple[float, ...]
    full_dependency: float
    stop_reason: str


def _dependency(pair_distances: np.ndarray) -> float:
    # For crisp labels each object's nonzero lower approximation can only be
    # for its own class. Same-class comparisons contribute 1 to that minimum.
    # Row and column minima cover both classes of this cross-class matrix.
    row_membership = np.minimum(1.0, pair_distances.min(axis=1))
    column_membership = np.minimum(1.0, pair_distances.min(axis=0))
    return float(
        (row_membership.sum() + column_membership.sum())
        / (row_membership.size + column_membership.size)
    )


def select_features(X: np.ndarray, y: np.ndarray) -> Reduct:
    """Select original columns using only these binary-labeled training rows.

    Similarity is max(1 - |x_i - x_j| / sample_std, 0). The Łukasiewicz
    t-norm and implicator reduce the binary dependency calculation to
    clipped sums of feature distances over opposing-class pairs.
    """
    values = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("X must be a finite matrix with at least one column")
    if labels.shape != (len(values),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("y must contain both binary labels and one label per row")

    benign = values[labels == 0]
    malignant = values[labels == 1]
    scales = values.std(axis=0, ddof=1)
    scales[scales == 0] = 1.0  # A constant has zero distance for every pair.

    # Cache one distance matrix per feature for this training partition only.
    # Distances above 1 need not be clipped until taking membership because
    # all additions are nonnegative and the Łukasiewicz sum saturates at 1.
    shape = (len(benign), len(malignant))
    distances = np.empty((values.shape[1], *shape), dtype=np.float64)
    for feature, scale in enumerate(scales):
        distance = distances[feature]
        np.subtract(benign[:, feature, None], malignant[None, :, feature], out=distance)
        np.abs(distance, out=distance)
        distance /= scale

    full_dependency = _dependency(distances.sum(axis=0))
    current_distances = np.zeros(shape, dtype=np.float64)
    trial_distances = np.empty_like(current_distances)
    remaining = list(range(values.shape[1]))
    chosen = []
    dependencies = []
    current_dependency = 0.0
    while full_dependency - current_dependency > DEPENDENCY_TOLERANCE:
        scores = []
        for feature in remaining:
            np.add(current_distances, distances[feature], out=trial_distances)
            scores.append(_dependency(trial_distances))
        highest = max(scores)
        if highest - current_dependency <= DEPENDENCY_TOLERANCE:
            return Reduct(tuple(chosen), tuple(dependencies), full_dependency, "no_gain")

        # Input column order breaks numerical ties deterministically.
        rank = next(
            rank for rank, score in enumerate(scores)
            if highest - score <= DEPENDENCY_TOLERANCE and score > current_dependency
        )
        selected = remaining.pop(rank)
        current_distances += distances[selected]
        current_dependency = scores[rank]
        chosen.append(selected)
        dependencies.append(current_dependency)

    return Reduct(tuple(chosen), tuple(dependencies), full_dependency, "full_dependency")
