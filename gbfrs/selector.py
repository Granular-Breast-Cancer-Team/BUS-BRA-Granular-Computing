"""Paper-based granular-ball fuzzy-rough feature selection.

This module follows the mathematical definitions in Xia et al.,
"GBFRS: Robust Fuzzy Rough Sets via Granular-ball Computing" (2025).
It deliberately does not reproduce the alternative scoring expression in
the authors' released, unlicensed research script.  See technical_note.md
for the implementation choices and paper/code ambiguities.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import time
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


ArrayLike = np.ndarray | Sequence[Sequence[float]]


def _as_feature_indices(feature_indices: Iterable[int], n_features: int) -> np.ndarray:
    indices = np.asarray(list(feature_indices), dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("feature_indices must contain at least one feature")
    if np.any(indices < 0) or np.any(indices >= n_features):
        raise IndexError("feature_indices contains an out-of-range column")
    if np.unique(indices).size != indices.size:
        raise ValueError("feature_indices must not contain duplicates")
    return indices


def _validate_binary_labels(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y)
    unique = np.unique(labels)
    if unique.size != 2 or set(unique.tolist()) != {0, 1}:
        raise ValueError("GBFRS requires binary labels encoded exactly as 0 and 1")
    return labels.astype(int, copy=False)


def _pairwise_euclidean_distances(
    left: np.ndarray, right: np.ndarray | None = None
) -> np.ndarray:
    """Compute Euclidean distances without an O(m²d) broadcast tensor."""

    left_array = np.asarray(left, dtype=float)
    right_array = left_array if right is None else np.asarray(right, dtype=float)
    return cdist(left_array, right_array, metric="euclidean")


def _majority_label(y: np.ndarray) -> tuple[int, float]:
    values, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    # np.unique sorts the values, so ties resolve deterministically to 0.
    label = int(values[np.flatnonzero(counts == max_count)[0]])
    return label, float(max_count / y.size)


@dataclass(frozen=True)
class GranularBall:
    """A ball represented by its member-row indices and full-space geometry."""

    indices: tuple[int, ...]
    center: np.ndarray
    radius: float
    label: int
    purity: float

    @classmethod
    def from_indices(
        cls, X: np.ndarray, y: np.ndarray, indices: Iterable[int]
    ) -> "GranularBall":
        member_indices = tuple(int(i) for i in indices)
        if not member_indices:
            raise ValueError("a granular ball cannot be empty")
        data = X[np.asarray(member_indices, dtype=int)]
        center = data.mean(axis=0)
        radius = float(np.linalg.norm(data - center, axis=1).mean())
        label, purity = _majority_label(y[np.asarray(member_indices, dtype=int)])
        return cls(member_indices, center, radius, label, purity)

    @property
    def n_samples(self) -> int:
        return len(self.indices)


def _split_ball(
    X: np.ndarray,
    y: np.ndarray,
    ball: GranularBall,
    random_state: int | None,
    n_init: int,
) -> list[GranularBall]:
    """Split a ball into two non-empty k-means children, if possible."""

    if ball.n_samples < 2:
        return [ball]
    member_indices = np.asarray(ball.indices, dtype=int)
    km = KMeans(n_clusters=2, n_init=n_init, random_state=random_state)
    assignments = km.fit_predict(X[member_indices])
    if np.unique(assignments).size != 2:
        return [ball]
    return [
        GranularBall.from_indices(X, y, member_indices[assignments == cluster])
        for cluster in (0, 1)
    ]


def heterogeneous_overlap_pairs(
    balls: Sequence[GranularBall],
) -> list[tuple[int, int]]:
    if len(balls) < 2:
        return []
    centers = np.vstack([ball.center for ball in balls])
    radii = np.asarray([ball.radius for ball in balls], dtype=float)
    labels = np.asarray([ball.label for ball in balls], dtype=int)
    distances = _pairwise_euclidean_distances(centers)
    overlap = distances < (radii[:, None] + radii[None, :])
    heterogeneous = labels[:, None] != labels[None, :]
    rows, cols = np.where(np.triu(overlap & heterogeneous, k=1))
    return list(zip(rows.tolist(), cols.tolist()))


def resolve_heterogeneous_overlaps(
    X: np.ndarray,
    y: np.ndarray,
    balls: Sequence[GranularBall],
    *,
    random_state: int | None = 0,
    min_samples: int = 1,
    n_init: int = 10,
    max_rounds: int | None = None,
) -> list[GranularBall]:
    """Resolve heterogeneous overlaps by recursively splitting affected balls.

    The paper requires heterogeneous-ball overlap removal but does not specify
    an exact operation.  This implementation uses the operation shown in the
    official research script: split balls in overlap components until no
    splittable heterogeneous overlap remains.  If every remaining overlap is
    between unsplittable singleton balls, it is retained and reported through
    the returned geometry rather than duplicating or dropping samples.
    """

    current = list(balls)
    if max_rounds is None:
        max_rounds = max(1, 4 * len(current) + 4)

    for _ in range(max_rounds):
        pairs = heterogeneous_overlap_pairs(current)
        if not pairs:
            return current

        adjacency = [set() for _ in current]
        for left, right in pairs:
            adjacency[left].add(right)
            adjacency[right].add(left)

        components: list[list[int]] = []
        seen: set[int] = set()
        for start in range(len(current)):
            if start in seen or not adjacency[start]:
                continue
            stack = [start]
            seen.add(start)
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbour in adjacency[node]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            components.append(sorted(component))

        component_members = {node for component in components for node in component}
        next_balls: list[GranularBall] = []
        changed = False
        for index, ball in enumerate(current):
            if index not in component_members:
                next_balls.append(ball)
                continue
            if ball.n_samples <= min_samples:
                next_balls.append(ball)
                continue
            children = _split_ball(X, y, ball, random_state, n_init)
            next_balls.extend(children)
            changed = changed or len(children) == 2

        current = next_balls
        if not changed:
            return current

    return current


def build_granular_balls(
    X: ArrayLike,
    y: Sequence[int],
    *,
    purity_threshold: float = 0.95,
    min_samples: int = 1,
    random_state: int | None = 0,
    n_init: int = 10,
    resolve_overlaps: bool = True,
    max_overlap_rounds: int | None = None,
) -> list[GranularBall]:
    """Construct and purity-refine the paper's granular-ball representation."""

    data = check_array(X, dtype=float, ensure_2d=True, ensure_min_samples=2)
    labels = _validate_binary_labels(np.asarray(y))
    if data.shape[0] != labels.shape[0]:
        raise ValueError("X and y have different numbers of rows")
    if not 0.0 < purity_threshold <= 1.0:
        raise ValueError("purity_threshold must be in (0, 1]")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")

    initial_count = min(data.shape[0], max(2, int(sqrt(data.shape[0]))))
    if initial_count == 1:
        initial = [GranularBall.from_indices(data, labels, range(data.shape[0]))]
    else:
        initial_model = KMeans(
            n_clusters=initial_count,
            n_init=n_init,
            random_state=random_state,
        )
        assignments = initial_model.fit_predict(data)
        # KMeans can return fewer occupied clusters than requested when rows
        # are coincident.  Build only occupied balls; never construct an
        # empty ball from an unused cluster id.
        initial = [
            GranularBall.from_indices(data, labels, np.flatnonzero(assignments == cluster))
            for cluster in np.unique(assignments)
        ]

    refined: list[GranularBall] = []
    pending = list(initial)
    while pending:
        ball = pending.pop(0)
        if ball.purity >= purity_threshold or ball.n_samples <= min_samples:
            refined.append(ball)
            continue
        children = _split_ball(data, labels, ball, random_state, n_init)
        if len(children) == 1:
            refined.append(ball)
        else:
            pending.extend(children)

    if resolve_overlaps:
        refined = resolve_heterogeneous_overlaps(
            data,
            labels,
            refined,
            random_state=random_state,
            min_samples=min_samples,
            n_init=n_init,
            max_rounds=max_overlap_rounds,
        )
    return refined


def _projected_ball_centers(
    X: np.ndarray, balls: Sequence[GranularBall], feature_indices: np.ndarray
) -> np.ndarray:
    return np.vstack([X[np.asarray(ball.indices)][:, feature_indices].mean(axis=0) for ball in balls])


def fuzzy_lower_approximations(
    balls: Sequence[GranularBall],
    X: ArrayLike,
    feature_indices: Iterable[int],
    *,
    distance_parameter: float | None = None,
) -> np.ndarray:
    """Return GBFRS positive-domain membership for each granular ball.

    For each decision class, the lower approximation is the minimum
    cross-class center distance divided by sqrt(C).  The fuzzy positive-domain
    value is the maximum of those lower approximations over decision classes.
    No clipping is applied because the paper's distance form does not define a
    clipping rule.
    """

    data = check_array(X, dtype=float, ensure_2d=True)
    if not balls:
        raise ValueError("at least one granular ball is required")
    indices = _as_feature_indices(feature_indices, data.shape[1])
    C = float(indices.size if distance_parameter is None else distance_parameter)
    if C <= 0.0:
        raise ValueError("distance_parameter must be positive")

    centers = _projected_ball_centers(data, balls, indices)
    labels = np.asarray([ball.label for ball in balls], dtype=int)
    positive_domain = np.zeros(len(balls), dtype=float)
    distances = _pairwise_euclidean_distances(centers)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(
            "fuzzy lower approximations require both decision classes among the balls"
        )
    for decision_class in np.unique(labels):
        not_class = labels != decision_class
        if np.any(not_class):
            lower = distances[not_class, :].min(axis=0) / sqrt(C)
            positive_domain = np.maximum(positive_domain, lower)
    return positive_domain


def weighted_fuzzy_dependency(
    balls: Sequence[GranularBall],
    X: ArrayLike,
    feature_indices: Iterable[int],
    *,
    n_total_samples: int | None = None,
    distance_parameter: float | None = None,
) -> float:
    """Compute the paper's weighted granular-ball fuzzy dependency."""

    data = check_array(X, dtype=float, ensure_2d=True)
    if not balls:
        raise ValueError("at least one granular ball is required")
    positive_domain = fuzzy_lower_approximations(
        balls,
        data,
        feature_indices,
        distance_parameter=distance_parameter,
    )
    total = int(sum(ball.n_samples for ball in balls) if n_total_samples is None else n_total_samples)
    if total <= 0:
        raise ValueError("n_total_samples must be positive")
    if total != sum(ball.n_samples for ball in balls):
        raise ValueError("n_total_samples must equal the represented sample count")
    weights = np.asarray([ball.n_samples for ball in balls], dtype=float)
    return float(np.dot(weights, positive_domain) / total)


class GBFRSSelector(BaseEstimator, TransformerMixin):
    """Scikit-learn-compatible greedy GBFRS feature selector.

    The selector expects binary labels encoded as 0 and 1.  It assumes that
    any required scaling is performed by an upstream pipeline step, so that a
    scaler can be fitted on the same training partition as this selector.
    """

    def __init__(
        self,
        purity_threshold: float = 0.95,
        min_samples: int = 1,
        random_state: int | None = 0,
        n_init: int = 10,
        resolve_overlaps: bool = True,
        max_overlap_rounds: int | None = None,
        improvement_tol: float = 0.0,
    ) -> None:
        self.purity_threshold = purity_threshold
        self.min_samples = min_samples
        self.random_state = random_state
        self.n_init = n_init
        self.resolve_overlaps = resolve_overlaps
        self.max_overlap_rounds = max_overlap_rounds
        self.improvement_tol = improvement_tol

    def fit(self, X: ArrayLike, y: Sequence[int]) -> "GBFRSSelector":
        started = time.perf_counter()
        input_features = getattr(X, "columns", None)
        data, labels = check_X_y(X, y, dtype=float, ensure_2d=True)
        labels = _validate_binary_labels(labels)
        if self.improvement_tol < 0:
            raise ValueError("improvement_tol must be non-negative")

        self.n_features_in_ = data.shape[1]
        self.initial_ball_count_ = min(
            data.shape[0], max(2, int(sqrt(data.shape[0])))
        )
        if input_features is not None and all(isinstance(name, str) for name in input_features):
            self.feature_names_in_ = np.asarray(input_features, dtype=object)
        self.n_samples_in_ = data.shape[0]
        self.classes_ = np.asarray([0, 1])
        self.balls_ = build_granular_balls(
            data,
            labels,
            purity_threshold=self.purity_threshold,
            min_samples=self.min_samples,
            random_state=self.random_state,
            n_init=self.n_init,
            resolve_overlaps=self.resolve_overlaps,
            max_overlap_rounds=self.max_overlap_rounds,
        )
        self.ball_classes_ = np.unique([ball.label for ball in self.balls_])
        self.both_ball_classes_ = set(self.ball_classes_.tolist()) == {0, 1}
        if not self.both_ball_classes_:
            raise ValueError(
                "final granular balls must represent both decision classes"
            )

        selected: list[int] = []
        remaining = list(range(self.n_features_in_))
        current_dependency = -np.inf
        history: list[dict[str, object]] = []
        while remaining:
            candidates = {}
            for feature in remaining:
                candidate = selected + [feature]
                candidates[feature] = weighted_fuzzy_dependency(
                    self.balls_, data, candidate, n_total_samples=self.n_samples_in_
                )
            best_feature = max(remaining, key=lambda feature: (candidates[feature], -feature))
            best_dependency = candidates[best_feature]
            direct_improvement = (
                None
                if not selected
                else float(best_dependency - current_dependency)
            )
            equation19_significance = (
                None
                if not selected
                else float(
                    best_dependency
                    - sqrt(len(selected) / (len(selected) + 1)) * current_dependency
                )
            )
            history.append(
                {
                    "selected_before": tuple(selected),
                    "candidate_scores": dict(candidates),
                    "best_feature": best_feature,
                    "best_dependency": best_dependency,
                    "direct_improvement": direct_improvement,
                    "equation19_significance": equation19_significance,
                }
            )
            if best_dependency <= current_dependency + self.improvement_tol:
                break
            selected.append(best_feature)
            remaining.remove(best_feature)
            current_dependency = best_dependency

        self.selected_features_ = np.asarray(selected, dtype=int)
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_features_] = True
        self.dependency_ = float(current_dependency) if selected else 0.0
        self.selection_history_ = history
        self.fit_runtime_seconds_ = time.perf_counter() - started
        return self

    def get_support(self, indices: bool = False) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.selected_features_.copy() if indices else self.support_.copy()

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "support_")
        data = check_array(X, dtype=float, ensure_2d=True)
        if data.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted selector")
        return data[:, self.selected_features_]

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, "support_")
        if input_features is None:
            if not hasattr(self, "feature_names_in_"):
                return np.asarray([f"x{i}" for i in self.selected_features_], dtype=object)
            input_features = self.feature_names_in_
        names = np.asarray(input_features, dtype=object)
        if names.shape[0] != self.n_features_in_:
            raise ValueError("input_features has the wrong length")
        return names[self.selected_features_]
