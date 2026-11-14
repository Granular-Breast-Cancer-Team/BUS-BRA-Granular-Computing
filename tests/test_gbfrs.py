import numpy as np
import pytest
from sklearn.base import clone

from gbfrs import (
    GBFRSSelector,
    GranularBall,
    build_granular_balls,
    fuzzy_lower_approximations,
    heterogeneous_overlap_pairs,
    resolve_heterogeneous_overlaps,
    weighted_fuzzy_dependency,
)
from gbfrs.selector import _pairwise_euclidean_distances


def _assert_partition(balls, n_samples):
    memberships = [index for ball in balls for index in ball.indices]
    assert sorted(memberships) == list(range(n_samples))
    assert len(memberships) == len(set(memberships))


def test_ball_statistics_center_radius_label_and_purity():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    y = np.array([0, 0, 1])

    ball = GranularBall.from_indices(X, y, [0, 1, 2])

    np.testing.assert_allclose(ball.center, [1.0, 0.0])
    assert ball.radius == pytest.approx(2.0 / 3.0)
    assert ball.label == 0
    assert ball.purity == pytest.approx(2.0 / 3.0)
    assert ball.n_samples == 3


def test_recursive_splitting_terminates_and_preserves_partition():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.arange(12) % 2

    balls = build_granular_balls(
        X,
        y,
        purity_threshold=1.0,
        random_state=7,
        resolve_overlaps=False,
    )

    _assert_partition(balls, len(X))
    assert all(ball.purity >= 1.0 or ball.n_samples == 1 for ball in balls)


def test_initial_construction_starts_with_sqrt_n_balls_when_pure():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0], [20.0], [21.0], [22.0]])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

    balls = build_granular_balls(
        X,
        y,
        purity_threshold=1.0,
        random_state=5,
        resolve_overlaps=False,
    )

    _assert_partition(balls, len(X))
    assert len(balls) == 3  # floor(sqrt(9)) initial k-means balls


def test_heterogeneous_overlaps_are_split_without_row_loss():
    # Memberships are disjoint, but the two class balls overlap geometrically.
    X = np.array([[0.0], [0.9], [0.8], [1.7]])
    y = np.array([0, 0, 1, 1])
    first = GranularBall.from_indices(X, y, [0, 1])
    second = GranularBall.from_indices(X, y, [2, 3])

    resolved = resolve_heterogeneous_overlaps(
        X,
        y,
        [first, second],
        random_state=11,
    )

    _assert_partition(resolved, len(X))
    assert all(
        not (left.label != right.label and distance < left.radius + right.radius)
        for left_index, left in enumerate(resolved)
        for right in resolved[left_index + 1 :]
        for distance in [float(np.linalg.norm(left.center - right.center))]
    )


def test_cdist_pairwise_distances_match_broadcast_reference():
    rng = np.random.default_rng(123)
    left = rng.normal(size=(7, 4))
    right = rng.normal(size=(5, 4))

    expected = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    actual = _pairwise_euclidean_distances(left, right)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_fuzzy_lower_approximations_match_broadcast_reference():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [3.0, 3.0],
            [3.0, 4.0],
            [0.0, 5.0],
            [0.0, 6.0],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 0])
    balls = [
        GranularBall.from_indices(X, y, [0, 1]),
        GranularBall.from_indices(X, y, [2, 3]),
        GranularBall.from_indices(X, y, [4, 5]),
    ]
    centers = np.vstack([X[list(ball.indices)][:, [0, 1]].mean(axis=0) for ball in balls])
    labels = np.asarray([ball.label for ball in balls])
    broadcast_distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2
    )
    expected = np.zeros(len(balls))
    for decision_class in (0, 1):
        expected = np.maximum(
            expected,
            broadcast_distances[labels != decision_class].min(axis=0) / np.sqrt(2.0),
        )

    actual = fuzzy_lower_approximations(balls, X, [0, 1])
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_coincident_zero_radius_centers_have_zero_dependency_and_no_overlap():
    X = np.array([[2.0], [2.0]])
    y = np.array([0, 1])
    balls = [
        GranularBall.from_indices(X, y, [0]),
        GranularBall.from_indices(X, y, [1]),
    ]

    assert heterogeneous_overlap_pairs(balls) == []
    np.testing.assert_allclose(fuzzy_lower_approximations(balls, X, [0]), [0.0, 0.0])
    assert weighted_fuzzy_dependency(balls, X, [0]) == pytest.approx(0.0)


def test_positive_radius_coincident_centers_are_resolved_deterministically():
    X = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    y = np.array([0, 0, 1, 1])
    balls = [
        GranularBall.from_indices(X, y, [0, 1]),
        GranularBall.from_indices(X, y, [2, 3]),
    ]

    assert heterogeneous_overlap_pairs(balls) == [(0, 1)]
    resolved = resolve_heterogeneous_overlaps(X, y, balls, random_state=4)

    assert heterogeneous_overlap_pairs(resolved) == []
    _assert_partition(resolved, len(X))


def test_unsplittable_singleton_overlap_terminates_without_row_loss():
    X = np.array([[0.0], [0.0]])
    y = np.array([0, 1])
    balls = [
        GranularBall.from_indices(X, y, [0]),
        GranularBall.from_indices(X, y, [1]),
    ]

    resolved = resolve_heterogeneous_overlaps(X, y, balls, random_state=4)

    assert resolved == balls
    _assert_partition(resolved, len(X))


def test_duplicate_points_do_not_create_empty_initial_balls():
    X = np.zeros((9, 2))
    y = np.array([0] * 8 + [1])

    balls = build_granular_balls(
        X,
        y,
        purity_threshold=0.1,
        random_state=17,
        resolve_overlaps=False,
    )

    assert balls
    assert all(ball.n_samples > 0 for ball in balls)
    _assert_partition(balls, len(X))


def test_selector_reports_missing_ball_class_explicitly(monkeypatch):
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0, 0, 1, 1])
    one_class_ball = GranularBall.from_indices(X, y, [0, 1])

    import gbfrs.selector as selector_module

    monkeypatch.setattr(
        selector_module,
        "build_granular_balls",
        lambda *args, **kwargs: [one_class_ball],
    )
    with pytest.raises(ValueError, match="final granular balls"):
        GBFRSSelector().fit(X, y)


def test_lower_approximation_requires_both_ball_classes():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 0])
    balls = [GranularBall.from_indices(X, y, [0]), GranularBall.from_indices(X, y, [1])]

    with pytest.raises(ValueError, match="both decision classes"):
        fuzzy_lower_approximations(balls, X, [0])


def test_overlap_resolution_is_deterministic_with_fixed_seed():
    X = np.array([[0.0], [0.9], [0.8], [1.7]])
    y = np.array([0, 0, 1, 1])
    balls = [
        GranularBall.from_indices(X, y, [0, 1]),
        GranularBall.from_indices(X, y, [2, 3]),
    ]

    first = resolve_heterogeneous_overlaps(X, y, balls, random_state=11)
    second = resolve_heterogeneous_overlaps(X, y, balls, random_state=11)

    assert [(ball.indices, ball.label) for ball in first] == [
        (ball.indices, ball.label) for ball in second
    ]
    for left, right in zip(first, second):
        np.testing.assert_allclose(left.center, right.center)


def test_weighted_dependency_matches_manual_two_ball_calculation():
    X = np.array([[0.0], [0.2], [10.0], [10.2]])
    y = np.array([0, 0, 1, 1])
    balls = [
        GranularBall.from_indices(X, y, [0, 1]),
        GranularBall.from_indices(X, y, [2, 3]),
    ]

    lower = fuzzy_lower_approximations(balls, X, [0], distance_parameter=1.0)
    dependency = weighted_fuzzy_dependency(balls, X, [0], distance_parameter=1.0)

    np.testing.assert_allclose(lower, [10.0, 10.0])
    assert dependency == pytest.approx(10.0)


def test_dependency_uses_feature_count_as_default_distance_parameter():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [3.0, 3.0], [3.0, 4.0]])
    y = np.array([0, 0, 1, 1])
    balls = [
        GranularBall.from_indices(X, y, [0, 1]),
        GranularBall.from_indices(X, y, [2, 3]),
    ]

    # Center distance is sqrt(18); C defaults to |B| = 2.
    dependency = weighted_fuzzy_dependency(balls, X, [0, 1])
    assert dependency == pytest.approx(3.0)


def test_fixed_distance_parameter_makes_dependency_monotonic():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.1],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 0])
    balls = [
        GranularBall.from_indices(X, y, [0, 1]),
        GranularBall.from_indices(X, y, [2, 3]),
        GranularBall.from_indices(X, y, [4, 5]),
    ]

    one = weighted_fuzzy_dependency(
        balls, X, [0], distance_parameter=1.0
    )
    two = weighted_fuzzy_dependency(
        balls, X, [0, 1], distance_parameter=1.0
    )
    three = weighted_fuzzy_dependency(
        balls, X, [0, 1, 2], distance_parameter=1.0
    )

    assert one <= two <= three


def test_paper_distance_form_is_not_bounded_to_one_without_normalization():
    X = np.array([[0.0], [3.0]])
    y = np.array([0, 1])
    balls = [
        GranularBall.from_indices(X, y, [0]),
        GranularBall.from_indices(X, y, [1]),
    ]

    dependency = weighted_fuzzy_dependency(balls, X, [0], distance_parameter=1.0)

    assert dependency == pytest.approx(3.0)
    assert dependency > 1.0


def test_default_cardinality_distance_parameter_can_reduce_dependency():
    X = np.array([[0.0, 0.0], [10.0, 0.0]])
    y = np.array([0, 1])
    balls = [
        GranularBall.from_indices(X, y, [0]),
        GranularBall.from_indices(X, y, [1]),
    ]

    one_feature = weighted_fuzzy_dependency(balls, X, [0])
    two_features = weighted_fuzzy_dependency(balls, X, [0, 1])

    assert one_feature == pytest.approx(10.0)
    assert two_features == pytest.approx(10.0 / np.sqrt(2.0))
    assert two_features < one_feature


def test_equation19_significance_can_be_positive_when_direct_dependency_decreases():
    X = np.array([[0.0, 0.0], [10.0, 1.0]])
    y = np.array([0, 1])
    balls = [
        GranularBall.from_indices(X, y, [0]),
        GranularBall.from_indices(X, y, [1]),
    ]

    current = weighted_fuzzy_dependency(balls, X, [0])
    candidate = weighted_fuzzy_dependency(balls, X, [0, 1])
    equation19_significance = candidate - np.sqrt(1.0 / 2.0) * current

    assert candidate < current
    assert equation19_significance > 0.0


def test_selector_uses_greedy_strict_improvement_and_stops():
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [10.0, 0.0],
            [10.2, 0.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    selector = GBFRSSelector(
        purity_threshold=1.0,
        resolve_overlaps=False,
        random_state=3,
    ).fit(X, y)

    np.testing.assert_array_equal(selector.get_support(indices=True), [0])
    assert selector.dependency_ > 0
    assert selector.selection_history_[-1]["best_feature"] == 1
    assert selector.selection_history_[-1]["best_dependency"] < selector.dependency_


def test_fixed_seed_reproduces_balls_and_selected_features():
    X = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.2, 1.0],
            [0.2, 0.0, 1.0],
            [4.0, 4.0, 1.0],
            [4.1, 3.9, 1.0],
            [3.9, 4.1, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    first = GBFRSSelector(random_state=19).fit(X, y)
    second = GBFRSSelector(random_state=19).fit(X, y)

    np.testing.assert_array_equal(first.get_support(indices=True), second.get_support(indices=True))
    first_geometry = [(ball.indices, ball.label, ball.purity) for ball in first.balls_]
    second_geometry = [(ball.indices, ball.label, ball.purity) for ball in second.balls_]
    assert first_geometry == second_geometry
    for left, right in zip(first.balls_, second.balls_):
        np.testing.assert_allclose(left.center, right.center)


def test_binary_encoding_is_required():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="encoded exactly as 0 and 1"):
        GBFRSSelector().fit(X, np.array([-1, -1, 1, 1]))
    with pytest.raises(ValueError, match="encoded exactly as 0 and 1"):
        GBFRSSelector().fit(X, np.array([0, 0, 0, 0]))


def test_sklearn_interface_and_transform_do_not_refit_on_test_rows():
    X_train = np.array([[0.0], [0.1], [5.0], [5.1]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[100.0], [100.1]])
    selector = GBFRSSelector(resolve_overlaps=False, random_state=2)

    assert clone(selector).get_params()["random_state"] == 2
    selector.fit(X_train, y_train)
    support_before = selector.get_support(indices=True)
    balls_before = [(ball.indices, ball.label) for ball in selector.balls_]
    history_before = list(selector.selection_history_)
    transformed = selector.transform(X_test)

    assert selector.n_samples_in_ == len(X_train)
    assert transformed.shape == (2, len(support_before))
    np.testing.assert_array_equal(selector.get_support(indices=True), support_before)
    assert selector.selection_history_ == history_before
    assert [(ball.indices, ball.label) for ball in selector.balls_] == balls_before
