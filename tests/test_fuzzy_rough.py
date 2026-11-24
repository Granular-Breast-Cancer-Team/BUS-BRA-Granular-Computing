import numpy as np
import pytest

from fuzzy_rough.selector import select_features


def test_jensen_and_shen_published_lower_approximation_example():
    # Figure 3 and Section III-A.2, DOI 10.1109/TFUZZ.2008.924209.
    X = np.array([
        [-0.4, -0.3, -0.5],
        [-0.4,  0.2, -0.1],
        [-0.3, -0.4, -0.3],
        [ 0.3, -0.3,  0.0],
        [ 0.2, -0.3,  0.0],
        [ 0.2,  0.0,  0.0],
    ])
    y = np.array([0, 1, 0, 1, 1, 0])

    result = select_features(X, y)

    assert result.indices == (2, 1)
    assert result.dependencies == pytest.approx((0.4078, 1.0), abs=0.0001)
    assert result.full_dependency == 1.0
    assert result.stop_reason == "full_dependency"


def test_identical_columns_break_ties_in_original_order():
    X = np.array([[9, 0, 0], [9, 0, 0], [9, 1, 1], [9, 1, 1]])

    result = select_features(X, np.array([0, 0, 1, 1]))

    assert result.indices == (1,)
    assert result.dependencies == (1.0,)


def test_indistinguishable_classes_have_no_reduct_for_svc():
    X = np.zeros((4, 3))

    result = select_features(X, np.array([0, 0, 1, 1]))

    assert result.indices == ()
    assert result.full_dependency == 0.0
