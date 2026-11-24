import numpy as np
import pandas as pd
import pytest

from fuzzy_rough import experiment
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


def test_nested_run_fits_a_fresh_selector_on_each_training_partition(monkeypatch):
    patients = np.arange(90)
    patient_labels = patients % 2
    frame = pd.DataFrame({
        "image_id": [f"image-{row}" for row in range(180)],
        "patient_id": np.repeat(patients, 2),
        "fold5_kfold": np.repeat(patients % 5 + 1, 2),
        "diagnosis": np.repeat(patient_labels, 2),
        "original_signal": np.repeat(patient_labels, 2) + np.arange(180) / 1000,
    })
    features = ["original_signal"]
    fitted_rows = []
    original_prepare = experiment._prepare_training

    def capture_training(X, y):
        fitted_rows.append(X[:, 0].copy())
        return original_prepare(X, y)

    monkeypatch.setattr(experiment, "_prepare_training", capture_training)
    fold = 2
    result = experiment.evaluate_fold(frame, features, fold)

    outer_train = frame.loc[frame.fold5_kfold != fold]
    x = outer_train[features].to_numpy()
    splits = list(experiment.inner_cv(fold).split(x, outer_train.diagnosis, outer_train.patient_id))
    assert len(fitted_rows) == 4
    for actual, (train, _) in zip(fitted_rows[:3], splits):
        np.testing.assert_array_equal(actual, x[train, 0])
    np.testing.assert_array_equal(fitted_rows[3], x[:, 0])
    assert len(result.audit["candidate_scores"]) == 32
    assert result.predictions.image_id.tolist() == frame.loc[frame.fold5_kfold == fold, "image_id"].tolist()
    assert (result.predictions.y_true == result.predictions.y_pred).all()


def test_exact_svm_ties_use_accuracy_then_feature_count_then_grid_order():
    scores = [
        dict(grid_position=1, inner_macro_f1=0.8, inner_accuracy=0.8, mean_inner_feature_count=4),
        dict(grid_position=2, inner_macro_f1=0.8, inner_accuracy=0.9, mean_inner_feature_count=4),
        dict(grid_position=3, inner_macro_f1=0.8, inner_accuracy=0.9, mean_inner_feature_count=3),
        dict(grid_position=4, inner_macro_f1=0.8, inner_accuracy=0.9, mean_inner_feature_count=3),
        dict(grid_position=5, inner_macro_f1=0.7, inner_accuracy=1.0, mean_inner_feature_count=1),
    ]

    assert experiment._best_candidate(scores)["grid_position"] == 3
