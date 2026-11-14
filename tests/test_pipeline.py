import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid

from gbfrs import (
    PROTOCOL_ID,
    binary_metrics,
    common_svm_grid,
    frozen_feature_columns,
    inner_cv,
    make_gbfrs_svm_pipeline,
    make_inner_search,
    outer_split,
    selected_feature_names,
)


FEATURES = [
    "original_firstorder_Mean",
    "original_firstorder_Variance",
    "original_glcm_Contrast",
    "original_glcm_Correlation",
    "wavelet-H_firstorder_Mean",
    "wavelet-L_firstorder_Variance",
]


def synthetic_frame(n_groups=20, rows_per_group=2):
    rows = []
    for group_index in range(n_groups):
        label = group_index % 2
        for image_index in range(rows_per_group):
            base = float(label * 5 + group_index / n_groups)
            rows.append(
                {
                    "image_id": f"image_{group_index}_{image_index}",
                    "patient_id": f"patient_{group_index:02d}",
                    "diagnosis": label,
                    "fold5_kfold": group_index % 5 + 1,
                    **{
                        name: base + 0.1 * image_index + 0.01 * feature_index
                        for feature_index, name in enumerate(FEATURES)
                    },
                }
            )
    return pd.DataFrame(rows)


def test_protocol_outer_and_inner_splits_are_patient_disjoint():
    frame = synthetic_frame()
    train_positions, test_positions = outer_split(frame, 1)
    assert set(frame.iloc[train_positions].patient_id).isdisjoint(
        set(frame.iloc[test_positions].patient_id)
    )

    X = frame[FEATURES]
    y = frame.diagnosis.to_numpy()
    groups = frame.patient_id.to_numpy()
    splits = list(inner_cv(1).split(X, y, groups))
    assert len(splits) == 3
    for train, validation in splits:
        assert set(groups[train]).isdisjoint(set(groups[validation]))
        assert set(y[train]) == {0, 1}
        assert set(y[validation]) == {0, 1}


def test_common_grid_has_exactly_32_valid_svm_candidates():
    candidates = list(ParameterGrid(common_svm_grid()))
    assert len(candidates) == 32
    assert all(
        candidate["svc__kernel"] == "linear" or "svc__gamma" in candidate
        for candidate in candidates
    )
    assert all(
        candidate["svc__kernel"] == "rbf" or "svc__gamma" not in candidate
        for candidate in candidates
    )


def test_pipeline_scales_and_selects_only_training_rows_with_names():
    frame = synthetic_frame()
    train_positions, test_positions = outer_split(frame, 1)
    pipeline = make_gbfrs_svm_pipeline(
        FEATURES,
        selector_params={"resolve_overlaps": False, "random_state": 13},
        svm_params={"kernel": "linear", "C": 1.0},
    )
    X_train = frame.iloc[train_positions][FEATURES]
    y_train = frame.iloc[train_positions].diagnosis
    pipeline.fit(X_train, y_train)

    np.testing.assert_allclose(
        pipeline.named_steps["scaler"].scaler_.mean_, X_train.to_numpy().mean(axis=0)
    )
    assert pipeline.named_steps["gbfrs"].n_samples_in_ == len(train_positions)
    assert all(
        0 <= index < len(train_positions)
        for ball in pipeline.named_steps["gbfrs"].balls_
        for index in ball.indices
    )
    selected = selected_feature_names(pipeline)
    assert set(selected).issubset(set(FEATURES))
    assert set(selected) == set(pipeline.named_steps["gbfrs"].get_feature_names_out())

    predictions = pipeline.predict(frame.iloc[test_positions][FEATURES])
    assert predictions.shape == (len(test_positions),)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_selector_is_refit_as_a_distinct_estimator_for_each_inner_fold():
    frame = synthetic_frame()
    X = frame[FEATURES]
    y = frame.diagnosis.to_numpy()
    groups = frame.patient_id.to_numpy()
    fitted = []
    for train, validation in inner_cv(1).split(X, y, groups):
        pipeline = make_gbfrs_svm_pipeline(
            FEATURES,
            selector_params={"resolve_overlaps": False, "random_state": 17},
            svm_params={"kernel": "linear", "C": 1.0},
        ).fit(X.iloc[train], y[train])
        fitted.append(pipeline.named_steps["gbfrs"])
        assert pipeline.named_steps["gbfrs"].n_samples_in_ == len(train)
        assert set(groups[train]).isdisjoint(set(groups[validation]))
    assert len({id(selector) for selector in fitted}) == 3


def test_held_out_rows_and_labels_do_not_change_fitted_selection():
    frame = synthetic_frame()
    train_positions, test_positions = outer_split(frame, 1)
    X_train = frame.iloc[train_positions][FEATURES]
    y_train = frame.iloc[train_positions].diagnosis.to_numpy()
    X_test = frame.iloc[test_positions][FEATURES]

    first = make_gbfrs_svm_pipeline(
        FEATURES,
        selector_params={"resolve_overlaps": False, "random_state": 23},
        svm_params={"kernel": "linear", "C": 1.0},
    ).fit(X_train, y_train)
    support_before = first.named_steps["gbfrs"].get_support(indices=True)
    first.predict(X_test)
    np.testing.assert_array_equal(
        first.named_steps["gbfrs"].get_support(indices=True), support_before
    )

    second = make_gbfrs_svm_pipeline(
        FEATURES,
        selector_params={"resolve_overlaps": False, "random_state": 23},
        svm_params={"kernel": "linear", "C": 1.0},
    ).fit(X_train, y_train)
    np.testing.assert_array_equal(
        second.named_steps["gbfrs"].get_support(indices=True), support_before
    )


def test_fixed_seed_pipeline_predictions_are_reproducible():
    frame = synthetic_frame()
    train_positions, test_positions = outer_split(frame, 1)
    X_train = frame.iloc[train_positions][FEATURES]
    y_train = frame.iloc[train_positions].diagnosis.to_numpy()
    X_test = frame.iloc[test_positions][FEATURES]
    kwargs = {
        "feature_names": FEATURES,
        "selector_params": {"resolve_overlaps": False, "random_state": 29},
        "svm_params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    }
    first = make_gbfrs_svm_pipeline(**kwargs).fit(X_train, y_train)
    second = make_gbfrs_svm_pipeline(**kwargs).fit(X_train, y_train)
    np.testing.assert_array_equal(first.predict(X_test), second.predict(X_test))
    np.testing.assert_array_equal(
        first.named_steps["gbfrs"].get_support(indices=True),
        second.named_steps["gbfrs"].get_support(indices=True),
    )


def test_binary_metrics_and_pipeline_require_both_classes_when_required():
    with pytest.raises(ValueError, match="both 0 and 1"):
        binary_metrics([0, 0], [0, 0])
    frame = synthetic_frame(n_groups=4)
    with pytest.raises(ValueError, match="binary labels"):
        make_gbfrs_svm_pipeline(
            FEATURES,
            selector_params={"resolve_overlaps": False},
            svm_params={"kernel": "linear"},
        ).fit(frame[FEATURES], np.zeros(len(frame), dtype=int))


def test_inner_search_uses_protocol_seed_and_macro_f1():
    search = make_inner_search(FEATURES, outer_fold=3, n_jobs=1)
    assert search.scoring == "f1_macro"
    assert search.cv.random_state == 45
    assert search.refit is True
    assert search.n_jobs == 1
    assert PROTOCOL_ID == "BUSBRA-V1-OUTER5-INNER3-MACROF1"


def test_frozen_feature_validation_rejects_non_288_schema():
    frame = synthetic_frame()
    with pytest.raises(ValueError, match="288"):
        frozen_feature_columns(frame)
