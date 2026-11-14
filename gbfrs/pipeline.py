"""Leakage-safe GBFRS + SVM pipeline helpers.

The scaler and selector are ordinary fitted pipeline steps.  When the pipeline
is fitted inside a cross-validation split, neither step can see validation or
test rows.  The name-preserving scaler keeps the frozen CSV feature names
available to :class:`GBFRSSelector` after scaling.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

from .selector import GBFRSSelector


class NamedStandardScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that returns a DataFrame with the original names."""

    def __init__(self, feature_names: Sequence[str] | None = None) -> None:
        self.feature_names = feature_names

    def fit(self, X: Any, y: Any = None) -> "NamedStandardScaler":
        data = check_array(X, dtype=float, ensure_2d=True)
        if self.feature_names is None:
            names = getattr(X, "columns", None)
            if names is None:
                names = [f"x{i}" for i in range(data.shape[1])]
        else:
            names = list(self.feature_names)
        if len(names) != data.shape[1]:
            raise ValueError("feature_names must match the number of input columns")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")

        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.scaler_ = StandardScaler().fit(data)
        self.n_features_in_ = data.shape[1]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, ("scaler_", "feature_names_in_"))
        data = check_array(X, dtype=float, ensure_2d=True)
        if data.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted scaler")
        transformed = self.scaler_.transform(data)
        index = getattr(X, "index", None)
        return pd.DataFrame(transformed, columns=self.feature_names_in_, index=index)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
            if names.shape != self.feature_names_in_.shape or not np.array_equal(
                names, self.feature_names_in_
            ):
                raise ValueError("input_features do not match the fitted feature names")
        return self.feature_names_in_.copy()


def make_gbfrs_svm_pipeline(
    feature_names: Sequence[str] | None = None,
    *,
    selector_params: dict[str, Any] | None = None,
    svm_params: dict[str, Any] | None = None,
    memory: Any = None,
) -> Pipeline:
    """Create ``NamedStandardScaler -> GBFRSSelector -> SVC``.

    ``memory`` is optional joblib-compatible pipeline caching.  If used by a
    future grid search, the cache is scoped to the fitted training data and
    transformer parameters by scikit-learn's Pipeline implementation; it does
    not share fitted transformations across folds.
    """

    selector_kwargs = dict(selector_params or {})
    selector_kwargs.setdefault("random_state", 42)
    svm_kwargs = dict(svm_params or {})
    return Pipeline(
        steps=[
            ("scaler", NamedStandardScaler(feature_names=feature_names)),
            ("gbfrs", GBFRSSelector(**selector_kwargs)),
            ("svc", SVC(**svm_kwargs)),
        ],
        memory=memory,
    )


def selected_feature_names(fitted_pipeline: Pipeline) -> np.ndarray:
    """Return selected frozen feature names from a fitted pipeline."""

    check_is_fitted(fitted_pipeline.named_steps["gbfrs"], "support_")
    scaler = fitted_pipeline.named_steps["scaler"]
    selector = fitted_pipeline.named_steps["gbfrs"]
    names = scaler.get_feature_names_out()
    return np.asarray(names, dtype=object)[selector.get_support(indices=True)]
