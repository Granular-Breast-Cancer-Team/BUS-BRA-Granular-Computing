
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DuplicateFeatureSelector(TransformerMixin, BaseEstimator):
    """Remove exact duplicate columns learned from training data."""

    def fit(self, X, y=None):
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("X must have two dimensions.")

        self.n_features_in_ = values.shape[1]
        self.keep_indices_ = []

        # Compare each column with previously retained columns.
        for column in range(self.n_features_in_):
            is_duplicate = any(
                np.array_equal(
                    values[:, column],
                    values[:, kept_column],
                )
                for kept_column in self.keep_indices_
            )

            if not is_duplicate:
                self.keep_indices_.append(column)

        self.n_features_out_ = len(self.keep_indices_)
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_indices_")
        values = np.asarray(X)

        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count does not match training.")

        return values[:, self.keep_indices_]
