"""Independent paper-based implementation of granular-ball fuzzy-rough selection."""

from .selector import (
    GBFRSSelector,
    GranularBall,
    build_granular_balls,
    fuzzy_lower_approximations,
    heterogeneous_overlap_pairs,
    resolve_heterogeneous_overlaps,
    weighted_fuzzy_dependency,
)
from .pipeline import NamedStandardScaler, make_gbfrs_svm_pipeline, selected_feature_names
from .submission import validate_submission_templates
from .experiment import (
    run_outer_fold,
    write_private_fold1_outputs,
    write_private_fold_outputs,
)
from .protocol import (
    PROTOCOL_ID,
    binary_metrics,
    common_svm_grid,
    frozen_feature_columns,
    inner_cv,
    make_inner_search,
    outer_split,
)

__all__ = [
    "GBFRSSelector",
    "GranularBall",
    "build_granular_balls",
    "fuzzy_lower_approximations",
    "heterogeneous_overlap_pairs",
    "resolve_heterogeneous_overlaps",
    "weighted_fuzzy_dependency",
    "NamedStandardScaler",
    "make_gbfrs_svm_pipeline",
    "selected_feature_names",
    "validate_submission_templates",
    "run_outer_fold",
    "write_private_fold_outputs",
    "write_private_fold1_outputs",
    "PROTOCOL_ID",
    "binary_metrics",
    "common_svm_grid",
    "frozen_feature_columns",
    "inner_cv",
    "make_inner_search",
    "outer_split",
]
