"""
models.py – Model registry for PATHSNAP map-matching.

Each entry maps a model name to a (estimator, scaler) tuple.
If scaler is not None, the estimator is wrapped in a Pipeline(scaler → estimator).
If scaler is None, the estimator is used directly (tree-based models handle scaling).

Class imbalance (~8–10 % positive rate) is handled via:
- class_weight='balanced'  for sklearn classifiers
- scale_pos_weight         for XGBoost (set to ratio of neg/pos counts)

SCALE_POS_WEIGHT is estimated from the overall training set distribution.
It is overridden at training time with the actual computed ratio.
"""

import logging
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

logger = logging.getLogger(__name__)

# Approximate scale_pos_weight for XGBoost (neg/pos ratio); refined at runtime.
_DEFAULT_SCALE_POS_WEIGHT = 11.0


class FastKNN(KNeighborsClassifier):
    """
    KNeighborsClassifier with automatic reference subsampling for massive tabular datasets.
    Subsamples up to max_samples rows for index building to ensure fast training and prediction.
    """
    def __init__(self, n_neighbors: int = 15, max_samples: int = 15000, random_state: int = 42):
        super().__init__(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if len(X_arr) > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X_arr), size=self.max_samples, replace=False)
            return super().fit(X_arr[idx], y_arr[idx])
        return super().fit(X_arr, y_arr)


def _make_pipeline(estimator, scaler):
    """Wrap estimator in a Pipeline if a scaler is provided."""
    if scaler is not None:
        return Pipeline([("scaler", scaler), ("clf", estimator)])
    return estimator


def get_models(scale_pos_weight: float = _DEFAULT_SCALE_POS_WEIGHT) -> dict:
    """
    Return the model registry as {name: sklearn estimator / Pipeline}.

    Parameters
    ----------
    scale_pos_weight : neg/pos ratio for XGBoost (computed from actual data).
    """
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        xgb_available = False
        logger.warning("XGBoost not available – skipping XGBoost model.")

    registry = {
        "logistic_regression": _make_pipeline(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                C=1.0,
                random_state=42,
            ),
            StandardScaler(),
        ),
        "knn": _make_pipeline(
            FastKNN(
                n_neighbors=15,
                max_samples=15000,
                random_state=42,
            ),
            StandardScaler(),
        ),
        "decision_tree": _make_pipeline(
            DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=30,
                class_weight="balanced",
                random_state=42,
            ),
            None,
        ),
        "random_forest": _make_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=30,
                max_samples=0.20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            None,
        ),
    }

    if xgb_available:
        registry["xgboost"] = _make_pipeline(
            XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            ),
            None,
        )

    return registry


def compute_scale_pos_weight(y) -> float:
    """Compute neg/pos ratio for XGBoost from label array."""
    import numpy as np
    y = np.array(y)
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return _DEFAULT_SCALE_POS_WEIGHT
    ratio = n_neg / n_pos
    logger.info(f"scale_pos_weight computed: {ratio:.2f} ({n_neg} neg / {n_pos} pos)")
    return ratio
