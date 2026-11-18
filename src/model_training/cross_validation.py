"""
cross_validation.py – Group-aware cross-validation for PATHSNAP.

Uses GroupKFold (default n_splits=5) with groups=trip_id so that all
observations from one trip are always in the same fold.  This prevents
trip-level data leakage between train and validation folds.

The module also provides observation-level Top-1 accuracy as a custom
scorer, which is more meaningful for map-matching than raw AUC.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)

logger = logging.getLogger(__name__)

N_SPLITS = 5


def _obs_top1_accuracy(y_true, y_proba, groups_series):
    """
    Observation-level Top-1 accuracy:
    For each obs_id, pick the candidate with the highest predicted probability.
    Score = fraction of observations where the top-1 candidate is the ground truth.

    Parameters
    ----------
    y_true       : array-like (0/1 labels)
    y_proba      : array-like (positive class probability)
    groups_series: pd.Series with obs_id for each row (not trip_id here)

    Returns float in [0, 1].
    """
    df = pd.DataFrame({
        "obs_id": groups_series.values,
        "y_true": np.array(y_true),
        "score": np.array(y_proba),
    })
    # Pick candidate with highest score per observation
    idx_top1 = df.groupby("obs_id")["score"].idxmax()
    top1 = df.loc[idx_top1]
    return top1["y_true"].mean()


def run_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    obs_ids: pd.Series | None = None,
    n_splits: int = N_SPLITS,
) -> dict:
    """
    Run group-aware cross-validation and return a dict of metric arrays.

    Parameters
    ----------
    model    : fitted sklearn estimator or Pipeline
    X        : feature DataFrame
    y        : label Series (0/1)
    groups   : trip_id Series (used for GroupKFold)
    obs_ids  : obs_id Series (used for Top-1 accuracy, optional)
    n_splits : number of CV folds

    Returns
    -------
    dict mapping metric name → np.array of per-fold scores
    """
    gkf = GroupKFold(n_splits=n_splits)

    try:
        roc_scorer = make_scorer(roc_auc_score, response_method="predict_proba")
        ap_scorer  = make_scorer(average_precision_score, response_method="predict_proba")
    except TypeError:
        roc_scorer = make_scorer(roc_auc_score, needs_proba=True)
        ap_scorer  = make_scorer(average_precision_score, needs_proba=True)

    scoring = {
        "f1":          make_scorer(f1_score, zero_division=0),
        "precision":   make_scorer(precision_score, zero_division=0),
        "recall":      make_scorer(recall_score, zero_division=0),
        "roc_auc":     roc_scorer,
        "avg_prec":    ap_scorer,
    }

    logger.info(f"Running {n_splits}-fold GroupKFold CV ...")
    cv_results = cross_validate(
        model, X, y,
        groups=groups,
        cv=gkf,
        scoring=scoring,
        return_train_score=False,
        return_estimator=True,
        n_jobs=1,
        error_score="raise",
    )

    # Compute observation-level Top-1 accuracy using the fold estimators
    if obs_ids is not None:
        top1_scores = []
        X_arr = X.values
        y_arr = y.values
        obs_arr = obs_ids.values
        groups_arr = groups.values

        for fold_idx, (_, val_idx) in enumerate(gkf.split(X_arr, y_arr, groups_arr)):
            m = cv_results["estimator"][fold_idx]
            if hasattr(m, "predict_proba"):
                probas = m.predict_proba(X_arr[val_idx])[:, 1]
            else:
                probas = m.decision_function(X_arr[val_idx])
            top1 = _obs_top1_accuracy(
                y_arr[val_idx], probas,
                pd.Series(obs_arr[val_idx])
            )
            top1_scores.append(top1)
        cv_results["test_obs_top1_acc"] = np.array(top1_scores)

    del cv_results["estimator"]
    return cv_results


def _clone_and_fit(model, X_train, y_train):
    """Clone and refit a model (used for manual fold computation)."""
    from sklearn.base import clone
    m = clone(model)
    m.fit(X_train, y_train)
    return m


def summarise_cv(cv_results: dict) -> dict:
    """Return mean ± std for each CV metric."""
    summary = {}
    for key, vals in cv_results.items():
        if key.startswith("test_"):
            metric = key[5:]
            summary[f"{metric}_mean"] = float(np.mean(vals))
            summary[f"{metric}_std"] = float(np.std(vals))
    return summary
