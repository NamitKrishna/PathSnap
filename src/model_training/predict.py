"""
predict.py – Inference utilities for PATHSNAP map-matching.

Generates candidate-level prediction scores using a trained model.
Preserves all ID columns and returns a ranked prediction DataFrame
ready for downstream HMM route reconstruction.
"""

import logging
import numpy as np
import pandas as pd

from .data_loader_ml import FEATURE_COLS, ID_COLS, LABEL_COL

logger = logging.getLogger(__name__)


def generate_predictions(
    model,
    features_df: pd.DataFrame,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Run inference on a feature DataFrame and produce ranked predictions.

    Parameters
    ----------
    model       : fitted sklearn estimator (may be a Pipeline with scaler)
    features_df : candidate feature DataFrame (must include ID_COLS + FEATURE_COLS)
    model_name  : string tag added to output for bookkeeping

    Returns
    -------
    pd.DataFrame with columns:
        trip_id, obs_id, candidate_arc_id, timestamp, candidate_rank,
        score, rank_by_score, is_ground_truth (if present), model
    Rows are sorted by trip_id, obs_id, rank_by_score.
    """
    df = features_df.copy()

    # Verify required columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for inference: {missing}")

    X = df[FEATURE_COLS]

    # Predict probability of positive class
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[:, 1]
    else:
        # Fallback: use decision function normalised to [0, 1]
        raw = model.decision_function(X)
        probas = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        logger.warning(f"{model_name}: using decision_function instead of predict_proba")

    df["score"] = probas
    df["model"] = model_name

    # Rank candidates within each observation (1 = best)
    df["rank_by_score"] = (
        df.groupby(["trip_id", "obs_id"])["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Keep ID cols + score + rank + label (if available)
    keep_cols = ["trip_id", "obs_id", "candidate_arc_id", "timestamp",
                 "candidate_rank", "score", "rank_by_score", "model"]
    if LABEL_COL in df.columns:
        keep_cols.append(LABEL_COL)

    out = df[keep_cols].sort_values(
        ["trip_id", "obs_id", "rank_by_score"]
    ).reset_index(drop=True)

    logger.info(
        f"Predictions generated: {len(out):,} candidates, "
        f"{out['obs_id'].nunique():,} observations, "
        f"{out['trip_id'].nunique()} trips"
    )
    return out


def top1_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the top-1 (highest-score) candidate per observation.

    Returns a DataFrame with one row per (trip_id, obs_id).
    Columns include candidate_arc_id, score, is_ground_truth (if present).
    """
    idx = predictions_df.groupby(["trip_id", "obs_id"])["score"].idxmax()
    return predictions_df.loc[idx].reset_index(drop=True)
