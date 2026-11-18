"""
metrics.py – Evaluation metrics for PATHSNAP map-matching.

Metrics are computed at two levels:

1. Candidate level  – standard binary classification metrics
2. Observation level – Top-1 accuracy: for each GPS observation pick the
   candidate with the highest predicted score and check if it matches the
   ground-truth arc.

Route-level continuity is also checked: given the sequence of predicted
arc IDs per trip, verify how many consecutive arc pairs are topologically
connected (i.e., to_node of arc_i == from_node of arc_{i+1}).
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate-level metrics
# ---------------------------------------------------------------------------

def compute_candidate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Standard binary classification metrics at candidate level.

    Parameters
    ----------
    y_true  : ground-truth labels (0/1)
    y_pred  : predicted binary labels
    y_proba : predicted probabilities for positive class (optional)
    threshold : decision threshold (default 0.5)

    Returns dict of metric name → scalar.
    """
    metrics = {
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["avg_precision"] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics["avg_precision"] = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tp"] = int(tp)

    return metrics


# ---------------------------------------------------------------------------
# Observation-level Top-1 accuracy
# ---------------------------------------------------------------------------

def compute_obs_level_top1_accuracy(predictions_df: pd.DataFrame) -> float:
    """
    For each (trip_id, obs_id) group, pick the candidate with the highest
    predicted score and check whether it matches is_ground_truth.

    Required columns in predictions_df:
        trip_id, obs_id, candidate_arc_id, score, is_ground_truth

    Returns
    -------
    float – fraction of observations correctly matched (Top-1 accuracy)
    """
    required = {"trip_id", "obs_id", "candidate_arc_id", "score", "is_ground_truth"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions_df missing columns: {missing}")

    df = predictions_df.copy()
    df["is_ground_truth"] = df["is_ground_truth"].astype(int)

    # For each observation pick the top-scoring candidate
    idx_top1 = df.groupby(["trip_id", "obs_id"])["score"].idxmax()
    top1 = df.loc[idx_top1]
    accuracy = top1["is_ground_truth"].mean()
    n_obs = len(top1)
    n_correct = top1["is_ground_truth"].sum()
    logger.info(f"Top-1 accuracy: {n_correct}/{n_obs} = {accuracy:.4f}")
    return float(accuracy)


def compute_per_trip_top1(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-trip Top-1 accuracy as a DataFrame."""
    rows = []
    for trip_id, grp in predictions_df.groupby("trip_id"):
        acc = compute_obs_level_top1_accuracy(grp)
        rows.append({"trip_id": trip_id, "top1_accuracy": acc,
                     "n_observations": grp["obs_id"].nunique()})
    return pd.DataFrame(rows).sort_values("trip_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Route continuity
# ---------------------------------------------------------------------------

def compute_route_continuity(
    predictions_df: pd.DataFrame,
    arc_lookup: dict[int, dict],  # arc_id → {"from_node": int, "to_node": int}
) -> dict:
    """
    Given the sequence of predicted arc IDs (top-1 per obs, ordered by timestamp),
    check topological continuity: to_node[arc_i] == from_node[arc_{i+1}].

    Parameters
    ----------
    predictions_df : must contain trip_id, obs_id, timestamp, candidate_arc_id, score
    arc_lookup     : {arc_id: {"from_node": int, "to_node": int}}

    Returns
    -------
    dict: overall continuity fraction and per-trip breakdown
    """
    df = predictions_df.copy()
    idx_top1 = df.groupby(["trip_id", "obs_id"])["score"].idxmax()
    top1 = df.loc[idx_top1].sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

    total_transitions = 0
    continuous_transitions = 0
    per_trip = []

    for trip_id, grp in top1.groupby("trip_id"):
        arc_seq = grp["candidate_arc_id"].tolist()
        n_cont = 0
        n_total = len(arc_seq) - 1
        for i in range(n_total):
            a1, a2 = arc_seq[i], arc_seq[i + 1]
            if a1 in arc_lookup and a2 in arc_lookup:
                if arc_lookup[a1]["to_node"] == arc_lookup[a2]["from_node"]:
                    n_cont += 1
        total_transitions += n_total
        continuous_transitions += n_cont
        per_trip.append({
            "trip_id": trip_id,
            "transitions": n_total,
            "continuous": n_cont,
            "continuity_pct": 100 * n_cont / n_total if n_total > 0 else 0.0,
        })

    overall = (
        100 * continuous_transitions / total_transitions
        if total_transitions > 0 else 0.0
    )
    return {
        "overall_continuity_pct": overall,
        "total_transitions": total_transitions,
        "continuous_transitions": continuous_transitions,
        "per_trip": pd.DataFrame(per_trip),
    }


# ---------------------------------------------------------------------------
# Candidate miss rate
# ---------------------------------------------------------------------------

def candidate_miss_rate(predictions_df: pd.DataFrame) -> float:
    """
    Fraction of observations where NO candidate has is_ground_truth=True.
    This measures the ceiling imposed by candidate generation recall.
    """
    df = predictions_df.copy()
    df["is_ground_truth"] = df["is_ground_truth"].astype(int)
    obs_has_gt = df.groupby(["trip_id", "obs_id"])["is_ground_truth"].max()
    miss_rate = (obs_has_gt == 0).mean()
    logger.info(f"Candidate miss rate: {miss_rate:.4f} "
                f"({(obs_has_gt == 0).sum()} / {len(obs_has_gt)} observations)")
    return float(miss_rate)
