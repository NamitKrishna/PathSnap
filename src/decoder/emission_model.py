"""
emission_model.py – Emission probability model for PATHSNAP HMM.

Converts candidate-level classification/ranking scores (from Random Forest)
into normalized observation-conditional emission probabilities:
    P(z_t | c_{t,k})
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


class EmissionModel:
    """
    Computes and normalizes emission probabilities from candidate scores.
    
    Safe probability normalization:
    - Clips NaN, negative, and infinite values
    - Adds epsilon smoothing to prevent zero emission probability
    - Normalizes scores within each observation's candidate set: sum(p_k) = 1.0
    - Converts to log-space with a safe lower bound (-100.0)
    """

    def __init__(self, epsilon: float = 1e-6, log_floor: float = -100.0):
        self.epsilon = epsilon
        self.log_floor = log_floor

    def normalize_scores(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize raw scores into probabilities and log-probabilities.

        Parameters
        ----------
        scores : 1D array of candidate scores (from ML model, in [0, 1])

        Returns
        -------
        probs : 1D array of normalized probabilities (sums to 1.0)
        log_probs : 1D array of log-probabilities
        """
        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) == 0:
            return np.array([]), np.array([])

        # Handle NaNs and non-finite values
        clean_scores = np.nan_to_num(scores, nan=self.epsilon, posinf=1.0, neginf=self.epsilon)
        clean_scores = np.maximum(clean_scores, self.epsilon)

        total = np.sum(clean_scores)
        if total > 0.0:
            probs = clean_scores / total
        else:
            probs = np.full_like(clean_scores, 1.0 / len(clean_scores))

        log_probs = np.log(np.maximum(probs, 1e-12))
        log_probs = np.maximum(log_probs, self.log_floor)

        return probs, log_probs

    def process_observation_candidates(
        self, candidate_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Process candidate DataFrame for a single observation.

        Returns dict with:
        - candidate_arc_ids: array of arc IDs
        - probs: normalized emission probabilities
        - log_probs: log emission probabilities
        - norm_proj_pos: normalized position along arc [0, 1]
        - arc_length: length of each candidate arc in meters
        - perp_distance: perpendicular distance to segment in meters
        """
        if len(candidate_df) == 0:
            return {
                "candidate_arc_ids": np.array([], dtype=np.int64),
                "probs": np.array([]),
                "log_probs": np.array([]),
                "norm_proj_pos": np.array([]),
                "arc_length": np.array([]),
                "perp_distance": np.array([]),
            }

        scores = candidate_df["score"].values if "score" in candidate_df.columns else np.ones(len(candidate_df))
        probs, log_probs = self.normalize_scores(scores)

        arc_ids = candidate_df["candidate_arc_id"].values.astype(np.int64)
        pos = candidate_df["norm_proj_pos"].values.astype(np.float64) if "norm_proj_pos" in candidate_df.columns else np.full(len(arc_ids), 0.5)
        length = candidate_df["arc_length"].values.astype(np.float64) if "arc_length" in candidate_df.columns else np.full(len(arc_ids), 50.0)
        perp_dist = candidate_df["perp_distance"].values.astype(np.float64) if "perp_distance" in candidate_df.columns else np.zeros(len(arc_ids))

        return {
            "candidate_arc_ids": arc_ids,
            "probs": probs,
            "log_probs": log_probs,
            "norm_proj_pos": pos,
            "arc_length": length,
            "perp_distance": perp_dist,
        }
