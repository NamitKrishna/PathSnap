"""
feature_engineering.py
Modular feature engineering for map matching ML models (ranking, classification, HMMs).
Transforms geometric candidate projections into normalized features.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def engineer_candidate_features(
    candidates_df: pd.DataFrame,
    gps_measurement_std_meters: float = 4.07,
    heading_std_degrees: float = 30.0
) -> pd.DataFrame:
    """
    Compute derived machine learning features for each (GPS observation, candidate arc) pair.
    
    Features engineered:
      - log_perp_distance: log(1 + perp_distance)
      - emission_prob_gaussian: Standard Newson-Krumm HMM emission likelihood
      - heading_diff_cos: cosine of heading difference in radians
      - heading_prob_gaussian: Gaussian directional compatibility
      - log_arc_length: log(1 + arc_length)
      - is_heading_aligned: binary flag (heading_diff <= 45 degrees)
      - is_heading_opposed: binary flag (heading_diff >= 135 degrees)
      - distance_to_best_ratio: ratio of perp_distance to the closest candidate's distance
    """
    if candidates_df.empty:
        return candidates_df.copy()

    df = candidates_df.copy()

    # 1. Distance transform & log
    df["log_perp_distance"] = np.log1p(df["perp_distance"])

    # 2. Gaussian emission probability (Newson & Krumm 2009)
    sigma_z = gps_measurement_std_meters
    df["emission_prob_gaussian"] = np.exp(-0.5 * (df["perp_distance"] / sigma_z) ** 2) / (np.sqrt(2 * np.pi) * sigma_z)

    # 3. Directional features
    rad_diff = np.radians(df["heading_diff"])
    df["heading_diff_cos"] = np.cos(rad_diff)
    df["heading_prob_gaussian"] = np.exp(-0.5 * (df["heading_diff"] / heading_std_degrees) ** 2)

    df["is_heading_aligned"] = (df["heading_diff"] <= 45.0).astype(int)
    df["is_heading_opposed"] = (df["heading_diff"] >= 135.0).astype(int)

    # 4. Arc structural properties
    df["log_arc_length"] = np.log1p(df["arc_length"])

    # 5. Relative distance within observation candidate set
    # Compute min distance per obs_id
    min_dist_per_obs = df.groupby("obs_id")["perp_distance"].transform("min")
    df["relative_dist_diff"] = df["perp_distance"] - min_dist_per_obs
    df["distance_rank_ratio"] = df["perp_distance"] / (min_dist_per_obs + 1e-3)

    return df
