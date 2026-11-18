"""
data_loader_ml.py – Load feature CSV files for ML training.

Design decisions:
- Files are loaded trip-by-trip and concatenated to manage memory.
- is_ground_truth (bool) is cast to int (0/1) label.
- Rows where is_ground_truth is NaN are dropped.
- ID columns (trip_id, obs_id, candidate_arc_id) are preserved throughout.
- Feature columns are a fixed 19-column set matching feature_engineering.py output.
"""

import os
import glob
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Feature column definitions
# ------------------------------------------------------------------
FEATURE_COLS = [
    "perp_distance",
    "norm_proj_pos",
    "heading_diff",
    "speed_kmh",
    "delta_t",
    "delta_s",
    "is_stationary",          # bool → int at load time
    "arc_length",
    "from_node_out_degree",
    "to_node_in_degree",
    "log_perp_distance",
    "emission_prob_gaussian",
    "heading_diff_cos",
    "heading_prob_gaussian",
    "is_heading_aligned",
    "is_heading_opposed",
    "log_arc_length",
    "relative_dist_diff",
    "distance_rank_ratio",
]

ID_COLS = ["trip_id", "obs_id", "candidate_arc_id", "timestamp", "candidate_rank"]
LABEL_COL = "is_ground_truth"


def load_feature_file(csv_path: str) -> pd.DataFrame:
    """Load a single candidate feature CSV and return a clean DataFrame."""
    df = pd.read_csv(csv_path)

    # Drop rows with no label
    before = len(df)
    df = df.dropna(subset=[LABEL_COL])
    dropped = before - len(df)
    if dropped:
        logger.debug(f"{os.path.basename(csv_path)}: dropped {dropped} NaN-label rows")

    # Cast label to int
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Cast boolean feature to int
    if df["is_stationary"].dtype == bool:
        df["is_stationary"] = df["is_stationary"].astype(int)

    # Verify all required columns present
    missing = [c for c in FEATURE_COLS + ID_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    return df


def load_all_features(
    partition: str,
    feature_dir: str = "data/features",
    max_trips: int | None = None,
    trip_ids: list[int] | None = None,
) -> pd.DataFrame:
    """
    Load all candidate feature files for a given partition.

    Parameters
    ----------
    partition : 'train' | 'student_test'
    feature_dir : root directory for feature CSVs
    max_trips : optional limit on number of trips (for debugging)
    trip_ids : optional explicit list of trip IDs to load

    Returns
    -------
    pd.DataFrame with columns: ID_COLS + FEATURE_COLS + [LABEL_COL]
    """
    pattern = os.path.join(feature_dir, partition, "*_candidates.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No feature files found at: {pattern}")

    if trip_ids is not None:
        trip_id_set = {int(t) for t in trip_ids}
        files = [f for f in files
                 if int(os.path.basename(f).replace("_candidates.csv", "")) in trip_id_set]

    if max_trips is not None:
        files = files[:max_trips]

    logger.info(f"Loading {len(files)} feature files for partition={partition}")

    frames = []
    for f in files:
        try:
            df = load_feature_file(f)
            frames.append(df)
            logger.debug(f"  {os.path.basename(f)}: {len(df)} rows, "
                         f"{df[LABEL_COL].sum()} positives")
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    if not frames:
        raise RuntimeError("No feature data loaded.")

    combined = pd.concat(frames, ignore_index=True)
    n_pos = combined[LABEL_COL].sum()
    n_total = len(combined)
    logger.info(
        f"Loaded {n_total:,} candidate rows, {combined['obs_id'].nunique():,} observations, "
        f"{combined['trip_id'].nunique()} trips | "
        f"positive rate = {100 * n_pos / n_total:.2f}%"
    )
    return combined


def get_X_y_groups(df: pd.DataFrame):
    """
    Split DataFrame into feature matrix X, label vector y, and group array.

    Returns
    -------
    X : pd.DataFrame (FEATURE_COLS only)
    y : pd.Series (int 0/1)
    groups : pd.Series (trip_id, used for GroupKFold)
    """
    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(int)
    groups = df["trip_id"]
    return X, y, groups
