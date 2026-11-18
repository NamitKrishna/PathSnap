"""
gps_preprocessing.py
GPS trajectory preprocessing, motion kinematics computation, and anomaly flagging.

Cleaning Decisions & Policies:
1. Observation ID Preservation: Original 0-based observation IDs ('obs_id') are preserved
   throughout. No GPS observation rows are arbitrarily dropped or interpolated, ensuring
   downstream map-matching models can evaluate every raw sensor reading without indexing shifts.
2. Anomaly Flagging over Blind Filtering: Rather than deleting noisy or outlier observations,
   we compute kinematic metrics (speed, acceleration, delta-t) and annotate each observation
   with explicit diagnostic flags (e.g., is_large_time_gap, is_unrealistic_speed, is_stationary).
3. Metric Kinematics: All spatial differences (distance, speed, acceleration) are computed
   in projected metric coordinates (UTM) using pyproj and Euclidean geometry, avoiding
   distortion from raw geographic degree differences.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from src.coordinate_utils import (
    CoordinateTransformer,
    calculate_bearing,
)


def preprocess_gps_track(
    track_df: pd.DataFrame,
    transformer: CoordinateTransformer,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process a raw GPS track DataFrame, project coordinates, compute movement kinematics,
    and generate anomaly diagnostics while preserving original observation IDs.
    
    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame with columns ['obs_id', 'longitude', 'latitude', 'timestamp']
    transformer : CoordinateTransformer
        Coordinate transformer instantiated for the trip's local UTM zone.
    config : dict, optional
        Preprocessing configuration dictionary.
        
    Returns
    -------
    cleaned_df : pd.DataFrame
        Enhanced DataFrame containing projected coords, dt, ds, speed, heading, and quality flags.
    cleaning_report : dict
        Summary statistics of the preprocessing operations.
    """
    cfg = config or {}
    gps_cfg = cfg.get("gps_cleaning", {})
    max_speed_kmh = gps_cfg.get("max_realistic_speed_kmh", 180.0)
    max_speed_ms = max_speed_kmh / 3.6
    stat_thresh_ms = gps_cfg.get("stationary_speed_threshold_ms", 0.5)
    gap_thresh_s = gps_cfg.get("flag_large_dt_threshold_seconds", 5.0)

    df = track_df.copy()
    n_pts = len(df)

    # 1. Project coordinates to UTM (meters)
    x_coords, y_coords = transformer.transform_to_projected(
        df["longitude"].values, df["latitude"].values
    )
    df["proj_x"] = x_coords
    df["proj_y"] = y_coords

    # 2. Compute time differences (dt)
    timestamps = df["timestamp"].values
    if n_pts > 1:
        dt = np.diff(timestamps, prepend=timestamps[0] + 1.0)
        # For the first point, default dt to 1.0s (nominal sampling)
        dt[0] = 1.0
    else:
        dt = np.array([1.0])
    df["delta_t"] = dt

    # 3. Compute spatial displacements (ds in meters)
    if n_pts > 1:
        dx = np.diff(x_coords, prepend=x_coords[0])
        dy = np.diff(y_coords, prepend=y_coords[0])
        ds = np.sqrt(dx ** 2 + dy ** 2)
        ds[0] = 0.0
    else:
        dx = np.array([0.0])
        dy = np.array([0.0])
        ds = np.array([0.0])
    df["delta_s"] = ds

    # 4. Compute instantaneous speed (m/s and km/h)
    # Safe division avoiding divide-by-zero if duplicate timestamp occurs
    safe_dt = np.where(dt > 0.0, dt, 1e-6)
    speed_ms = ds / safe_dt
    speed_ms[0] = 0.0 if n_pts <= 1 else (speed_ms[1] if n_pts > 1 else 0.0)
    df["speed_ms"] = speed_ms
    df["speed_kmh"] = speed_ms * 3.6

    # 5. Compute acceleration (m/s^2)
    if n_pts > 1:
        dv = np.diff(speed_ms, prepend=speed_ms[0])
        accel = dv / safe_dt
        accel[0] = 0.0
    else:
        accel = np.array([0.0])
    df["accel_ms2"] = accel

    # 6. Compute GPS heading (bearing in degrees [0, 360))
    headings = np.zeros(n_pts, dtype=np.float64)
    if n_pts > 1:
        # Backward headings for points 1 to N-1
        b_headings = calculate_bearing(x_coords[:-1], y_coords[:-1], x_coords[1:], y_coords[1:])
        headings[1:] = b_headings
        headings[0] = b_headings[0]  # First point takes forward heading
    df["gps_heading"] = headings

    # 7. Diagnostic flags
    df["is_large_time_gap"] = df["delta_t"] > gap_thresh_s
    df["is_unrealistic_speed"] = df["speed_ms"] > max_speed_ms
    df["is_stationary"] = (df["speed_ms"] < stat_thresh_ms) | (df["delta_s"] < 1.0)
    df["is_duplicate_timestamp"] = df["delta_t"] == 0.0

    cleaning_report = {
        "point_count": n_pts,
        "large_time_gaps_count": int(df["is_large_time_gap"].sum()),
        "unrealistic_speed_count": int(df["is_unrealistic_speed"].sum()),
        "stationary_count": int(df["is_stationary"].sum()),
        "duplicate_timestamps_count": int(df["is_duplicate_timestamp"].sum()),
        "mean_speed_kmh": float(df["speed_kmh"].mean()),
        "max_speed_kmh": float(df["speed_kmh"].max()),
        "total_distance_km": float(df["delta_s"].sum() / 1000.0),
        "total_duration_hours": float((timestamps[-1] - timestamps[0]) / 3600.0) if n_pts > 1 else 0.0,
    }

    return df, cleaning_report
