"""
candidate_generation.py
Spatial indexing and candidate road segment generation for map matching.

Implements modular candidate retrieval, perpendicular metric distance calculation,
directional alignment, kinematic feature attachment, and candidate miss-rate evaluation.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from shapely.geometry import Point
from src.coordinate_utils import calculate_heading_difference
from src.road_network import RoadNetwork
from src.data_loader import TripData


def align_reference_route_to_gps(
    cleaned_track_df: pd.DataFrame,
    route_df: Optional[pd.DataFrame],
    road_network: RoadNetwork,
    forward_window: int = 25,
    max_matching_dist: float = 60.0
) -> Optional[np.ndarray]:
    """
    Monotonically align a chronological reference route sequence with GPS observations.
    
    IMPORTANT:
    Route row numbers DO NOT correspond to GPS row numbers.
    The vehicle traverses arcs sequentially in time. This function tracks progress
    along the sequence of route arcs in metric space.
    
    Returns
    -------
    aligned_gt_arcs : np.ndarray of shape (N,) or None
        Ground truth arc ID associated with each GPS observation, or None if unlabelled.
    """
    if route_df is None or len(route_df) == 0:
        return None

    route_arc_ids = route_df["arc_id"].values
    n_route = len(route_arc_ids)
    n_pts = len(cleaned_track_df)

    tx = cleaned_track_df["proj_x"].values
    ty = cleaned_track_df["proj_y"].values

    aligned_gt = np.full(n_pts, -1, dtype=np.int64)
    curr_route_idx = 0

    for i in range(n_pts):
        gx = tx[i]
        gy = ty[i]

        win_start = max(0, curr_route_idx - 2)
        win_end = min(n_route, curr_route_idx + forward_window)

        best_arc = route_arc_ids[curr_route_idx]
        best_dist = float("inf")
        best_r_idx = curr_route_idx

        for r_idx in range(win_start, win_end):
            aid = route_arc_ids[r_idx]
            dist, _, _, _ = road_network.compute_projection_along_arc(gx, gy, aid)
            if dist < best_dist:
                best_dist = dist
                best_arc = aid
                best_r_idx = r_idx

        if best_dist <= max_matching_dist:
            curr_route_idx = max(curr_route_idx, best_r_idx)
            aligned_gt[i] = best_arc
        else:
            # If distance exceeds threshold, still assign closest local route arc
            aligned_gt[i] = best_arc

    return aligned_gt


def generate_candidates_for_trip(
    trip_data: TripData,
    road_network: RoadNetwork,
    cleaned_track_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate candidate road segments for each GPS observation in a trip.
    
    Parameters
    ----------
    trip_data : TripData
        Loaded trip data.
    road_network : RoadNetwork
        Spatial road network with initialized STRtree.
    cleaned_track_df : pd.DataFrame
        Preprocessed GPS track with kinematics.
    config : dict, optional
        Configuration dictionary.
        
    Returns
    -------
    candidates_df : pd.DataFrame
        DataFrame where each row is a (GPS observation, candidate arc) pair.
    summary : dict
        Candidate generation quality metrics including miss rate.
    """
    cfg = config or {}
    cand_cfg = cfg.get("candidate_generation", {})
    search_radius = float(cand_cfg.get("search_radius_meters", 50.0))
    forward_window = int(cand_cfg.get("alignment_forward_window", 25))
    max_cands = int(cand_cfg.get("max_candidates_per_point", 50))

    # Align reference route if available
    aligned_gt_arcs = align_reference_route_to_gps(
        cleaned_track_df, trip_data.route, road_network, forward_window=forward_window
    )

    records = []
    n_pts = len(cleaned_track_df)
    zero_cand_count = 0
    gt_in_candidates_count = 0
    gt_evaluated_count = 0

    obs_ids = cleaned_track_df["obs_id"].values
    timestamps = cleaned_track_df["timestamp"].values
    px = cleaned_track_df["proj_x"].values
    py = cleaned_track_df["proj_y"].values
    gps_headings = cleaned_track_df["gps_heading"].values
    speeds_kmh = cleaned_track_df["speed_kmh"].values
    delta_ts = cleaned_track_df["delta_t"].values
    delta_ss = cleaned_track_df["delta_s"].values
    is_stationaries = cleaned_track_df["is_stationary"].values

    for i in range(n_pts):
        oid = int(obs_ids[i])
        ts = float(timestamps[i])
        gx = float(px[i])
        gy = float(py[i])
        gh = float(gps_headings[i])
        sp = float(speeds_kmh[i])
        dt = float(delta_ts[i])
        ds = float(delta_ss[i])
        stat = bool(is_stationaries[i])

        gt_arc = int(aligned_gt_arcs[i]) if aligned_gt_arcs is not None else None

        # Query candidate arcs
        cands = road_network.query_candidate_arcs(gx, gy, search_radius)
        if len(cands) == 0:
            zero_cand_count += 1
            if gt_arc is not None:
                gt_evaluated_count += 1
            continue

        # Cap maximum candidates per point
        if len(cands) > max_cands:
            cands = cands[:max_cands]

        cand_arc_ids = [c[0] for c in cands]
        if gt_arc is not None:
            gt_evaluated_count += 1
            if gt_arc in cand_arc_ids:
                gt_in_candidates_count += 1

        for rank, (aid, perp_dist) in enumerate(cands):
            # Compute projection along arc
            _, norm_pos, proj_px, proj_py = road_network.compute_projection_along_arc(gx, gy, aid)
            arc_prop = road_network.get_arc_properties(aid)
            arc_bearing = arc_prop["bearing_degrees"]
            heading_diff = calculate_heading_difference(gh, arc_bearing)

            is_gt = (aid == gt_arc) if gt_arc is not None else np.nan

            records.append({
                "trip_id": trip_data.trip_id,
                "partition": trip_data.partition,
                "obs_id": oid,
                "timestamp": ts,
                "candidate_arc_id": aid,
                "candidate_rank": rank,
                "perp_distance": perp_dist,
                "norm_proj_pos": norm_pos,
                "heading_diff": heading_diff,
                "gps_heading": gh,
                "arc_bearing": arc_bearing,
                "speed_kmh": sp,
                "delta_t": dt,
                "delta_s": ds,
                "is_stationary": stat,
                "arc_length": arc_prop["length_meters"],
                "from_node": arc_prop["from_node"],
                "to_node": arc_prop["to_node"],
                "from_node_out_degree": arc_prop["from_node_out_degree"],
                "to_node_in_degree": arc_prop["to_node_in_degree"],
                "is_ground_truth": is_gt
            })

    candidates_df = pd.DataFrame(records)

    # Miss rate calculation
    if gt_evaluated_count > 0:
        candidate_hit_rate = float(gt_in_candidates_count / gt_evaluated_count)
        candidate_miss_rate = float(1.0 - candidate_hit_rate)
    else:
        candidate_hit_rate = np.nan
        candidate_miss_rate = np.nan

    summary = {
        "trip_id": trip_data.trip_id,
        "partition": trip_data.partition,
        "search_radius_meters": search_radius,
        "gps_points_count": n_pts,
        "total_candidates_generated": len(candidates_df),
        "mean_candidates_per_point": float(len(candidates_df) / n_pts) if n_pts > 0 else 0.0,
        "zero_candidate_points_count": zero_cand_count,
        "zero_candidate_ratio": float(zero_cand_count / n_pts) if n_pts > 0 else 0.0,
        "gt_evaluated_points": gt_evaluated_count,
        "gt_hit_count": gt_in_candidates_count,
        "candidate_hit_rate": candidate_hit_rate,
        "candidate_miss_rate": candidate_miss_rate
    }

    return candidates_df, summary
