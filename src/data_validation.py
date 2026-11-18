"""
data_validation.py
Comprehensive data quality and integrity checks for GPS observations,
road networks, and reference routes.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from src.data_loader import TripData


def validate_gps_track(track_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate GPS trajectory observations.
    
    Checks:
      - Missing (NaN/null) values
      - Latitude / longitude bounds [-90, 90] and [-180, 180]
      - Duplicate observations (identical coordinates and timestamp)
      - Timestamp monotonicity and sorting
      - Timestamp gaps (sampling intervals dt)
    """
    cfg = config or {}
    gps_cfg = cfg.get("gps_cleaning", {})
    gap_thresh = gps_cfg.get("flag_large_dt_threshold_seconds", 5.0)

    results: Dict[str, Any] = {}
    n_pts = len(track_df)
    results["gps_point_count"] = n_pts
    results["gps_null_count"] = int(track_df.isnull().sum().sum())

    # Coordinates validity
    invalid_lon = ((track_df["longitude"] < -180.0) | (track_df["longitude"] > 180.0)).sum()
    invalid_lat = ((track_df["latitude"] < -90.0) | (track_df["latitude"] > 90.0)).sum()
    results["gps_invalid_coords_count"] = int(invalid_lon + invalid_lat)

    # Duplicates
    results["gps_duplicate_rows"] = int(track_df.duplicated(subset=["longitude", "latitude", "timestamp"]).sum())
    results["gps_duplicate_timestamps"] = int(track_df["timestamp"].duplicated().sum())

    # Timestamps ordering & intervals
    timestamps = track_df["timestamp"].values
    results["gps_is_time_monotonic"] = bool(track_df["timestamp"].is_monotonic_increasing)
    
    if n_pts > 1:
        dt = np.diff(timestamps)
        results["gps_dt_min"] = float(np.min(dt))
        results["gps_dt_max"] = float(np.max(dt))
        results["gps_dt_mean"] = float(np.mean(dt))
        results["gps_dt_median"] = float(np.median(dt))
        results["gps_dt_std"] = float(np.std(dt))
        results["gps_negative_dt_count"] = int((dt < 0).sum())
        results["gps_zero_dt_count"] = int((dt == 0).sum())
        results["gps_large_gaps_count"] = int((dt > gap_thresh).sum())
        results["gps_duration_seconds"] = float(timestamps[-1] - timestamps[0])
    else:
        results["gps_dt_min"] = 0.0
        results["gps_dt_max"] = 0.0
        results["gps_dt_mean"] = 0.0
        results["gps_dt_median"] = 0.0
        results["gps_dt_std"] = 0.0
        results["gps_negative_dt_count"] = 0
        results["gps_zero_dt_count"] = 0
        results["gps_large_gaps_count"] = 0
        results["gps_duration_seconds"] = 0.0

    # Bounding box
    results["gps_min_lon"] = float(track_df["longitude"].min())
    results["gps_max_lon"] = float(track_df["longitude"].max())
    results["gps_min_lat"] = float(track_df["latitude"].min())
    results["gps_max_lat"] = float(track_df["latitude"].max())

    results["gps_valid"] = bool(
        results["gps_null_count"] == 0
        and results["gps_invalid_coords_count"] == 0
        and results["gps_negative_dt_count"] == 0
    )
    return results


def validate_road_network(nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate road-network topology, node coordinates, and arc references.
    
    Checks:
      - Node coordinate bounds and nulls
      - Arc node references within [0, len(nodes) - 1]
      - Self loops (from_node == to_node)
      - Duplicate arcs (same from_node and to_node)
      - Degree distribution (isolated nodes: in-degree == 0 and out-degree == 0)
      - Graph connectivity: weakly connected components (WCC) and strongly connected components (SCC)
    """
    results: Dict[str, Any] = {}
    n_nodes = len(nodes_df)
    n_arcs = len(arcs_df)

    results["node_count"] = n_nodes
    results["arc_count"] = n_arcs
    results["node_null_count"] = int(nodes_df.isnull().sum().sum())
    results["arc_null_count"] = int(arcs_df.isnull().sum().sum())

    # Coordinate validity for nodes
    invalid_lon = ((nodes_df["longitude"] < -180.0) | (nodes_df["longitude"] > 180.0)).sum()
    invalid_lat = ((nodes_df["latitude"] < -90.0) | (nodes_df["latitude"] > 90.0)).sum()
    results["node_invalid_coords_count"] = int(invalid_lon + invalid_lat)

    # Arc node reference check
    min_ref = int(min(arcs_df["from_node"].min(), arcs_df["to_node"].min()))
    max_ref = int(max(arcs_df["from_node"].max(), arcs_df["to_node"].max()))
    results["arcs_min_node_ref"] = min_ref
    results["arcs_max_node_ref"] = max_ref
    results["arcs_all_nodes_exist"] = bool(min_ref >= 0 and max_ref < n_nodes)

    # Self-loops & duplicates
    results["arc_self_loops_count"] = int((arcs_df["from_node"] == arcs_df["to_node"]).sum())
    results["arc_duplicate_count"] = int(arcs_df.duplicated(subset=["from_node", "to_node"]).sum())

    # Graph connectivity and component analysis
    if results["arcs_all_nodes_exist"] and n_nodes > 0:
        # Build sparse adjacency matrix
        adj = sp.csr_matrix(
            (np.ones(n_arcs, dtype=np.int32), (arcs_df["from_node"].values, arcs_df["to_node"].values)),
            shape=(n_nodes, n_nodes)
        )
        in_degrees = np.array(adj.sum(axis=0)).flatten()
        out_degrees = np.array(adj.sum(axis=1)).flatten()
        isolated_nodes = np.where((in_degrees == 0) & (out_degrees == 0))[0]
        results["isolated_nodes_count"] = int(len(isolated_nodes))
        results["dead_end_nodes_count"] = int((out_degrees == 0).sum() - len(isolated_nodes))

        # Connected components
        n_wcc, labels_wcc = connected_components(adj, directed=False)
        n_scc, labels_scc = connected_components(adj, directed=True)
        results["wcc_count"] = int(n_wcc)
        results["scc_count"] = int(n_scc)

        # Size of largest weakly connected component
        if n_wcc > 0:
            wcc_sizes = np.bincount(labels_wcc)
            results["largest_wcc_size"] = int(np.max(wcc_sizes))
            results["largest_wcc_ratio"] = float(np.max(wcc_sizes) / n_nodes)
        else:
            results["largest_wcc_size"] = 0
            results["largest_wcc_ratio"] = 0.0
    else:
        results["isolated_nodes_count"] = -1
        results["dead_end_nodes_count"] = -1
        results["wcc_count"] = -1
        results["scc_count"] = -1
        results["largest_wcc_size"] = -1
        results["largest_wcc_ratio"] = -1.0

    results["network_valid"] = bool(
        results["node_null_count"] == 0
        and results["arc_null_count"] == 0
        and results["node_invalid_coords_count"] == 0
        and results["arcs_all_nodes_exist"]
    )
    return results


def validate_reference_route(
    route_df: Optional[pd.DataFrame], arcs_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate reference route path and its topological alignment with road arcs.
    
    Checks:
      - Route file presence
      - Route nulls
      - Arc IDs valid in [0, len(arcs) - 1]
      - Consecutive arc transition connectivity in arc graph
      - Route length and loop/repetition count
    """
    results: Dict[str, Any] = {}
    if route_df is None:
        results["route_available"] = False
        results["route_step_count"] = 0
        results["route_null_count"] = 0
        results["route_all_arcs_exist"] = False
        results["route_is_continuous"] = False
        results["route_connected_transitions_ratio"] = 0.0
        results["route_unique_arcs_count"] = 0
        results["route_valid"] = False
        return results

    results["route_available"] = True
    n_steps = len(route_df)
    results["route_step_count"] = n_steps
    results["route_null_count"] = int(route_df.isnull().sum().sum())

    n_arcs = len(arcs_df)
    min_arc = int(route_df["arc_id"].min())
    max_arc = int(route_df["arc_id"].max())
    results["route_min_arc_ref"] = min_arc
    results["route_max_arc_ref"] = max_arc
    results["route_all_arcs_exist"] = bool(min_arc >= 0 and max_arc < n_arcs)

    # Route continuity along the directed arc network
    if results["route_all_arcs_exist"] and n_steps > 1:
        route_arc_ids = route_df["arc_id"].values
        from_nodes = arcs_df["from_node"].values
        to_nodes = arcs_df["to_node"].values
        
        # Check u_to == v_from
        curr_to = to_nodes[route_arc_ids[:-1]]
        next_from = from_nodes[route_arc_ids[1:]]
        connected_mask = (curr_to == next_from)
        connected_transitions = int(connected_mask.sum())
        total_transitions = n_steps - 1
        results["route_total_transitions"] = total_transitions
        results["route_connected_transitions"] = connected_transitions
        results["route_is_continuous"] = bool(connected_transitions == total_transitions)
        results["route_connected_transitions_ratio"] = float(connected_transitions / total_transitions)
    else:
        results["route_total_transitions"] = max(0, n_steps - 1)
        results["route_connected_transitions"] = max(0, n_steps - 1)
        results["route_is_continuous"] = True
        results["route_connected_transitions_ratio"] = 1.0

    results["route_unique_arcs_count"] = int(route_df["arc_id"].nunique())
    results["route_valid"] = bool(
        results["route_null_count"] == 0
        and results["route_all_arcs_exist"]
    )
    return results


def validate_trip(trip_data: TripData, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run full validation suite on a TripData instance.
    """
    gps_res = validate_gps_track(trip_data.track, config)
    net_res = validate_road_network(trip_data.nodes, trip_data.arcs)
    rte_res = validate_reference_route(trip_data.route, trip_data.arcs)

    summary = {
        "trip_id": trip_data.trip_id,
        "partition": trip_data.partition,
        **gps_res,
        **net_res,
        **rte_res,
        "overall_trip_valid": bool(
            gps_res["gps_valid"] and net_res["network_valid"] and (not rte_res["route_available"] or rte_res["route_valid"])
        )
    }
    return summary
