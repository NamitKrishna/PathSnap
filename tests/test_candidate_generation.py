"""
test_candidate_generation.py
Unit tests for candidate road segment generation and perpendicular distance calculation.
"""

import pandas as pd
import numpy as np
import pytest
from src.coordinate_utils import CoordinateTransformer
from src.road_network import RoadNetwork
from src.candidate_generation import generate_candidates_for_trip
from src.data_loader import TripData


def test_perpendicular_distance_and_candidate_retrieval():
    trans = CoordinateTransformer.from_center(8.0, 50.0)
    
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "longitude": [8.0, 8.001, 8.0],
        "latitude": [50.0, 50.0, 50.001]
    })
    arcs_df = pd.DataFrame({
        "arc_id": [0, 1],
        "from_node": [0, 0],
        "to_node": [1, 2]
    })
    
    net = RoadNetwork(nodes_df, arcs_df, trans)
    net.build_spatial_index()
    
    # Coordinates of Arc 0 endpoints in UTM
    u_x, u_y = net.arc_ux[0], net.arc_uy[0]
    v_x, v_y = net.arc_vx[0], net.arc_vy[0]
    
    # Construct a point exactly 10.0 meters perpendicular to Arc 0 segment
    # Arc direction vector
    dx = v_x - u_x
    dy = v_y - u_y
    length = np.hypot(dx, dy)
    unit_dx = dx / length
    unit_dy = dy / length
    # Perpendicular normal vector (-unit_dy, unit_dx)
    perp_x = -unit_dy
    perp_y = unit_dx
    
    # Point at midpoint + 10.0m normal
    mid_x = (u_x + v_x) / 2.0
    mid_y = (u_y + v_y) / 2.0
    test_px = mid_x + 10.0 * perp_x
    test_py = mid_y + 10.0 * perp_y
    
    dist, norm_t, proj_x, proj_y = net.compute_projection_along_arc(test_px, test_py, 0)
    assert pytest.approx(dist, abs=1e-3) == 10.0
    assert pytest.approx(norm_t, abs=1e-3) == 0.5
    assert pytest.approx(proj_x, abs=1e-3) == mid_x
    assert pytest.approx(proj_y, abs=1e-3) == mid_y
    
    # Query candidates with radius 15m -> should retrieve Arc 0 (dist=10m)
    cands_15 = net.query_candidate_arcs(test_px, test_py, 15.0)
    assert len(cands_15) >= 1
    assert cands_15[0][0] == 0
    assert pytest.approx(cands_15[0][1], abs=1e-3) == 10.0
    
    # Query candidates with radius 5m -> should be empty
    cands_5 = net.query_candidate_arcs(test_px, test_py, 5.0)
    assert len(cands_5) == 0


def test_candidate_generation_pipeline():
    trans = CoordinateTransformer.from_center(8.0, 50.0)
    nodes_df = pd.DataFrame({
        "node_id": [0, 1],
        "longitude": [8.0, 8.001],
        "latitude": [50.0, 50.0]
    })
    arcs_df = pd.DataFrame({
        "arc_id": [0],
        "from_node": [0],
        "to_node": [1]
    })
    # GPS point near arc 0
    track_df = pd.DataFrame({
        "obs_id": [0],
        "longitude": [8.0005],
        "latitude": [50.0],
        "timestamp": [1.0]
    })
    route_df = pd.DataFrame({
        "step_idx": [0],
        "arc_id": [0]
    })
    trip_data = TripData("test_trip", "train", track_df, nodes_df, arcs_df, route_df)
    
    # Preprocess GPS
    from src.gps_preprocessing import preprocess_gps_track
    cleaned_track, _ = preprocess_gps_track(track_df, trans)
    
    net = RoadNetwork(nodes_df, arcs_df, trans)
    net.build_spatial_index()
    
    candidates_df, summary = generate_candidates_for_trip(
        trip_data, net, cleaned_track, config={"candidate_generation": {"search_radius_meters": 50.0}}
    )
    
    assert len(candidates_df) == 1
    assert candidates_df.iloc[0]["candidate_arc_id"] == 0
    assert bool(candidates_df.iloc[0]["is_ground_truth"]) is True
    assert summary["candidate_hit_rate"] == 1.0
    assert summary["candidate_miss_rate"] == 0.0
