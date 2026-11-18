"""
test_data_validation.py
Unit tests for data validation logic.
"""

import pandas as pd
import numpy as np
import pytest
from src.data_validation import validate_gps_track, validate_road_network, validate_reference_route


def test_validate_gps_track_valid():
    df = pd.DataFrame({
        "obs_id": [0, 1, 2],
        "longitude": [8.595, 8.596, 8.597],
        "latitude": [49.844, 49.845, 49.846],
        "timestamp": [1.0, 2.0, 3.0]
    })
    res = validate_gps_track(df)
    assert res["gps_valid"] is True
    assert res["gps_point_count"] == 3
    assert res["gps_null_count"] == 0
    assert res["gps_is_time_monotonic"] is True


def test_validate_gps_track_invalid_coords_and_time():
    df = pd.DataFrame({
        "obs_id": [0, 1],
        "longitude": [8.595, 200.0],  # Invalid longitude > 180
        "latitude": [49.844, 49.845],
        "timestamp": [3.0, 1.0]      # Non-monotonic timestamp
    })
    res = validate_gps_track(df)
    assert res["gps_valid"] is False
    assert res["gps_invalid_coords_count"] > 0
    assert res["gps_negative_dt_count"] == 1


def test_validate_road_network():
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "longitude": [8.1, 8.2, 8.3],
        "latitude": [49.1, 49.2, 49.3]
    })
    # Valid arcs
    arcs_df = pd.DataFrame({
        "arc_id": [0, 1],
        "from_node": [0, 1],
        "to_node": [1, 2]
    })
    res = validate_road_network(nodes_df, arcs_df)
    assert res["network_valid"] is True
    assert res["arcs_all_nodes_exist"] is True
    assert res["wcc_count"] == 1

    # Invalid arc reference (references node 5 which does not exist)
    arcs_invalid = pd.DataFrame({
        "arc_id": [0],
        "from_node": [0],
        "to_node": [5]
    })
    res_inv = validate_road_network(nodes_df, arcs_invalid)
    assert res_inv["network_valid"] is False
    assert res_inv["arcs_all_nodes_exist"] is False


def test_validate_reference_route():
    arcs_df = pd.DataFrame({
        "arc_id": [0, 1, 2],
        "from_node": [0, 1, 2],
        "to_node": [1, 2, 3]
    })
    route_df = pd.DataFrame({
        "step_idx": [0, 1],
        "arc_id": [0, 1]
    })
    res = validate_reference_route(route_df, arcs_df)
    assert res["route_valid"] is True
    assert res["route_is_continuous"] is True

    # Route referencing nonexistent arc
    route_invalid = pd.DataFrame({
        "step_idx": [0],
        "arc_id": [99]
    })
    res_inv = validate_reference_route(route_invalid, arcs_df)
    assert res_inv["route_valid"] is False
    assert res_inv["route_all_arcs_exist"] is False
