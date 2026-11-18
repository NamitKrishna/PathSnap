"""
test_coordinate_utils.py
Unit tests for coordinate transformations, projections, and bearings.
"""

import pytest
import numpy as np
from src.coordinate_utils import (
    get_utm_epsg,
    CoordinateTransformer,
    calculate_bearing,
    calculate_heading_difference,
    calculate_geodesic_distance
)


def test_get_utm_epsg():
    # Darmstadt, Germany (~8.6E, 49.8N) -> UTM Zone 32N -> EPSG:32632
    assert get_utm_epsg(8.6, 49.8) == 32632
    # Alabama, USA (~-86.6W, 34.7N) -> UTM Zone 16N -> EPSG:32616
    assert get_utm_epsg(-86.6, 34.7) == 32616
    # Sydney, Australia (~151.2E, -33.9S) -> UTM Zone 56S -> EPSG:32756
    assert get_utm_epsg(151.2, -33.9) == 32756


def test_coordinate_transformer_roundtrip():
    lon, lat = 8.595203, 49.844053
    trans = CoordinateTransformer.from_center(lon, lat)
    
    x, y = trans.transform_to_projected(lon, lat)
    assert isinstance(x, (float, np.floating))
    assert isinstance(y, (float, np.floating))
    
    back_lon, back_lat = trans.transform_to_wgs84(x, y)
    assert pytest.approx(lon, abs=1e-6) == back_lon
    assert pytest.approx(lat, abs=1e-6) == back_lat


def test_calculate_bearing():
    # Due North: (0, 0) to (0, 10) -> 0 degrees
    assert pytest.approx(calculate_bearing(0, 0, 0, 10), abs=1e-4) == 0.0
    # Due East: (0, 0) to (10, 0) -> 90 degrees
    assert pytest.approx(calculate_bearing(0, 0, 10, 0), abs=1e-4) == 90.0
    # Due South: (0, 0) to (0, -10) -> 180 degrees
    assert pytest.approx(calculate_bearing(0, 0, 0, -10), abs=1e-4) == 180.0
    # Due West: (0, 0) to (-10, 0) -> 270 degrees
    assert pytest.approx(calculate_bearing(0, 0, -10, 0), abs=1e-4) == 270.0


def test_calculate_heading_difference():
    # Same heading
    assert pytest.approx(calculate_heading_difference(45.0, 45.0)) == 0.0
    # Perpendicular
    assert pytest.approx(calculate_heading_difference(0.0, 90.0)) == 90.0
    # Opposite
    assert pytest.approx(calculate_heading_difference(0.0, 180.0)) == 180.0
    # Wraparound across 0/360: 10 deg and 350 deg -> difference is 20 deg
    assert pytest.approx(calculate_heading_difference(10.0, 350.0)) == 20.0
    assert pytest.approx(calculate_heading_difference(355.0, 5.0)) == 10.0


def test_geodesic_distance():
    # Approx 1 degree latitude at equator is ~111 km
    dist = calculate_geodesic_distance(0.0, 0.0, 0.0, 1.0)
    assert 110000.0 < dist < 112000.0
