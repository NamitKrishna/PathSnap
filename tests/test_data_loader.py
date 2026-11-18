"""
test_data_loader.py
Unit tests verifying strict preservation of 0-based node and arc IDs.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from src.data_loader import load_nodes, load_arcs, load_track, load_route


def test_load_nodes_preserves_0_indexed_ids():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".nodes") as f:
        f.write("8.581679  49.857880\n")
        f.write("8.584637  49.858295\n")
        f.write("8.580173  49.858567\n")
        temp_path = f.name

    try:
        df = load_nodes(temp_path)
        assert len(df) == 3
        assert list(df["node_id"]) == [0, 1, 2]
        assert list(df.index) == [0, 1, 2]
        assert "longitude" in df.columns
        assert "latitude" in df.columns
        # Ensure row 0 has node_id 0 and correct coords
        assert pytest.approx(df.at[0, "longitude"]) == 8.581679
        assert pytest.approx(df.at[2, "latitude"]) == 49.858567
    finally:
        os.remove(temp_path)


def test_load_arcs_preserves_0_indexed_ids():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".arcs") as f:
        f.write("0  281\n")
        f.write("0  1\n")
        f.write("1  0\n")
        temp_path = f.name

    try:
        df = load_arcs(temp_path)
        assert len(df) == 3
        assert list(df["arc_id"]) == [0, 1, 2]
        assert list(df.index) == [0, 1, 2]
        assert df.at[0, "from_node"] == 0
        assert df.at[0, "to_node"] == 281
        assert df.at[1, "to_node"] == 1
    finally:
        os.remove(temp_path)


def test_load_track_preserves_obs_id():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".track") as f:
        f.write("8.595203  49.844053  1.000000\n")
        f.write("8.595294  49.844441  2.000000\n")
        temp_path = f.name

    try:
        df = load_track(temp_path)
        assert len(df) == 2
        assert list(df["obs_id"]) == [0, 1]
        assert "timestamp" in df.columns
        assert df.at[1, "timestamp"] == 2.0
    finally:
        os.remove(temp_path)


def test_load_route():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".route") as f:
        f.write("2641\n")
        f.write("64973\n")
        f.write("133948\n")
        temp_path = f.name

    try:
        df = load_route(temp_path)
        assert len(df) == 3
        assert list(df["step_idx"]) == [0, 1, 2]
        assert list(df["arc_id"]) == [2641, 64973, 133948]
    finally:
        os.remove(temp_path)
