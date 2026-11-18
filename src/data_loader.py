"""
data_loader.py
Robust loading of PATHSNAP trip files (.track, .nodes, .arcs, .route)
strictly preserving 0-based row index IDs for nodes and arcs.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class TripData:
    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: Optional[pd.DataFrame] = None


def load_track(track_path: str) -> pd.DataFrame:
    """
    Load a .track file containing GPS observations.
    
    Columns:
      1: longitude (float)
      2: latitude (float)
      3: timestamp (float in seconds)
      
    Returns DataFrame with:
      obs_id: 0-based integer observation ID (explicitly preserved)
      longitude: float
      latitude: float
      timestamp: float
    """
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Track file not found: {track_path}")
        
    df = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"],
        dtype={"longitude": np.float64, "latitude": np.float64, "timestamp": np.float64},
        engine="c"
    )
    df.insert(0, "obs_id", np.arange(len(df), dtype=np.int64))
    return df


def load_nodes(nodes_path: str) -> pd.DataFrame:
    """
    Load a .nodes file containing road network nodes.
    
    CRITICAL:
    The 0-based row index strictly represents the node ID.
    We preserve the original row index as an explicit 'node_id' column
    and maintain the DataFrame index as 0, 1, ..., N-1.
    Rows MUST NOT be deleted, sorted, or re-indexed.
    
    Columns:
      node_id: 0-based integer node ID
      longitude: float
      latitude: float
    """
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
        
    df = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"],
        dtype={"longitude": np.float64, "latitude": np.float64},
        engine="c"
    )
    df.insert(0, "node_id", np.arange(len(df), dtype=np.int64))
    return df


def load_arcs(arcs_path: str) -> pd.DataFrame:
    """
    Load a .arcs file containing directed road network edges.
    
    CRITICAL:
    The 0-based row index strictly represents the arc ID.
    We preserve the original row index as an explicit 'arc_id' column
    and maintain the DataFrame index as 0, 1, ..., M-1.
    Rows MUST NOT be deleted, sorted, or re-indexed.
    
    Columns:
      arc_id: 0-based integer arc ID
      from_node: integer ID of source node
      to_node: integer ID of target node
    """
    if not os.path.exists(arcs_path):
        raise FileNotFoundError(f"Arcs file not found: {arcs_path}")
        
    df = pd.read_csv(
        arcs_path,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={"from_node": np.int64, "to_node": np.int64},
        engine="c"
    )
    df.insert(0, "arc_id", np.arange(len(df), dtype=np.int64))
    return df


def load_route(route_path: str) -> Optional[pd.DataFrame]:
    """
    Load a .route file containing the chronological sequence of ground truth arc IDs.
    
    Returns None if the route file does not exist (e.g. for unlabelled or evaluation trips).
    
    CRITICAL:
    Route row numbers DO NOT correspond to GPS observation row numbers.
    Each row represents a traversed arc ID in the ground-truth path sequence.
    
    Columns:
      step_idx: 0-based sequence index of the traversed arc
      arc_id: integer arc ID referenced in .arcs
    """
    if not os.path.exists(route_path):
        return None
        
    df = pd.read_csv(
        route_path,
        sep=r"\s+",
        header=None,
        names=["arc_id"],
        dtype={"arc_id": np.int64},
        engine="c"
    )
    df.insert(0, "step_idx", np.arange(len(df), dtype=np.int64))
    return df


def load_trip(trip_dir: str, trip_id: str, partition: str = "train") -> TripData:
    """
    Load all files for a given trip into a TripData object.
    """
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    track_df = load_track(track_path)
    nodes_df = load_nodes(nodes_path)
    arcs_df = load_arcs(arcs_path)
    route_df = load_route(route_path)

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
        route=route_df
    )
