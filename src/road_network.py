"""
road_network.py
Road network modeling, spatial indexing with Shapely STRtree,
and geometric arc projection.

CRITICAL:
Original 0-based node IDs and arc IDs are strictly preserved.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, box
from shapely.strtree import STRtree
from src.coordinate_utils import CoordinateTransformer, calculate_bearing


class RoadNetwork:
    """
    Representation of a trip's road network with projected metric geometry
    and spatial index for rapid candidate segment querying.
    """

    def __init__(
        self,
        nodes_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        transformer: CoordinateTransformer
    ):
        """
        Initialize and project road network nodes and arcs.
        """
        self.nodes = nodes_df
        self.arcs = arcs_df
        self.transformer = transformer

        # Project all node coordinates to UTM meters
        nx, ny = self.transformer.transform_to_projected(
            self.nodes["longitude"].values, self.nodes["latitude"].values
        )
        self.node_x = nx
        self.node_y = ny

        # Precompute node degrees
        n_nodes = len(self.nodes)
        self.in_degree = np.zeros(n_nodes, dtype=np.int32)
        self.out_degree = np.zeros(n_nodes, dtype=np.int32)
        np.add.at(self.out_degree, self.arcs["from_node"].values, 1)
        np.add.at(self.in_degree, self.arcs["to_node"].values, 1)

        # Precompute arc geometries, lengths, and bearings
        u = self.arcs["from_node"].values
        v = self.arcs["to_node"].values
        self.u_nodes = u
        self.v_nodes = v

        self.arc_ux = self.node_x[u]
        self.arc_uy = self.node_y[u]
        self.arc_vx = self.node_x[v]
        self.arc_vy = self.node_y[v]

        dx = self.arc_vx - self.arc_ux
        dy = self.arc_vy - self.arc_uy
        self.arc_lengths = np.sqrt(dx ** 2 + dy ** 2)
        self.arc_bearings = calculate_bearing(self.arc_ux, self.arc_uy, self.arc_vx, self.arc_vy)

        # Spatial index cache
        self.tree: Optional[STRtree] = None
        self.indexed_arc_ids: Optional[np.ndarray] = None
        self.indexed_lines: Optional[List[LineString]] = None

    def build_spatial_index(self, bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        """
        Build a Shapely STRtree spatial index.
        
        Parameters
        ----------
        bbox : tuple of (min_x, min_y, max_x, max_y), optional
            If provided, only arcs intersecting this bounding box will be indexed.
            Every arc retains its true original arc ID.
        """
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            # Bounding box filter on endpoints
            in_bbox = (
                ((self.arc_ux >= min_x) & (self.arc_ux <= max_x) & (self.arc_uy >= min_y) & (self.arc_uy <= max_y)) |
                ((self.arc_vx >= min_x) & (self.arc_vx <= max_x) & (self.arc_vy >= min_y) & (self.arc_vy <= max_y))
            )
            arc_indices = np.where(in_bbox)[0]
        else:
            arc_indices = np.arange(len(self.arcs), dtype=np.int64)

        self.indexed_arc_ids = arc_indices
        lines = [
            LineString([(self.arc_ux[aid], self.arc_uy[aid]), (self.arc_vx[aid], self.arc_vy[aid])])
            for aid in arc_indices
        ]
        self.indexed_lines = lines
        self.tree = STRtree(lines)

    def query_candidate_arcs(
        self, point_x: float, point_y: float, radius_meters: float
    ) -> List[Tuple[int, float]]:
        """
        Query candidate arcs within radius_meters of a projected GPS point.
        Calculates perpendicular distance to the road segment geometry, not just endpoints.
        
        Returns
        -------
        candidates : list of tuples
            Each tuple is (original_arc_id, perpendicular_distance_meters),
            sorted by ascending perpendicular distance.
        """
        if self.tree is None:
            raise RuntimeError("Spatial index has not been built. Call build_spatial_index() first.")

        pt = Point(point_x, point_y)
        # Query STRtree using circular buffer
        near_tree_indices = self.tree.query(pt.buffer(radius_meters))

        candidates = []
        for tree_idx in near_tree_indices:
            line_geom = self.indexed_lines[tree_idx]
            dist = pt.distance(line_geom)
            if dist <= radius_meters:
                orig_arc_id = int(self.indexed_arc_ids[tree_idx])
                candidates.append((orig_arc_id, dist))

        # Sort by distance
        candidates.sort(key=lambda c: c[1])
        return candidates

    def get_arc_properties(self, arc_id: int) -> Dict[str, Any]:
        """
        Get metadata and geometric properties of an arc by its original arc ID.
        """
        u = self.u_nodes[arc_id]
        v = self.v_nodes[arc_id]
        return {
            "arc_id": arc_id,
            "from_node": int(u),
            "to_node": int(v),
            "length_meters": float(self.arc_lengths[arc_id]),
            "bearing_degrees": float(self.arc_bearings[arc_id]),
            "from_node_out_degree": int(self.out_degree[u]),
            "to_node_in_degree": int(self.in_degree[v]),
        }

    def compute_projection_along_arc(
        self, point_x: float, point_y: float, arc_id: int
    ) -> Tuple[float, float, float]:
        """
        Compute projection of a point onto an arc segment.
        
        Returns
        -------
        perpendicular_distance : float
            Shortest Euclidean distance (meters) to the segment.
        normalized_position : float
            Position along the segment from 0.0 (from_node) to 1.0 (to_node).
        projected_x, projected_y : float
            Coordinates of the nearest projected point on the segment.
        """
        ux = self.arc_ux[arc_id]
        uy = self.arc_uy[arc_id]
        vx = self.arc_vx[arc_id]
        vy = self.arc_vy[arc_id]
        length = self.arc_lengths[arc_id]

        if length == 0.0:
            dist = float(np.hypot(point_x - ux, point_y - uy))
            return dist, 0.0, ux, uy

        # Vector from U to V
        dx = vx - ux
        dy = vy - uy

        # Vector from U to point
        px = point_x - ux
        py = point_y - uy

        # Projection factor t
        t = (px * dx + py * dy) / (length ** 2)
        t_clamped = max(0.0, min(1.0, t))

        proj_x = ux + t_clamped * dx
        proj_y = uy + t_clamped * dy

        dist = float(np.hypot(point_x - proj_x, point_y - proj_y))
        return dist, float(t_clamped), float(proj_x), float(proj_y)
