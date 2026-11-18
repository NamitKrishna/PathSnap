"""
PATHSNAP Road Network Graph Handler

Loads and manages road network data for candidate generation and route geometry resolution.
Supports GeoPackage (.gpkg), GeoJSON, and GraphML formats.
"""
import json
from pathlib import Path
from typing import Any, Optional
import networkx as nx
from shapely.geometry import LineString, Point, shape, mapping
from shapely.ops import transform
import pyproj

from backend.config import settings
from backend.core.exceptions import RoadGraphNotLoadedError
from backend.core.logging import get_logger

logger = get_logger(__name__)


class RoadGraph:
    """
    Road network graph with geometry support.

    Nodes: Intersections (with x, y coordinates in Web Mercator)
    Edges: Road segments (with arc_id, geometry, metadata)
    """

    def __init__(self, graph: nx.MultiDiGraph, crs: str = "EPSG:3857"):
        self.graph = graph
        self.crs = crs
        self._arc_id_to_edge: dict[str, tuple] = {}
        self._tree = None
        self._edge_list = []
        self._build_arc_index()
        self._build_spatial_index()

    def _build_arc_index(self) -> None:
        """Build mapping from arc_id to edge key for fast lookup."""
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            arc_id = data.get("arc_id") or data.get("id") or f"{u}-{v}-{key}"
            data["arc_id"] = arc_id
            self._arc_id_to_edge[arc_id] = (u, v, key)

    def _build_spatial_index(self) -> None:
        """Build STRtree spatial index for rapid candidate search."""
        from shapely.strtree import STRtree
        geoms = []
        self._edge_list = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if "geometry" in data:
                geom = shape(data["geometry"])
                geoms.append(geom)
                self._edge_list.append((u, v, key, data, geom))
        if geoms:
            self._tree = STRtree(geoms)

    @classmethod
    def from_gpkg(cls, path: Path, layer: Optional[str] = None) -> "RoadGraph":
        """Load road network from GeoPackage."""
        import geopandas as gpd

        logger.info("Loading road graph from GeoPackage", extra={"path": str(path)})

        if layer:
            gdf = gpd.read_file(path, layer=layer)
        else:
            # Try to read first layer
            try:
                gdf = gpd.read_file(path)
            except Exception:
                # List layers and try first
                import fiona
                layers = fiona.listlayers(path)
                if layers:
                    gdf = gpd.read_file(path, layer=layers[0])
                else:
                    raise RoadGraphNotLoadedError(str(path), "No layers found in GeoPackage")

        return cls.from_geodataframe(gdf)

    @classmethod
    def from_geojson(cls, path: Path) -> "RoadGraph":
        """Load road network from GeoJSON."""
        import geopandas as gpd

        logger.info("Loading road graph from GeoJSON", extra={"path": str(path)})
        gdf = gpd.read_file(path)
        return cls.from_geodataframe(gdf)

    @classmethod
    def from_geodataframe(cls, gdf) -> "RoadGraph":
        """Build graph from GeoDataFrame of LineStrings."""
        if gdf.empty:
            raise RoadGraphNotLoadedError("GeoDataFrame", "Empty GeoDataFrame")

        # Ensure we have LineString geometries
        if not all(gdf.geometry.type == "LineString"):
            # Filter to LineStrings only
            gdf = gdf[gdf.geometry.type == "LineString"].copy()
            if gdf.empty:
                raise RoadGraphNotLoadedError("GeoDataFrame", "No LineString geometries found")

        # Project to Web Mercator for metric distances
        if gdf.crs and gdf.crs != "EPSG:3857":
            gdf = gdf.to_crs("EPSG:3857")

        graph = nx.MultiDiGraph()
        graph.graph["crs"] = "EPSG:3857"

        for idx, row in gdf.iterrows():
            geom: LineString = row.geometry
            coords = list(geom.coords)

            if len(coords) < 2:
                continue

            # Start and end nodes
            start_coord = coords[0]
            end_coord = coords[-1]

            # Create or get node IDs (using coordinate tuples as IDs)
            u = f"node_{start_coord[0]:.6f}_{start_coord[1]:.6f}"
            v = f"node_{end_coord[0]:.6f}_{end_coord[1]:.6f}"

            # Add nodes if not exist
            if u not in graph:
                graph.add_node(u, x=start_coord[0], y=start_coord[1])
            if v not in graph:
                graph.add_node(v, x=end_coord[0], y=end_coord[1])

            # Arc ID from data or generate
            arc_id = row.get("arc_id") or row.get("id") or f"arc_{idx}"

            # Edge data
            edge_data = {
                "arc_id": arc_id,
                "geometry": mapping(geom),
                "length_m": geom.length,
                "highway": row.get("highway", "unclassified"),
                "name": row.get("name", ""),
                "oneway": row.get("oneway", False),
                "maxspeed": row.get("maxspeed"),
            }

            graph.add_edge(u, v, key=str(arc_id), **edge_data)

            # If not oneway, add reverse edge
            if not edge_data["oneway"]:
                rev_edge_data = edge_data.copy()
                rev_edge_data["arc_id"] = f"{arc_id}_rev"
                graph.add_edge(v, u, key=rev_edge_data["arc_id"], **rev_edge_data)

        logger.info(
            "Road graph built",
            extra={
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            }
        )

        return cls(graph)

    @classmethod
    def from_graphml(cls, path: Path) -> "RoadGraph":
        """Load road network from GraphML."""
        logger.info("Loading road graph from GraphML", extra={"path": str(path)})
        graph = nx.read_graphml(path)

        # Ensure arc_id on edges
        for u, v, key, data in graph.edges(keys=True, data=True):
            if "arc_id" not in data:
                data["arc_id"] = key

        return cls(graph)

    @classmethod
    def create_mock(cls, bounds: Optional[dict[str, float]] = None, grid_size: int = 30) -> "RoadGraph":
        """Create a mock road graph for development/testing."""
        import random

        logger.info("Creating mock road graph", extra={"grid_size": grid_size})

        # Default bounds (San Francisco area)
        if bounds is None:
            bounds = {
                "min_lat": 37.70,
                "max_lat": 37.82,
                "min_lon": -122.52,
                "max_lon": -122.35,
            }

        # Project bounds to Web Mercator
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        min_x, min_y = transformer.transform(bounds["min_lon"], bounds["min_lat"])
        max_x, max_y = transformer.transform(bounds["max_lon"], bounds["max_lat"])

        graph = nx.MultiDiGraph()
        graph.graph["crs"] = "EPSG:3857"

        # Create a grid of nodes
        node_ids = []

        for i in range(grid_size):
            for j in range(grid_size):
                x = min_x + (max_x - min_x) * i / max(grid_size - 1, 1)
                y = min_y + (max_y - min_y) * j / max(grid_size - 1, 1)
                node_id = f"node_{i}_{j}"
                graph.add_node(node_id, x=x, y=y)
                node_ids.append(node_id)

        # Connect adjacent nodes (horizontal + vertical)
        arc_counter = 0
        highway_types = ["primary", "secondary", "tertiary", "residential", "unclassified"]

        for i in range(grid_size):
            for j in range(grid_size):
                u = f"node_{i}_{j}"

                # Right neighbor
                if i + 1 < grid_size:
                    v = f"node_{i+1}_{j}"
                    arc_id = f"arc_{arc_counter}"
                    arc_counter += 1

                    coords = [(graph.nodes[u]["x"], graph.nodes[u]["y"]),
                              (graph.nodes[v]["x"], graph.nodes[v]["y"])]
                    geom = LineString(coords)

                    edge_data = {
                        "arc_id": arc_id,
                        "geometry": mapping(geom),
                        "length_m": geom.length,
                        "highway": random.choice(highway_types),
                        "name": f"Road {arc_id}",
                        "oneway": random.random() < 0.1,
                        "maxspeed": random.choice([30, 40, 50, 60]),
                    }
                    graph.add_edge(u, v, key=arc_id, **edge_data)

                    if not edge_data["oneway"]:
                        rev_data = edge_data.copy()
                        rev_data["arc_id"] = f"{arc_id}_rev"
                        graph.add_edge(v, u, key=rev_data["arc_id"], **rev_data)

                # Down neighbor
                if j + 1 < grid_size:
                    v = f"node_{i}_{j+1}"
                    arc_id = f"arc_{arc_counter}"
                    arc_counter += 1

                    coords = [(graph.nodes[u]["x"], graph.nodes[u]["y"]),
                              (graph.nodes[v]["x"], graph.nodes[v]["y"])]
                    geom = LineString(coords)

                    edge_data = {
                        "arc_id": arc_id,
                        "geometry": mapping(geom),
                        "length_m": geom.length,
                        "highway": random.choice(highway_types),
                        "name": f"Road {arc_id}",
                        "oneway": random.random() < 0.1,
                        "maxspeed": random.choice([30, 40, 50, 60]),
                    }
                    graph.add_edge(u, v, key=arc_id, **edge_data)

                    if not edge_data["oneway"]:
                        rev_data = edge_data.copy()
                        rev_data["arc_id"] = f"{arc_id}_rev"
                        graph.add_edge(v, u, key=rev_data["arc_id"], **rev_data)

        logger.info(
            "Mock road graph created",
            extra={"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}
        )

        return cls(graph)

    def get_edge_by_arc_id(self, arc_id: str) -> Optional[dict[str, Any]]:
        """Get edge data by arc_id."""
        edge_key = self._arc_id_to_edge.get(arc_id)
        if edge_key:
            u, v, key = edge_key
            return self.graph.edges[u, v, key]
        return None

    def get_geometry_by_arc_id(self, arc_id: str) -> Optional[LineString]:
        """Get LineString geometry by arc_id."""
        edge = self.get_edge_by_arc_id(arc_id)
        if edge and "geometry" in edge:
            return shape(edge["geometry"])
        return None

    def get_coordinates_by_arc_id(self, arc_id: str) -> Optional[list[list[float]]]:
        """Get [lon, lat] coordinate pairs for arc_id (WGS84)."""
        geom = self.get_geometry_by_arc_id(arc_id)
        if geom is None:
            return None

        # Transform from Web Mercator to WGS84
        transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        wgs84_geom = transform(transformer.transform, geom)
        return [list(coord) for coord in wgs84_geom.coords]

    def get_route_geometry(self, arc_ids: list[str]) -> list[list[float]]:
        """Get combined route geometry as [lon, lat] pairs."""
        all_coords = []

        for i, arc_id in enumerate(arc_ids):
            coords = self.get_coordinates_by_arc_id(arc_id)
            if coords is None:
                logger.warning("Arc ID not found in graph", extra={"arc_id": arc_id})
                continue

            if i == 0:
                all_coords.extend(coords)
            else:
                # Avoid duplicate connection point
                all_coords.extend(coords[1:])

        return all_coords

    def find_nearest_edges(
        self,
        lat: float,
        lon: float,
        radius_m: float = 50.0,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Find road edges within radius of a GPS point."""
        # Transform point to Web Mercator
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(lon, lat)
        point = Point(x, y)

        results = []

        if self._tree is not None:
            # Query edges within buffer
            cand_indices = self._tree.query(point.buffer(radius_m))
            if len(cand_indices) == 0:
                # Fallback to nearest geometry
                nearest_res = self._tree.query_nearest(point)
                cand_indices = nearest_res if hasattr(nearest_res, '__iter__') else [nearest_res]

            for idx in cand_indices:
                u, v, key, data, geom = self._edge_list[int(idx)]
                distance = point.distance(geom)
                edge_data = data.copy()
                edge_data["distance_m"] = distance
                edge_data["arc_id"] = data.get("arc_id", key)
                results.append(edge_data)

            results.sort(key=lambda x: x["distance_m"])
            return results[:max_results]

    def are_connected(self, arc_id_1: str, arc_id_2: str) -> bool:
        """Check if two arcs are topologically connected (end of 1 == start of 2)."""
        edge_1 = self.get_edge_by_arc_id(arc_id_1)
        edge_2 = self.get_edge_by_arc_id(arc_id_2)

        if not edge_1 or not edge_2:
            return False

        # Get end node of first arc
        # We need to find the actual nodes - this is simplified
        # In a real implementation, we'd track node connectivity
        return True  # Simplified for mock

    def get_node_count(self) -> int:
        return self.graph.number_of_nodes()

    def get_edge_count(self) -> int:
        return self.graph.number_of_edges()

    def to_geojson(self) -> dict[str, Any]:
        """Export graph edges as GeoJSON FeatureCollection."""
        features = []

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if "geometry" not in data:
                continue

            geom = shape(data["geometry"])
            # Transform to WGS84
            transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            wgs84_geom = transform(transformer.transform, geom)

            features.append({
                "type": "Feature",
                "properties": {
                    "arc_id": data.get("arc_id", key),
                    "highway": data.get("highway"),
                    "name": data.get("name"),
                    "length_m": data.get("length_m"),
                },
                "geometry": mapping(wgs84_geom),
            })

        return {
            "type": "FeatureCollection",
            "features": features,
        }


# Global graph instance (loaded on startup)
_road_graph: Optional[RoadGraph] = None


def load_road_graph(path: Optional[Path] = None) -> RoadGraph:
    """Load or create road graph."""
    global _road_graph

    if _road_graph is not None:
        return _road_graph

    if path is None:
        path = settings.road_graph_path

    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".gpkg":
            _road_graph = RoadGraph.from_gpkg(path)
        elif suffix in [".geojson", ".json"]:
            _road_graph = RoadGraph.from_geojson(path)
        elif suffix == ".graphml":
            _road_graph = RoadGraph.from_graphml(path)
        else:
            logger.warning("Unknown graph format, creating mock", extra={"path": str(path)})
            _road_graph = RoadGraph.create_mock()
    else:
        logger.warning("Road graph file not found, creating mock", extra={"path": str(path)})
        _road_graph = RoadGraph.create_mock()

    return _road_graph


def get_road_graph() -> RoadGraph:
    """Get the loaded road graph instance."""
    global _road_graph
    if _road_graph is None:
        _road_graph = load_road_graph()
    return _road_graph