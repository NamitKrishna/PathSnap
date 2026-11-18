"""
PATHSNAP Member 1 Integration Adapter

Interfaces for preprocessing and candidate generation.
Replace mock implementations with actual Member 1 code when available.
"""
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from shapely.geometry import shape
from backend.api.models import GPSPoint, CandidateSegment
from backend.data.road_graph import RoadGraph, get_road_graph
from backend.core.exceptions import PreprocessingError, CandidateGenerationError
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedGPSPoint:
    """GPS point after preprocessing."""
    original: GPSPoint
    projected_x: float  # Web Mercator X
    projected_y: float  # Web Mercator Y
    speed_smoothed: Optional[float] = None
    heading_smoothed: Optional[float] = None
    features: dict[str, float] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}


@dataclass
class CandidateSet:
    """Candidate road segments for a single GPS point."""
    gps_point: ProcessedGPSPoint
    candidates: list[CandidateSegment]


class Preprocessor(ABC):
    """Abstract preprocessor interface."""

    @abstractmethod
    def preprocess(self, gps_points: list[GPSPoint]) -> list[ProcessedGPSPoint]:
        """Preprocess raw GPS points."""
        pass


class CandidateGenerator(ABC):
    """Abstract candidate generator interface."""

    @abstractmethod
    def generate_candidates(
        self,
        processed_points: list[ProcessedGPSPoint],
        road_graph: RoadGraph,
        radius_m: float,
        max_candidates: int,
    ) -> list[CandidateSet]:
        """Generate candidate road segments for each GPS point."""
        pass


# --- Mock Implementations (Replace with Member 1's actual code) ---

class MockPreprocessor(Preprocessor):
    """
    Mock preprocessor for development.

    Member 1 should implement:
    - Outlier removal
    - Speed/heading smoothing
    - Map projection (WGS84 -> Web Mercator)
    - Feature extraction (e.g., acceleration, jerk, turn angles)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.projector = None
        self._init_projector()

    def _init_projector(self):
        try:
            import pyproj
            self.projector = pyproj.Transformer.from_crs(
                "EPSG:4326", "EPSG:3857", always_xy=True
            )
        except ImportError:
            logger.warning("pyproj not available, using approximate projection")

    def preprocess(self, gps_points: list[GPSPoint]) -> list[ProcessedGPSPoint]:
        start_time = time.perf_counter()
        logger.info("Preprocessing GPS points", extra={"count": len(gps_points)})

        processed = []

        for i, point in enumerate(gps_points):
            # Project to Web Mercator
            if self.projector:
                x, y = self.projector.transform(point.longitude, point.latitude)
            else:
                # Approximate (not accurate, for mock only)
                x = point.longitude * 111320 * np.cos(np.radians(point.latitude))
                y = point.latitude * 111320

            # Simple speed smoothing (moving average)
            speed_smoothed = point.speed
            if point.speed is not None and i > 0 and i < len(gps_points) - 1:
                prev_speed = gps_points[i - 1].speed or 0
                next_speed = gps_points[i + 1].speed or 0
                speed_smoothed = (prev_speed + point.speed + next_speed) / 3

            # Simple heading smoothing
            heading_smoothed = point.heading
            if point.heading is not None and i > 0 and i < len(gps_points) - 1:
                prev_heading = gps_points[i - 1].heading or 0
                next_heading = gps_points[i + 1].heading or 0
                # Handle circular mean
                headings = [h for h in [prev_heading, point.heading, next_heading] if h is not None]
                if headings:
                    sin_sum = sum(np.sin(np.radians(h)) for h in headings)
                    cos_sum = sum(np.cos(np.radians(h)) for h in headings)
                    heading_smoothed = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360

            # Extract features
            features = {
                "speed": point.speed or 0,
                "speed_smoothed": speed_smoothed or 0,
                "heading": point.heading or 0,
                "heading_smoothed": heading_smoothed or 0,
                "accuracy": point.accuracy or 0,
            }

            # Add temporal features
            if i > 0:
                dt = (point.timestamp - gps_points[i - 1].timestamp).total_seconds()
                features["dt"] = max(dt, 0.001)
                if speed_smoothed and dt > 0:
                    features["acceleration"] = (speed_smoothed - (gps_points[i - 1].speed or 0)) / dt
                else:
                    features["acceleration"] = 0

            processed.append(ProcessedGPSPoint(
                original=point,
                projected_x=x,
                projected_y=y,
                speed_smoothed=speed_smoothed,
                heading_smoothed=heading_smoothed,
                features=features,
            ))

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Preprocessing complete", extra={"duration_ms": duration_ms, "count": len(processed)})

        return processed


class MockCandidateGenerator(CandidateGenerator):
    """
    Mock candidate generator for development.

    Member 1 should implement:
    - Spatial index (R-tree / STRtree) for fast nearest-neighbor search
    - Candidate filtering by road class, one-way, etc.
    - Distance and heading difference computation
    - Feature computation for each candidate (for Member 2)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def generate_candidates(
        self,
        processed_points: list[ProcessedGPSPoint],
        road_graph: RoadGraph,
        radius_m: float,
        max_candidates: int,
    ) -> list[CandidateSet]:
        start_time = time.perf_counter()
        logger.info(
            "Generating candidates",
            extra={"points": len(processed_points), "radius_m": radius_m, "max_candidates": max_candidates}
        )

        candidate_sets = []

        for processed_point in processed_points:
            # Find nearby edges using road graph
            nearby = road_graph.find_nearest_edges(
                lat=processed_point.original.latitude,
                lon=processed_point.original.longitude,
                radius_m=radius_m,
                max_results=max_candidates,
            )

            candidates = []
            for edge in nearby:
                # Compute heading difference
                heading_diff = None
                if processed_point.heading_smoothed is not None:
                    # Get edge bearing
                    geom = shape(edge["geometry"])
                    coords = list(geom.coords)
                    if len(coords) >= 2:
                        dx = coords[-1][0] - coords[0][0]
                        dy = coords[-1][1] - coords[0][1]
                        edge_bearing = np.degrees(np.arctan2(dx, dy)) % 360
                        diff = abs(processed_point.heading_smoothed - edge_bearing)
                        heading_diff = min(diff, 360 - diff)

                candidate = CandidateSegment(
                    arc_id=edge["arc_id"],
                    probability=0.0,  # Will be filled by Member 2
                    distance_m=edge["distance_m"],
                    heading_diff_deg=heading_diff,
                    road_class=edge.get("highway"),
                    geometry=edge.get("geometry"),
                )
                candidates.append(candidate)

            candidate_sets.append(CandidateSet(
                gps_point=processed_point,
                candidates=candidates,
            ))

        duration_ms = (time.perf_counter() - start_time) * 1000
        total_candidates = sum(len(cs.candidates) for cs in candidate_sets)
        logger.info(
            "Candidate generation complete",
            extra={"duration_ms": duration_ms, "total_candidates": total_candidates}
        )

        return candidate_sets


# --- Factory Functions ---

def create_preprocessor(config: Optional[dict] = None) -> Preprocessor:
    """Create preprocessor instance. Replace with Member 1's implementation."""
    # TODO: Replace with actual Member 1 implementation
    # from member1.preprocessing import ProductionPreprocessor
    # return ProductionPreprocessor(config)
    return MockPreprocessor(config)


def create_candidate_generator(config: Optional[dict] = None) -> CandidateGenerator:
    """Create candidate generator instance. Replace with Member 1's implementation."""
    # TODO: Replace with actual Member 1 implementation
    # from member1.candidates import ProductionCandidateGenerator
    # return ProductionCandidateGenerator(config)
    return MockCandidateGenerator(config)