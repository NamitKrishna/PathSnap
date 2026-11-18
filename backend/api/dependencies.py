"""
PATHSNAP API Dependencies

FastAPI dependency injection for shared resources.
"""
from functools import lru_cache
from backend.config import settings
from backend.data.road_graph import get_road_graph, RoadGraph
from backend.core.pipeline import get_pipeline, ReconstructionPipeline
from backend.integrations.member1 import create_preprocessor, create_candidate_generator
from backend.integrations.member2 import create_ml_model
from backend.integrations.member3 import create_hmm_decoder


@lru_cache()
def get_cached_road_graph() -> RoadGraph:
    """Get cached road graph instance."""
    return get_road_graph()


@lru_cache()
def get_cached_pipeline() -> ReconstructionPipeline:
    """Get cached pipeline instance."""
    return get_pipeline()


# For testing: allow overriding dependencies
def override_road_graph(graph: RoadGraph):
    """Override road graph for testing."""
    get_cached_road_graph.cache_clear()
    # Note: In real tests, you'd use FastAPI's dependency_overrides


def override_pipeline(pipeline: ReconstructionPipeline):
    """Override pipeline for testing."""
    get_cached_pipeline.cache_clear()