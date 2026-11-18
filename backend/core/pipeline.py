"""
PATHSNAP Reconstruction Pipeline

Orchestrates the full pipeline: Member 1 → Member 2 → Member 3
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.api.models import (
    GPSPoint,
    ReconstructedRoute,
    ReconstructionDiagnostics,
    ReconstructionOptions,
    CandidateSegment,
)
from backend.integrations.member1 import (
    Preprocessor,
    CandidateGenerator,
    CandidateSet,
    create_preprocessor,
    create_candidate_generator,
)
from backend.integrations.member2 import MLModel, create_ml_model
from backend.integrations.member3 import HMVDecoder, create_hmm_decoder, ViterbiResult
from backend.data.road_graph import RoadGraph, get_road_graph
from backend.core.exceptions import PipelineError, ValidationError
from backend.core.logging import get_logger, LoggingContext
from backend.config import settings

logger = get_logger(__name__)


@dataclass
class PipelineContext:
    """Context passed through pipeline stages."""
    request_id: str
    gps_points: list[GPSPoint]
    options: ReconstructionOptions
    processed_points: list = field(default_factory=list)
    candidate_sets: list = field(default_factory=list)
    viterbi_result: Optional[ViterbiResult] = None
    road_graph: Optional[RoadGraph] = None
    diagnostics: dict = field(default_factory=dict)


class ReconstructionPipeline:
    """
    Main reconstruction pipeline orchestrating all members.
    """

    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        ml_model: Optional[MLModel] = None,
        hmm_decoder: Optional[HMVDecoder] = None,
        road_graph: Optional[RoadGraph] = None,
    ):
        self.preprocessor = preprocessor or create_preprocessor()
        self.candidate_generator = candidate_generator or create_candidate_generator()
        self.ml_model = ml_model or create_ml_model()
        self.hmm_decoder = hmm_decoder or create_hmm_decoder()
        self.road_graph = road_graph or get_road_graph()

        logger.info("Pipeline initialized", extra={
            "preprocessor": type(self.preprocessor).__name__,
            "candidate_generator": type(self.candidate_generator).__name__,
            "ml_model": type(self.ml_model).__name__,
            "hmm_decoder": type(self.hmm_decoder).__name__,
        })

    def reconstruct(
        self,
        gps_points: list[GPSPoint],
        options: Optional[ReconstructionOptions] = None,
        request_id: Optional[str] = None,
    ) -> tuple[ReconstructedRoute, ReconstructionDiagnostics]:
        """
        Run full reconstruction pipeline.

        Args:
            gps_points: List of GPS observations
            options: Optional reconstruction parameters
            request_id: Unique request identifier

        Returns:
            Tuple of (ReconstructedRoute, ReconstructionDiagnostics)

        Raises:
            PipelineError: If any stage fails
        """
        request_id = request_id or str(uuid.uuid4())[:8]
        options = options or ReconstructionOptions()

        with LoggingContext(logger, request_id=request_id):
            logger.info("Starting reconstruction", extra={"points": len(gps_points)})

            # Validate input
            self._validate_input(gps_points, options)

            # Initialize context
            ctx = PipelineContext(
                request_id=request_id,
                gps_points=gps_points,
                options=options,
                road_graph=self.road_graph,
            )

            try:
                # Stage 1: Preprocessing
                ctx = self._run_preprocessing(ctx)

                # Stage 2: Candidate Generation
                ctx = self._run_candidate_generation(ctx)

                # Stage 3: ML Model Prediction
                ctx = self._run_model_prediction(ctx)

                # Stage 4: HMM/Viterbi Decoding
                ctx = self._run_decoding(ctx)

                # Stage 5: Build Response
                route, diagnostics = self._build_response(ctx)

                logger.info("Reconstruction complete")
                return route, diagnostics

            except Exception as e:
                logger.error("Pipeline failed", extra={"error": str(e)})
                if isinstance(e, PipelineError):
                    raise
                raise PipelineError(
                    f"Pipeline failed: {e}",
                    failed_stage="unknown",
                ) from e

    def _validate_input(self, gps_points: list[GPSPoint], options: ReconstructionOptions) -> None:
        """Validate input parameters."""
        if len(gps_points) < settings.gps_min_points:
            raise ValidationError(
                f"At least {settings.gps_min_points} GPS points required",
                details={"provided": len(gps_points), "minimum": settings.gps_min_points}
            )

        if len(gps_points) > settings.gps_max_points:
            raise ValidationError(
                f"Maximum {settings.gps_max_points} GPS points allowed",
                details={"provided": len(gps_points), "maximum": settings.gps_max_points}
            )

        # Check for duplicate timestamps
        timestamps = [p.timestamp for p in gps_points]
        if len(timestamps) != len(set(timestamps)):
            raise ValidationError("Duplicate timestamps found in GPS points")

    def _run_preprocessing(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 1: Preprocess GPS points."""
        stage_start = time.perf_counter()

        with LoggingContext(logger, stage="preprocessing"):
            ctx.processed_points = self.preprocessor.preprocess(ctx.gps_points)

            duration = (time.perf_counter() - stage_start) * 1000
            ctx.diagnostics["preprocessing_time_ms"] = duration
            logger.info("Preprocessing done", extra={"duration_ms": duration, "points": len(ctx.processed_points)})

        return ctx

    def _run_candidate_generation(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 2: Generate candidate road segments."""
        stage_start = time.perf_counter()

        radius = ctx.options.candidate_radius_m or settings.candidate_search_radius_m
        max_cands = ctx.options.max_candidates_per_point or settings.hmm_max_candidates_per_point

        with LoggingContext(logger, stage="candidate_generation"):
            ctx.candidate_sets = self.candidate_generator.generate_candidates(
                processed_points=ctx.processed_points,
                road_graph=ctx.road_graph,
                radius_m=radius,
                max_candidates=max_cands,
            )

            duration = (time.perf_counter() - stage_start) * 1000
            total_cands = sum(len(cs.candidates) for cs in ctx.candidate_sets)
            ctx.diagnostics["candidate_generation_time_ms"] = duration
            ctx.diagnostics["candidate_stats"] = {
                "total_candidates": total_cands,
                "avg_per_point": total_cands / len(ctx.candidate_sets) if ctx.candidate_sets else 0,
                "points_with_candidates": len([cs for cs in ctx.candidate_sets if cs.candidates]),
                "points_without_candidates": len([cs for cs in ctx.candidate_sets if not cs.candidates]),
            }
            logger.info("Candidate generation done", extra={"duration_ms": duration, "total_candidates": total_cands})

        # Check for observations with no candidates
        empty_points = [i for i, cs in enumerate(ctx.candidate_sets) if not cs.candidates]
        if empty_points:
            logger.warning("Some points have no candidates", extra={"indices": empty_points})
            ctx.diagnostics.setdefault("warnings", []).append(
                f"{len(empty_points)} GPS point(s) have no candidate roads within search radius"
            )

        return ctx

    def _run_model_prediction(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 3: Run ML model to score candidates."""
        stage_start = time.perf_counter()

        with LoggingContext(logger, stage="model_prediction"):
            ctx.candidate_sets = self.ml_model.predict(ctx.candidate_sets)

            duration = (time.perf_counter() - stage_start) * 1000
            ctx.diagnostics["model_prediction_time_ms"] = duration

            # Log prediction statistics
            all_probs = []
            for cs in ctx.candidate_sets:
                for c in cs.candidates:
                    all_probs.append(c.probability)

            if all_probs:
                ctx.diagnostics["prediction_stats"] = {
                    "mean_probability": float(np.mean(all_probs)),
                    "max_probability": float(np.max(all_probs)),
                    "min_probability": float(np.min(all_probs)),
                }

            logger.info("Model prediction done", extra={"duration_ms": duration})

        return ctx

    def _run_decoding(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 4: Run HMM/Viterbi decoding."""
        stage_start = time.perf_counter()

        penalty = ctx.options.transition_penalty or settings.hmm_transition_penalty

        with LoggingContext(logger, stage="decoding"):
            ctx.viterbi_result = self.hmm_decoder.decode(
                candidate_sets=ctx.candidate_sets,
                road_graph=ctx.road_graph,
                transition_penalty=penalty,
            )

            duration = (time.perf_counter() - stage_start) * 1000
            ctx.diagnostics["decoding_time_ms"] = duration
            ctx.diagnostics["hmm_stats"] = {
                "best_path_score": ctx.viterbi_result.best_score,
                "path_length": len(ctx.viterbi_result.best_path),
                "avg_step_probability": float(np.mean(ctx.viterbi_result.path_probabilities)) if ctx.viterbi_result.path_probabilities else 0,
            }
            logger.info("Decoding done", extra={"duration_ms": duration, "path_length": len(ctx.viterbi_result.best_path)})

        return ctx

    def _build_response(self, ctx: PipelineContext) -> tuple[ReconstructedRoute, ReconstructionDiagnostics]:
        """Stage 5: Build final response objects."""
        vr = ctx.viterbi_result

        # Get route geometry
        coordinates = ctx.road_graph.get_route_geometry(vr.best_path)

        # Compute total distance
        total_distance = 0.0
        for arc_id in vr.best_path:
            edge = ctx.road_graph.get_edge_by_arc_id(arc_id)
            if edge and "length_m" in edge:
                total_distance += edge["length_m"]

        route = ReconstructedRoute(
            arc_ids=vr.best_path,
            coordinates=coordinates,
            total_distance_m=total_distance,
            point_count=len(ctx.gps_points),
            segment_count=len(vr.best_path),
        )

        diagnostics = ReconstructionDiagnostics(
            preprocessing_time_ms=ctx.diagnostics.get("preprocessing_time_ms", 0),
            candidate_generation_time_ms=ctx.diagnostics.get("candidate_generation_time_ms", 0),
            model_prediction_time_ms=ctx.diagnostics.get("model_prediction_time_ms", 0),
            decoding_time_ms=ctx.diagnostics.get("decoding_time_ms", 0),
            total_time_ms=sum(v for k, v in ctx.diagnostics.items() if k.endswith("_time_ms")),
            candidate_stats=ctx.diagnostics.get("candidate_stats", {}),
            hmm_stats=ctx.diagnostics.get("hmm_stats", {}),
            warnings=ctx.diagnostics.get("warnings", []),
        )

        return route, diagnostics


# Import numpy for statistics
import numpy as np

# Singleton pipeline instance
_pipeline: Optional[ReconstructionPipeline] = None


def get_pipeline() -> ReconstructionPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ReconstructionPipeline()
    return _pipeline