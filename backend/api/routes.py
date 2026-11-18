"""
PATHSNAP API Routes

FastAPI endpoints for health checks and route reconstruction.
"""
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from backend.api.models import (
    ErrorResponse,
    FileUploadResponse,
    GPSFileFormat,
    GPSPointInput,
    HealthResponse,
    ReconstructionRequest,
    ReconstructionResponse,
    ReconstructionStatus,
    EvaluationTripSummary,
    EvaluationTripDetail,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationListResponse,
)
from backend.core.exceptions import (
    PathSnapError,
    FileFormatError,
    FileSizeError,
    ValidationError,
    PipelineError,
    ResourceNotFoundError,
)
from backend.core.logging import get_logger, LoggingContext
from backend.core.pipeline import get_pipeline, ReconstructionPipeline
from backend.data.gps_parser import parse_gps_file, compute_bounds
from backend.data.evaluation import get_evaluation_trips, get_evaluation_trip, get_evaluation_summaries, calculate_accuracy_metrics
from backend.config import settings

logger = get_logger(__name__)

router = APIRouter()


# --- Dependency Injection ---
def get_reconstruction_pipeline() -> ReconstructionPipeline:
    """Dependency for getting the pipeline instance."""
    return get_pipeline()


# --- Health Check ---
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health and component availability",
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    pipeline = get_pipeline()

    components = {
        "pipeline": "healthy",
        "preprocessor": "healthy",
        "candidate_generator": "healthy",
        "ml_model": "healthy" if pipeline.ml_model else "unavailable",
        "hmm_decoder": "healthy",
        "road_graph": "healthy" if pipeline.road_graph else "unavailable",
    }

    # Check model file
    if not settings.model_path.exists():
        components["ml_model"] = "missing_model_file"

    # Check road graph
    if not settings.road_graph_path.exists():
        components["road_graph"] = "missing_graph_file"

    overall_status = "healthy"
    if any(v in ["unavailable", "missing_model_file", "missing_graph_file"] for v in components.values()):
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        components=components,
    )


# --- File Upload & Validation ---
@router.post(
    "/upload",
    response_model=FileUploadResponse,
    summary="Upload and validate GPS file",
    description="Upload a GPS track file (CSV, GPX, JSON) for validation before reconstruction",
)
async def upload_gps_file(
    file: UploadFile = File(...),
) -> FileUploadResponse:
    """
    Upload and validate a GPS track file.

    Supported formats: CSV, GPX, JSON
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    with LoggingContext(logger, request_id=request_id):
        logger.info("File upload started", extra={"filename": file.filename})

        # Validate file extension
        if file.filename is None:
            raise FileFormatError("No filename provided")

        file_ext = Path(file.filename).suffix.lower()
        try:
            format_type = GPSFileFormat(file_ext.lstrip("."))
        except ValueError:
            raise FileFormatError(f"Unsupported file format: {file_ext}. Supported: .csv, .gpx, .json")

        # Check file size
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise FileSizeError(settings.max_upload_size_mb)

        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # Parse and validate
            gps_points, detected_format = parse_gps_file(tmp_path)
            bounds = compute_bounds(gps_points)

            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info("File upload successful", extra={
                "request_id": request_id,
                "points": len(gps_points),
                "format": detected_format.value,
            })

            return FileUploadResponse(
                success=True,
                filename=file.filename,
                format=detected_format,
                point_count=len(gps_points),
                bounds=bounds,
                duration_ms=duration_ms,
                message=f"Successfully parsed {len(gps_points)} GPS points",
            )

        except PathSnapError:
            raise
        except Exception as e:
            logger.error("File parsing failed", extra={"error": str(e)})
            raise FileFormatError(f"Failed to parse file: {e}")
        finally:
            # Cleanup temp file
            try:
                tmp_path.unlink()
            except Exception:
                pass


# --- Route Reconstruction ---
@router.post(
    "/reconstruct",
    response_model=ReconstructionResponse,
    summary="Reconstruct road route from GPS observations",
    description="Process GPS points through the full PATHSNAP pipeline: preprocessing → candidates → ML scoring → HMM/Viterbi decoding",
    status_code=status.HTTP_200_OK,
)
async def reconstruct_route(
    request: ReconstructionRequest,
    pipeline: ReconstructionPipeline = Depends(get_reconstruction_pipeline),
) -> ReconstructionResponse:
    """
    Reconstruct a road route from GPS observations.

    This endpoint runs the complete PATHSNAP pipeline:
    1. Preprocessing (cleaning, projection, feature extraction)
    2. Candidate road segment generation
    3. ML model scoring of candidates
    4. HMM/Viterbi sequence decoding
    5. Route geometry reconstruction
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    with LoggingContext(logger, request_id=request_id):
        logger.info("Reconstruction request received", extra={
            "request_id": request_id,
            "points": len(request.gps_points),
        })

        try:
            # Run pipeline
            route, diagnostics = pipeline.reconstruct(
                gps_points=request.gps_points,
                options=request.options,
                request_id=request_id,
            )

            total_duration = (time.perf_counter() - start_time) * 1000

            response = ReconstructionResponse(
                success=True,
                status=ReconstructionStatus.COMPLETED,
                request_id=request_id,
                route=route,
                diagnostics=diagnostics,
            )

            logger.info("Reconstruction successful", extra={
                "request_id": request_id,
                "total_duration_ms": total_duration,
                "arc_ids": len(route.arc_ids),
            })

            return response

        except Exception as e:
            logger.exception("Unexpected error")
            raise


# --- Alternative: JSON Input (for manual entry) ---
@router.post(
    "/reconstruct/json",
    response_model=ReconstructionResponse,
    summary="Reconstruct route from JSON input",
    description="Alternative endpoint accepting GPS points as JSON array",
)
async def reconstruct_route_json(
    input_data: GPSPointInput,
    pipeline: ReconstructionPipeline = Depends(get_reconstruction_pipeline),
) -> ReconstructionResponse:
    """Reconstruct route from JSON GPS points."""
    request = ReconstructionRequest(gps_points=input_data.points)
    return await reconstruct_route(request, pipeline)


# --- Evaluation Endpoints ---
@router.get(
    "/evaluation/trips",
    response_model=EvaluationListResponse,
    summary="List all evaluation trips",
    description="Get the list of 10 reserved evaluation trips with summaries",
)
async def list_evaluation_trips() -> EvaluationListResponse:
    """List all available evaluation trips."""
    summaries = get_evaluation_summaries()
    return EvaluationListResponse(trips=summaries, total=len(summaries))


@router.get(
    "/evaluation/trips/{trip_id}",
    response_model=EvaluationTripDetail,
    summary="Get evaluation trip details",
    description="Get full GPS points and reference route for a specific evaluation trip",
)
async def get_evaluation_trip_detail(trip_id: str) -> EvaluationTripDetail:
    """Get detailed evaluation trip data including GPS points and reference route."""
    trip = get_evaluation_trip(trip_id)
    if trip is None:
        raise ResourceNotFoundError(f"Evaluation trip not found: {trip_id}")
    return trip


@router.post(
    "/evaluation/trips/{trip_id}/reconstruct",
    response_model=EvaluationResponse,
    summary="Reconstruct and evaluate a trip",
    description="Run the full PATHSNAP pipeline on an evaluation trip and calculate accuracy metrics",
)
async def reconstruct_evaluation_trip(
    trip_id: str,
    request: EvaluationRequest,
    pipeline: ReconstructionPipeline = Depends(get_reconstruction_pipeline),
) -> EvaluationResponse:
    """
    Reconstruct an evaluation trip and calculate accuracy metrics.

    This endpoint:
    1. Loads the evaluation trip's GPS points
    2. Runs the complete PATHSNAP pipeline
    3. Compares predicted route with reference route
    4. Returns accuracy metrics
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    # Load evaluation trip
    trip = get_evaluation_trip(trip_id)
    if trip is None:
        raise ResourceNotFoundError(f"Evaluation trip not found: {trip_id}")

    with LoggingContext(logger, request_id=request_id):
        logger.info("Evaluation reconstruction started", extra={
            "request_id": request_id,
            "trip_id": trip_id,
            "points": len(trip.gps_points),
        })

        try:
            # Run pipeline on trip's GPS points
            route, diagnostics = pipeline.reconstruct(
                gps_points=trip.gps_points,
                options=request.options,
                request_id=request_id,
            )

            total_duration = (time.perf_counter() - start_time) * 1000

            # Calculate accuracy metrics
            accuracy_metrics = calculate_accuracy_metrics(route, trip.reference_route)

            logger.info("Evaluation reconstruction successful", extra={
                "trip_id": trip_id,
                "total_duration_ms": total_duration,
                "segment_accuracy": accuracy_metrics["segment_accuracy"],
                "overall_score": accuracy_metrics["overall_score"],
            })

            return EvaluationResponse(
                success=True,
                trip_id=trip_id,
                request_id=request_id,
                predicted_route=route,
                reference_route=trip.reference_route,
                accuracy_metrics=accuracy_metrics,
                diagnostics=diagnostics,
            )

        except Exception as e:
            logger.exception("Evaluation reconstruction failed", extra={
                "trip_id": trip_id,
            })
            raise


# Export router
__all__ = ["router"]