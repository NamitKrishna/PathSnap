"""
PATHSNAP API Request/Response Models

Pydantic models for type-safe API contracts.
"""
from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# --- Enums ---
class GPSFileFormat(str, Enum):
    """Supported GPS file formats."""
    CSV = "csv"
    GPX = "gpx"
    JSON = "json"


class ReconstructionStatus(str, Enum):
    """Reconstruction job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Request Models ---
class GPSPoint(BaseModel):
    """Single GPS observation."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp")
    speed: Optional[float] = Field(default=None, ge=0, description="Speed in m/s")
    heading: Optional[float] = Field(default=None, ge=0, lt=360, description="Heading in degrees")
    accuracy: Optional[float] = Field(default=None, ge=0, description="Horizontal accuracy in meters")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class GPSPointInput(BaseModel):
    """Input format for manual GPS entry."""
    points: list[GPSPoint] = Field(..., min_length=2, max_length=10000)


class ReconstructionRequest(BaseModel):
    """Request to reconstruct a route from GPS observations."""
    gps_points: list[GPSPoint] = Field(..., min_length=2, max_length=10000)
    options: Optional["ReconstructionOptions"] = Field(default=None)

    @property
    def point_count(self) -> int:
        return len(self.gps_points)


class ReconstructionOptions(BaseModel):
    """Optional reconstruction parameters."""
    candidate_radius_m: Optional[float] = Field(default=None, ge=1, le=500)
    max_candidates_per_point: Optional[int] = Field(default=None, ge=1, le=50)
    transition_penalty: Optional[float] = Field(default=None, ge=0, le=10)
    include_diagnostics: bool = Field(default=True)
    include_geometry: bool = Field(default=True)


# --- Response Models ---
class CandidateSegment(BaseModel):
    """A candidate road segment for a GPS point."""
    arc_id: str = Field(..., description="Unique road segment identifier")
    probability: float = Field(..., ge=0, le=1, description="Model prediction probability")
    distance_m: Optional[float] = Field(default=None, ge=0, description="Distance from GPS point to segment")
    heading_diff_deg: Optional[float] = Field(default=None, ge=0, le=180, description="Heading difference")
    road_class: Optional[str] = Field(default=None, description="Road classification")
    geometry: Optional[dict[str, Any]] = Field(default=None, description="GeoJSON LineString geometry")


class GPSPointWithCandidates(BaseModel):
    """GPS point with its candidate road segments."""
    gps_point: GPSPoint
    candidates: list[CandidateSegment] = Field(default_factory=list)
    selected_candidate: Optional[CandidateSegment] = Field(default=None)


class ReconstructedRoute(BaseModel):
    """Reconstructed road route."""
    arc_ids: list[str] = Field(..., description="Ordered sequence of road segment IDs")
    coordinates: list[list[float]] = Field(..., description="Route geometry as [lon, lat] pairs")
    total_distance_m: float = Field(..., ge=0, description="Total route distance in meters")
    point_count: int = Field(..., ge=0, description="Number of GPS observations")
    segment_count: int = Field(..., ge=0, description="Number of road segments")


class ReconstructionDiagnostics(BaseModel):
    """Diagnostic information from reconstruction pipeline."""
    preprocessing_time_ms: float = Field(..., ge=0)
    candidate_generation_time_ms: float = Field(..., ge=0)
    model_prediction_time_ms: float = Field(..., ge=0)
    decoding_time_ms: float = Field(..., ge=0)
    total_time_ms: float = Field(..., ge=0)
    candidate_stats: dict[str, Any] = Field(default_factory=dict)
    hmm_stats: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ReconstructionResponse(BaseModel):
    """Response from route reconstruction."""
    success: bool
    status: ReconstructionStatus
    request_id: str
    route: Optional[ReconstructedRoute] = None
    diagnostics: Optional[ReconstructionDiagnostics] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, str] = Field(default_factory=dict)
    uptime_seconds: Optional[float] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    error_code: str
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[dict[str, Any]] = None


# --- File Upload Models ---
class FileUploadResponse(BaseModel):
    """Response after file upload and validation."""
    success: bool
    filename: str
    format: GPSFileFormat
    point_count: int
    bounds: Optional[dict[str, float]] = None  # min_lat, max_lat, min_lon, max_lon
    duration_ms: float
    message: Optional[str] = None
    error: Optional[str] = None


# --- Evaluation Models ---
class EvaluationTripSummary(BaseModel):
    """Summary of an evaluation trip."""
    trip_id: str
    name: str
    description: str
    point_count: int
    bounds: dict[str, float]
    reference_distance_m: float
    reference_segment_count: int


class EvaluationTripDetail(BaseModel):
    """Full evaluation trip with GPS points and reference route."""
    trip_id: str
    name: str
    description: str
    gps_points: list[GPSPoint]
    reference_route: ReconstructedRoute
    reference_geometry: list[list[float]]


class EvaluationRequest(BaseModel):
    """Request to evaluate a trip."""
    trip_id: str
    options: Optional[ReconstructionOptions] = None


class EvaluationResponse(BaseModel):
    """Response from trip evaluation."""
    success: bool
    trip_id: str
    request_id: str
    predicted_route: Optional[ReconstructedRoute] = None
    reference_route: Optional[ReconstructedRoute] = None
    accuracy_metrics: Optional[dict[str, float]] = None
    diagnostics: Optional[ReconstructionDiagnostics] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


class EvaluationListResponse(BaseModel):
    """List of evaluation trips."""
    trips: list[EvaluationTripSummary]
    total: int


# Forward reference resolution
ReconstructionRequest.model_rebuild()
EvaluationRequest.model_rebuild()