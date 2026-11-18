"""
PATHSNAP Backend Configuration

Centralized settings management using Pydantic BaseSettings.
All configuration via environment variables with sensible defaults.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "PATHSNAP"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="development", description="Environment name")

    # Server
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Allowed CORS origins"
    )

    # File Upload
    max_upload_size_mb: int = Field(default=10, description="Max upload size in MB")
    allowed_extensions: list[str] = Field(
        default=[".csv", ".gpx", ".json"],
        description="Allowed file extensions"
    )
    upload_dir: Path = Field(
        default=Path("data/uploads"),
        description="Temporary upload directory"
    )

    # Model & Artifacts
    model_path: Path = Field(
        default=Path("artifacts/model.joblib"),
        description="Path to trained ML model"
    )
    road_graph_path: Path = Field(
        default=Path("data/processed/road_network.gpkg"),
        description="Path to road network graph file"
    )
    scaler_path: Optional[Path] = Field(
        default=None,
        description="Path to feature scaler (if used)"
    )

    # Member 3 (HMM/Viterbi) Parameters
    hmm_transition_penalty: float = Field(
        default=1.0,
        description="Penalty for unlikely transitions"
    )
    hmm_max_candidates_per_point: int = Field(
        default=10,
        description="Max candidates per GPS point"
    )
    viterbi_use_log_space: bool = Field(
        default=True,
        description="Use log probabilities in Viterbi"
    )

    # Processing
    candidate_search_radius_m: float = Field(
        default=50.0,
        description="Search radius for candidate generation (meters)"
    )
    gps_min_points: int = Field(
        default=2,
        description="Minimum GPS points for reconstruction"
    )
    gps_max_points: int = Field(
        default=10000,
        description="Maximum GPS points per request"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()

# Ensure upload directory exists
settings.upload_dir.mkdir(parents=True, exist_ok=True)