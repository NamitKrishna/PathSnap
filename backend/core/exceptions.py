"""
PATHSNAP Custom Exceptions

Hierarchical exception classes for precise error handling
and user-friendly error messages.
"""
from typing import Any, Optional


class PathSnapError(Exception):
    """Base exception for all PATHSNAP errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.user_message = user_message or message

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
        }


# --- Input Validation Errors ---
class ValidationError(PathSnapError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        code: str = "VALIDATION_ERROR",
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ):
        if details is None:
            details = {}
        if field:
            details["field"] = field
        if user_message is None:
            user_message = f"Invalid input: {message}"
        super().__init__(message, code=code, details=details, user_message=user_message)


class FileFormatError(ValidationError):
    """Raised when uploaded file format is invalid."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            code="FILE_FORMAT_ERROR",
            user_message=f"File format error: {message}",
            **kwargs,
        )


class FileSizeError(ValidationError):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, max_size_mb: int, **kwargs):
        super().__init__(
            f"File size exceeds {max_size_mb} MB limit",
            code="FILE_SIZE_ERROR",
            details={"max_size_mb": max_size_mb},
            user_message=f"File too large. Maximum size is {max_size_mb} MB.",
            **kwargs,
        )


# --- Processing Errors ---
class ProcessingError(PathSnapError):
    """Raised when a processing step fails."""

    def __init__(
        self,
        stage: str,
        message: str,
        code: str = "PROCESSING_ERROR",
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ):
        if details is None:
            details = {}
        details["stage"] = stage
        if user_message is None:
            user_message = f"Processing failed at {stage.lower()}. Please try again."
        super().__init__(message, code=code, details=details, user_message=user_message)


class PreprocessingError(ProcessingError):
    """Raised when GPS preprocessing fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__("Preprocessing", message, **kwargs)


class CandidateGenerationError(ProcessingError):
    """Raised when candidate road generation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__("Candidate Generation", message, **kwargs)


class ModelPredictionError(ProcessingError):
    """Raised when ML model prediction fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__("Model Prediction", message, **kwargs)


class DecodingError(ProcessingError):
    """Raised when HMM/Viterbi decoding fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__("Route Decoding", message, **kwargs)


# --- Resource Errors ---
class ResourceNotFoundError(PathSnapError):
    """Raised when a required resource is missing."""

    def __init__(
        self,
        resource: str,
        path: Optional[str] = None,
        code: str = "RESOURCE_NOT_FOUND",
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ):
        if details is None:
            details = {}
        details["resource"] = resource
        if path:
            details["path"] = path
        if user_message is None:
            user_message = f"Required {resource.lower()} is not available. Please contact support."
        super().__init__(f"Required resource not found: {resource}", code=code, details=details, user_message=user_message)


class ModelNotLoadedError(ResourceNotFoundError):
    """Raised when ML model fails to load."""

    def __init__(self, path: str, **kwargs):
        super().__init__("ML Model", path, **kwargs)


class RoadGraphNotLoadedError(ResourceNotFoundError):
    """Raised when road network graph fails to load."""

    def __init__(self, path: str, **kwargs):
        super().__init__("Road Network", path, **kwargs)


# --- Pipeline Errors ---
class PipelineError(PathSnapError):
    """Raised when the full reconstruction pipeline fails."""

    def __init__(
        self,
        message: str,
        failed_stage: str,
        code: str = "PIPELINE_ERROR",
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ):
        if details is None:
            details = {}
        details["failed_stage"] = failed_stage
        if user_message is None:
            user_message = "Route reconstruction failed. Please check your input and try again."
        super().__init__(message, code=code, details=details, user_message=user_message)


# --- External Service Errors ---
class ExternalServiceError(PathSnapError):
    """Raised when an external service is unavailable."""

    def __init__(
        self,
        service: str,
        message: str,
        code: str = "EXTERNAL_SERVICE_ERROR",
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ):
        if details is None:
            details = {}
        details["service"] = service
        if user_message is None:
            user_message = f"{service} is temporarily unavailable. Please try again later."
        super().__init__(f"{service}: {message}", code=code, details=details, user_message=user_message)