"""
PATHSNAP Structured Logging Configuration

JSON-formatted structured logging for production observability.
"""
import json
import logging
import sys
from datetime import datetime
from typing import Any
from backend.config import settings


class JSONFormatter(logging.Formatter):
    """JSON log formatter with consistent fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms
        if hasattr(record, "stage"):
            log_obj["stage"] = record.stage
        if hasattr(record, "extra"):
            log_obj.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.utcnow().isoformat()
        level = record.levelname
        name = record.name
        message = record.getMessage()

        # Build extra info
        extra_parts = []
        if hasattr(record, "request_id"):
            extra_parts.append(f"req_id={record.request_id}")
        if hasattr(record, "stage"):
            extra_parts.append(f"stage={record.stage}")
        if hasattr(record, "duration_ms"):
            extra_parts.append(f"duration={record.duration_ms:.1f}ms")

        extra_str = f" [{', '.join(extra_parts)}]" if extra_parts else ""

        return f"{timestamp} | {level:8} | {name}{extra_str} | {message}"


def setup_logging() -> None:
    """Configure application logging based on settings."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if settings.log_format.lower() == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(TextFormatter())

    root_logger.addHandler(console_handler)

    # Reduce noise from third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Log startup
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={"level": settings.log_level, "format": settings.log_format}
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for adding contextual fields to logs."""

    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, *args):
        logging.setLogRecordFactory(self.old_factory)