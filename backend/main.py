"""
PATHSNAP FastAPI Application

Main entry point for the backend API server.
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.api.routes import router as api_router
from backend.core.logging import setup_logging, get_logger
from backend.core.exceptions import PathSnapError
from backend.core.pipeline import get_pipeline

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting PATHSNAP API server", extra={"version": settings.app_version})

    # Initialize pipeline (loads models, graph, etc.)
    try:
        pipeline = get_pipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize pipeline", extra={"error": str(e)})
        # Don't fail startup - let health check report degraded status

    yield

    # Shutdown
    logger.info("Shutting down PATHSNAP API server")


# Create FastAPI app
app = FastAPI(
    title="PATHSNAP API",
    description="AI-powered GPS trajectory reconstruction and map matching",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header to responses."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# Global exception handler
@app.exception_handler(PathSnapError)
async def pathsnap_exception_handler(request: Request, exc: PathSnapError):
    """Handle PATHSNAP custom exceptions."""
    # Client errors (4xx) vs server errors (5xx)
    client_error_codes = {
        "VALIDATION_ERROR",
        "FILE_FORMAT_ERROR",
        "FILE_SIZE_ERROR",
        "RESOURCE_NOT_FOUND",  # 404 might be better but using 400 for consistency
    }
    status_code = 400 if exc.code in client_error_codes else 500
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": exc.user_message,
            "error_code": exc.code,
            "request_id": getattr(request.state, "request_id", "unknown"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception("Unhandled exception", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "Internal server error",
            "user_message": "An unexpected error occurred. Please try again.",
            "details": {},
        },
    )


# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with basic info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI-powered GPS trajectory reconstruction and map matching",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# Run with: uvicorn backend.main:app --reload
if __name__ == "__main__":
    import uvicorn

    setup_logging()
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
    )