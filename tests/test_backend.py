"""
PATHSNAP Backend Tests

Comprehensive tests for the FastAPI backend.
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from backend.main import app
from backend.api.models import GPSPoint, GPSFileFormat
from backend.data.gps_parser import parse_csv, parse_gpx, parse_json, compute_bounds, parse_gps_file
from backend.core.exceptions import ValidationError, FileFormatError
from backend.integrations.member1 import MockPreprocessor, MockCandidateGenerator
from backend.integrations.member2 import MockMLModel
from backend.integrations.member3 import MockHMVDecoder
from backend.core.pipeline import ReconstructionPipeline
from backend.data.road_graph import RoadGraph
from backend.config import settings

client = TestClient(app)


# --- Test Fixtures ---

@pytest.fixture
def sample_gps_points():
    """Sample GPS points for testing."""
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    return [
        GPSPoint(
            latitude=37.7749 + i * 0.001,
            longitude=-122.4194 + i * 0.001,
            timestamp=base_time + timedelta(seconds=i * 10),
            speed=10.0 + i * 0.5,
            heading=45.0 + i * 5,
            accuracy=5.0,
        )
        for i in range(10)
    ]


@pytest.fixture
def mock_road_graph():
    """Create a mock road graph for testing with bounds covering test GPS points."""
    # Test GPS points are around 37.7749, -122.4194
    # Create a denser graph with custom bounds
    bounds = {
        "min_lat": 37.77,
        "max_lat": 37.79,
        "min_lon": -122.43,
        "max_lon": -122.40,
    }
    return RoadGraph.create_mock(bounds=bounds)


# --- Health Check Tests ---

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data
    assert "components" in data


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "PATHSNAP"
    assert "version" in data


# --- File Upload Tests ---

def test_upload_csv_valid():
    """Test uploading a valid CSV file."""
    csv_content = """latitude,longitude,timestamp,speed,heading,accuracy
37.7749,-122.4194,2024-01-15T10:00:00,10.0,45.0,5.0
37.7750,-122.4193,2024-01-15T10:00:10,10.5,50.0,5.0
37.7751,-122.4192,2024-01-15T10:00:20,11.0,55.0,5.0
"""

    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["format"] == "csv"
    assert data["point_count"] == 3
    assert "bounds" in data


def test_upload_csv_missing_columns():
    """Test uploading CSV with missing required columns."""
    csv_content = """lat,lon,time
37.7749,-122.4194,2024-01-15T10:00:00
"""

    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "FILE_FORMAT_ERROR"


def test_upload_csv_invalid_data():
    """Test uploading CSV with invalid data."""
    csv_content = """latitude,longitude,timestamp
not_a_number,-122.4194,2024-01-15T10:00:00
"""

    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 400


def test_upload_unsupported_format():
    """Test uploading unsupported file format."""
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.txt", "some content", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "FILE_FORMAT_ERROR"


def test_upload_empty_file():
    """Test uploading empty file."""
    response = client.post(
        "/api/v1/upload",
        files={"file": ("empty.csv", "", "text/csv")},
    )
    assert response.status_code == 400


def test_upload_too_large():
    """Test file size limit."""
    # Create a file larger than max_upload_size_mb
    large_content = "latitude,longitude,timestamp\n" + "x" * (settings.max_upload_size_mb * 1024 * 1024 + 1)

    response = client.post(
        "/api/v1/upload",
        files={"file": ("large.csv", large_content, "text/csv")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "FILE_SIZE_ERROR"


# --- GPS Parser Tests ---

def test_parse_csv_valid():
    """Test parsing valid CSV."""
    import tempfile
    csv_content = """latitude,longitude,timestamp,speed,heading,accuracy
37.7749,-122.4194,2024-01-15T10:00:00,10.0,45.0,5.0
37.7750,-122.4193,2024-01-15T10:00:10,10.5,50.0,5.0
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        points, fmt = parse_gps_file(f.name)

    assert len(points) == 2
    assert fmt == GPSFileFormat.CSV
    assert points[0].latitude == 37.7749
    assert points[0].speed == 10.0


def test_parse_json_valid():
    """Test parsing valid JSON."""
    import tempfile
    json_content = {
        "points": [
            {"latitude": 37.7749, "longitude": -122.4194, "timestamp": "2024-01-15T10:00:00"},
            {"latitude": 37.7750, "longitude": -122.4193, "timestamp": "2024-01-15T10:00:10"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        import json
        json.dump(json_content, f)
        f.flush()
        points, fmt = parse_gps_file(f.name)

    assert len(points) == 2
    assert fmt == GPSFileFormat.JSON


def test_compute_bounds(sample_gps_points):
    """Test bounding box computation."""
    bounds = compute_bounds(sample_gps_points)
    assert "min_lat" in bounds
    assert "max_lat" in bounds
    assert "min_lon" in bounds
    assert "max_lon" in bounds
    assert bounds["min_lat"] < bounds["max_lat"]
    assert bounds["min_lon"] < bounds["max_lon"]


# --- Member 1 Integration Tests ---

def test_preprocessor(sample_gps_points):
    """Test mock preprocessor."""
    preprocessor = MockPreprocessor()
    processed = preprocessor.preprocess(sample_gps_points)

    assert len(processed) == len(sample_gps_points)
    for p in processed:
        assert p.projected_x != 0
        assert p.projected_y != 0
        assert "speed" in p.features
        assert "heading" in p.features


def test_candidate_generator(sample_gps_points, mock_road_graph):
    """Test mock candidate generator."""
    preprocessor = MockPreprocessor()
    processed = preprocessor.preprocess(sample_gps_points)

    generator = MockCandidateGenerator()
    candidates = generator.generate_candidates(
        processed_points=processed,
        road_graph=mock_road_graph,
        radius_m=50.0,
        max_candidates=5,
    )

    assert len(candidates) == len(sample_gps_points)
    for cs in candidates:
        assert len(cs.candidates) <= 5
        for c in cs.candidates:
            assert c.arc_id
            assert c.distance_m is not None
            assert c.distance_m >= 0


# --- Member 2 Integration Tests ---

def test_ml_model(sample_gps_points, mock_road_graph):
    """Test mock ML model."""
    preprocessor = MockPreprocessor()
    processed = preprocessor.preprocess(sample_gps_points)

    generator = MockCandidateGenerator()
    candidates = generator.generate_candidates(
        processed_points=processed,
        road_graph=mock_road_graph,
        radius_m=50.0,
        max_candidates=5,
    )

    model = MockMLModel()
    result = model.predict(candidates)

    assert len(result) == len(candidates)
    for cs in result:
        for c in cs.candidates:
            assert 0 <= c.probability <= 1

    # Check sorting (descending probability)
    for cs in result:
        probs = [c.probability for c in cs.candidates]
        assert probs == sorted(probs, reverse=True)


def test_ml_model_feature_names():
    """Test model feature names."""
    model = MockMLModel()
    names = model.get_feature_names()
    assert len(names) > 0
    assert "distance_m" in names
    assert "heading_diff_deg" in names


def test_ml_model_info():
    """Test model info."""
    model = MockMLModel()
    info = model.get_model_info()
    assert "type" in info
    assert "version" in info


# --- Member 3 Integration Tests ---

def test_hmm_decoder(sample_gps_points, mock_road_graph):
    """Test mock HMM/Viterbi decoder."""
    preprocessor = MockPreprocessor()
    processed = preprocessor.preprocess(sample_gps_points)

    generator = MockCandidateGenerator()
    candidates = generator.generate_candidates(
        processed_points=processed,
        road_graph=mock_road_graph,
        radius_m=500.0,  # Larger radius to ensure candidates found
        max_candidates=5,
    )

    model = MockMLModel()
    candidates = model.predict(candidates)

    decoder = MockHMVDecoder()
    result = decoder.decode(candidates, mock_road_graph)

    assert len(result.best_path) == len(candidates)
    assert all(arc_id in [c.arc_id for cs in candidates for c in cs.candidates] for arc_id in result.best_path)
    assert result.best_score != float("-inf")


def test_hmm_decoder_empty_candidates():
    """Test decoder with empty candidates."""
    from backend.integrations.member1 import CandidateSet, ProcessedGPSPoint

    decoder = MockHMVDecoder()

    # Create candidate sets with no candidates
    empty_sets = [
        CandidateSet(
            gps_point=ProcessedGPSPoint(
                original=GPSPoint(latitude=37.7749, longitude=-122.4194, timestamp=datetime.now()),
                projected_x=0, projected_y=0,
            ),
            candidates=[],
        )
    ]

    with pytest.raises(Exception) as exc_info:
        decoder.decode(empty_sets, mock_road_graph)
    assert "No candidates" in str(exc_info.value)


# --- Pipeline Integration Tests ---

def test_full_pipeline(sample_gps_points, mock_road_graph):
    """Test full reconstruction pipeline."""
    pipeline = ReconstructionPipeline(
        preprocessor=MockPreprocessor(),
        candidate_generator=MockCandidateGenerator(),
        ml_model=MockMLModel(),
        hmm_decoder=MockHMVDecoder(),
        road_graph=mock_road_graph,
    )

    from backend.api.models import ReconstructionOptions
    options = ReconstructionOptions(candidate_radius_m=500.0)
    route, diagnostics = pipeline.reconstruct(sample_gps_points, options=options)

    assert route is not None
    assert len(route.arc_ids) == len(sample_gps_points)
    assert route.total_distance_m >= 0
    assert route.point_count == len(sample_gps_points)
    assert route.segment_count == len(route.arc_ids)

    assert diagnostics.preprocessing_time_ms > 0
    assert diagnostics.candidate_generation_time_ms > 0
    assert diagnostics.model_prediction_time_ms > 0
    assert diagnostics.decoding_time_ms > 0
    assert diagnostics.total_time_ms > 0


def test_pipeline_validation_min_points():
    """Test pipeline validation for minimum points."""
    pipeline = ReconstructionPipeline()

    with pytest.raises(ValidationError):
        pipeline.reconstruct([
            GPSPoint(latitude=37.7749, longitude=-122.4194, timestamp=datetime.now())
        ])


def test_pipeline_validation_max_points():
    """Test pipeline validation for maximum points."""
    pipeline = ReconstructionPipeline()

    # Create too many points
    base_time = datetime.now()
    points = [
        GPSPoint(
            latitude=37.7749,
            longitude=-122.4194,
            timestamp=base_time + timedelta(seconds=i),
        )
        for i in range(settings.gps_max_points + 1)
    ]

    with pytest.raises(ValidationError):
        pipeline.reconstruct(points)


# --- Reconstruction API Tests ---

def test_reconstruct_endpoint(sample_gps_points):
    """Test /reconstruct endpoint."""
    request_data = {
        "gps_points": [
            {
                "latitude": p.latitude,
                "longitude": p.longitude,
                "timestamp": p.timestamp.isoformat(),
                "speed": p.speed,
                "heading": p.heading,
                "accuracy": p.accuracy,
            }
            for p in sample_gps_points
        ],
        "options": {
            "include_diagnostics": True,
            "include_geometry": True,
            "candidate_radius_m": 500.0,
        },
    }

    response = client.post("/api/v1/reconstruct", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "completed"
    assert "route" in data
    assert "diagnostics" in data
    assert len(data["route"]["arc_ids"]) == len(sample_gps_points)


def test_reconstruct_endpoint_invalid_input():
    """Test /reconstruct with invalid input."""
    request_data = {
        "gps_points": [
            {"latitude": 91, "longitude": -122.4194, "timestamp": "2024-01-15T10:00:00"},  # Invalid lat
        ],
    }

    response = client.post("/api/v1/reconstruct", json=request_data)
    assert response.status_code == 422  # Pydantic validation error


def test_reconstruct_endpoint_duplicate_timestamps():
    """Test /reconstruct with duplicate timestamps."""
    base_time = datetime.now().isoformat()
    request_data = {
        "gps_points": [
            {"latitude": 37.7749, "longitude": -122.4194, "timestamp": base_time},
            {"latitude": 37.7750, "longitude": -122.4193, "timestamp": base_time},  # Duplicate
        ],
    }

    response = client.post("/api/v1/reconstruct", json=request_data)
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


# --- JSON Input Endpoint Test ---

def test_reconstruct_json_endpoint(sample_gps_points):
    """Test /reconstruct/json endpoint."""
    request_data = {
        "points": [
            {
                "latitude": p.latitude,
                "longitude": p.longitude,
                "timestamp": p.timestamp.isoformat(),
                "speed": p.speed,
                "heading": p.heading,
                "accuracy": p.accuracy,
            }
            for p in sample_gps_points
        ],
    }

    response = client.post("/api/v1/reconstruct/json", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


# --- Road Graph Tests ---

def test_road_graph_mock():
    """Test mock road graph creation."""
    graph = RoadGraph.create_mock()
    assert graph.get_node_count() > 0
    assert graph.get_edge_count() > 0

    # Test geometry retrieval
    edge = list(graph.graph.edges(data=True))[0]
    arc_id = edge[2].get("arc_id")
    coords = graph.get_coordinates_by_arc_id(arc_id)
    assert coords is not None
    assert len(coords) >= 2


def test_road_graph_get_route_geometry(mock_road_graph):
    """Test route geometry retrieval."""
    # Get a few arc IDs
    arc_ids = []
    for u, v, key, data in mock_road_graph.graph.edges(keys=True, data=True):
        arc_ids.append(data["arc_id"])
        if len(arc_ids) >= 3:
            break

    coords = mock_road_graph.get_route_geometry(arc_ids)
    assert len(coords) >= 2
    for coord in coords:
        assert len(coord) == 2  # [lon, lat]


def test_road_graph_geojson_export(mock_road_graph):
    """Test GeoJSON export."""
    geojson = mock_road_graph.to_geojson()
    assert geojson["type"] == "FeatureCollection"
    assert "features" in geojson
    assert len(geojson["features"]) > 0


# --- Edge Cases ---

def test_reconstruct_with_options(sample_gps_points):
    """Test reconstruction with custom options."""
    request_data = {
        "gps_points": [
            {
                "latitude": p.latitude,
                "longitude": p.longitude,
                "timestamp": p.timestamp.isoformat(),
                "speed": p.speed,
                "heading": p.heading,
                "accuracy": p.accuracy,
            }
            for p in sample_gps_points
        ],
        "options": {
            "candidate_radius_m": 500.0,
            "max_candidates_per_point": 3,
            "transition_penalty": 2.0,
            "include_diagnostics": True,
            "include_geometry": False,
        },
    }

    response = client.post("/api/v1/reconstruct", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    # With include_geometry=False, coordinates might be empty or not
    # (implementation may still return them)


# Run with: pytest tests/test_backend.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])