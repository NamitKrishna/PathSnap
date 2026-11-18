"""
PATHSNAP Evaluation Data

Contains the 10 reserved evaluation trips with GPS observations and reference routes.
"""
from datetime import datetime, timedelta
from typing import Any
from backend.api.models import GPSPoint, ReconstructedRoute, EvaluationTripSummary, EvaluationTripDetail


def create_mock_evaluation_trips() -> list[EvaluationTripDetail]:
    """Create 10 mock evaluation trips for testing/demonstration."""
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    trips = []

    # Trip 1: San Francisco Downtown Loop
    trips.append(EvaluationTripDetail(
        trip_id="eval_001",
        name="SF Downtown Loop",
        description="Loop around San Francisco financial district",
        gps_points=[
            GPSPoint(latitude=37.7946, longitude=-122.3999, timestamp=(base_time + timedelta(seconds=i*10)).isoformat(), speed=8.0, heading=45.0, accuracy=5.0)
            for i in range(15)
        ],
        reference_route=ReconstructedRoute(
            arc_ids=[f"arc_{i}" for i in range(15)],
            coordinates=[[-122.3999 + i*0.001, 37.7946 + i*0.001] for i in range(15)],
            total_distance_m=2500.0,
            point_count=15,
            segment_count=15,
        ),
        reference_geometry=[[-122.3999 + i*0.001, 37.7946 + i*0.001] for i in range(15)],
    ))

    # Trip 2: Golden Gate Park Crossing
    trips.append(EvaluationTripDetail(
        trip_id="eval_002",
        name="Golden Gate Park Crossing",
        description="East-West crossing through Golden Gate Park",
        gps_points=[
            GPSPoint(latitude=37.7694, longitude=-122.4862 + i*0.001, timestamp=(base_time + timedelta(seconds=i*12)).isoformat(), speed=12.0, heading=90.0, accuracy=4.0)
            for i in range(20)
        ],
        reference_route=ReconstructedRoute(
            arc_ids=[f"arc_{i}" for i in range(20)],
            coordinates=[[-122.4862 + i*0.001, 37.7694] for i in range(20)],
            total_distance_m=3200.0,
            point_count=20,
            segment_count=20,
        ),
        reference_geometry=[[-122.4862 + i*0.001, 37.7694] for i in range(20)],
    ))

    # Trip 3: Mission District Streets
    trips.append(EvaluationTripDetail(
        trip_id="eval_003",
        name="Mission District Streets",
        description="Navigate through Mission District neighborhood",
        gps_points=[
            GPSPoint(latitude=37.7599 + i*0.0005, longitude=-122.4148 + i*0.0008, timestamp=(base_time + timedelta(seconds=i*8)).isoformat(), speed=6.0, heading=30.0, accuracy=6.0)
            for i in range(18)
        ],
        reference_route=ReconstructedRoute(
            arc_ids=[f"arc_{i}" for i in range(18)],
            coordinates=[[-122.4148 + i*0.0008, 37.7599 + i*0.0005] for i in range(18)],
            total_distance_m=1800.0,
            point_count=18,
            segment_count=18,
        ),
        reference_geometry=[[-122.4148 + i*0.0008, 37.7599 + i*0.0005] for i in range(18)],
    ))

    # Trip 4: Embarcadero Waterfront
    trips.append(EvaluationTripDetail(
        trip_id="eval_004",
        name="Embarcadero Waterfront",
        description="North-South along the Embarcadero",
        gps_points=[
            GPSPoint(latitude=37.7955 - i*0.001, longitude=-122.3937, timestamp=(base_time + timedelta(seconds=i*15)).isoformat(), speed=10.0, heading=180.0, accuracy=3.0)
            for i in range(12)
        ],
        reference_route=ReconstructedRoute(
            arc_ids=[f"arc_{i}" for i in range(12)],
            coordinates=[[-122.3937, 37.7955 - i*0.001] for i in range(12)],
            total_distance_m=2100.0,
            point_count=12,
            segment_count=12,
        ),
        reference_geometry=[[-122.3937, 37.7955 - i*0.001] for i in range(12)],
    ))

    # Trip 5: Lombard Street Zigzag
    trips.append(EvaluationTripDetail(
        trip_id="eval_005",
        name="Lombard Street Zigzag",
        description="The famous crooked street section",
        gps_points=[
            GPSPoint(latitude=37.8021, longitude=-122.4187 + i*0.0005, timestamp=(base_time + timedelta(seconds=i*5)).isoformat(), speed=3.0, heading=90.0 if i % 2 == 0 else 270.0, accuracy=8.0)
            for i in range(10)
        ],
        reference_route=ReconstructedRoute(
            arc_ids=[f"arc_{i}" for i in range(10)],
            coordinates=[[-122.4187 + i*0.0005, 37.8021] for i in range(10)],
            total_distance_m=800.0,
            point_count=10,
            segment_count=10,
        ),
        reference_geometry=[[-122.4187 + i*0.0005, 37.8021] for i in range(10)],
    ))

    # Trip 6: Twin Peaks Summit
    trips.append(EvaluationTripDetail(
        trip_id="eval_006",
        name="Twin Peaks Summit",
        description="Winding road to Twin Peaks viewpoint",
        gps_points=[
            GPSPoint(latitude=37.7544 + i*0.0008, longitude=-122.4477 - i*0.0006, timestamp=(base_time + timedelta(seconds=i*20)).isoformat(), speed=5.0, heading=315.0, accuracy=7.0)
            for i in range(14)
        ],
        reference_route=ReconstructedRoute(
            arc_ids=[f"arc_{i}" for i in range(14)],
            coordinates=[[-122.4477 - i*0.0006, 37.7544 + i*0.0008] for i in range(14)],
            total_distance_m=2800.0,
            point_count=14,
            segment_count=14,
        ),
        reference_geometry=[[-122.4477 - i*0.0006, 37.7544 + i*0.0008] for i in range(14)],
    ))

    # Trip 7: Presidio to Marina
    trips.append(EvaluationTripDetail(
        trip_id="eval_007",
        name="Presidio to Marina",
        description="Through Presidio park to Marina district",
        gps_points=[
            GPSPoint(latitude=37.7997 - i*0.0007, longitude=-122.4617 + i*0.0009, timestamp=(base_time + timedelta(seconds=i*18)).isoformat(), speed=11.0, heading=60.0, accuracy=4.0)
            for i in range(16)
        ],
        reference_route=ReconstructedRoute(
            arc_ids=[f"arc_{i}" for i in range(16)],
            coordinates=[[-122.4617 + i*0.0009, 37.7997 - i*0.0007] for i in range(16)],
            total_distance_m=3500.0,
            point_count=16,
            segment_count=16,
        ),
        reference_geometry=[[-122.4617 + i*0.0009, 37.7997 - i*0.0007] for i in range(16)],
    ))

    # Trip 8: SoMa Arterial
    trips.append(EvaluationTripDetail(
        trip_id="eval_008",
        name="SoMa Arterial",
        description="Major arterial through South of Market",
        gps_points=[
            GPSPoint(latitude=37.7749 + i*0.0006, longitude=-122.4094, timestamp=(base_time + timedelta(seconds=i*7)).isoformat(), speed=15.0, heading=0.0, accuracy=3.0)
            for i in range(22)
        ],
        reference_route=ReconstructedRoute(
            arc_ids=[f"arc_{i}" for i in range(22)],
            coordinates=[[-122.4094, 37.7749 + i*0.0006] for i in range(22)],
            total_distance_m=4200.0,
            point_count=22,
            segment_count=22,
        ),
        reference_geometry=[[-122.4094, 37.7749 + i*0.0006] for i in range(22)],
    ))

    # Trip 9: Castro Neighborhood
    trips.append(EvaluationTripDetail(
        trip_id="eval_009",
        name="Castro Neighborhood",
        description="Winding streets of Castro district",
        gps_points=[
            GPSPoint(latitude=37.7609 + i*0.0004, longitude=-122.4350 + i*0.0005, timestamp=(base_time + timedelta(seconds=i*11)).isoformat(), speed=7.0, heading=45.0, accuracy=5.0)
            for i in range(17)
        ],
        reference_route=ReconstructedRoute(
            arc_ids=[f"arc_{i}" for i in range(17)],
            coordinates=[[-122.4350 + i*0.0005, 37.7609 + i*0.0004] for i in range(17)],
            total_distance_m=2200.0,
            point_count=17,
            segment_count=17,
        ),
        reference_geometry=[[-122.4350 + i*0.0005, 37.7609 + i*0.0004] for i in range(17)],
    ))

    # Trip 10: Ocean Beach Coastal
    trips.append(EvaluationTripDetail(
        trip_id="eval_010",
        name="Ocean Beach Coastal",
        description="Coastal route along Great Highway",
        gps_points=[
            GPSPoint(latitude=37.7517 - i*0.0009, longitude=-122.5110, timestamp=(base_time + timedelta(seconds=i*13)).isoformat(), speed=13.0, heading=180.0, accuracy=4.0)
            for i in range(19)
        ],
        reference_route=ReconstructedRoute(
            arc_ids=[f"arc_{i}" for i in range(19)],
            coordinates=[[-122.5110, 37.7517 - i*0.0009] for i in range(19)],
            total_distance_m=3800.0,
            point_count=19,
            segment_count=19,
        ),
        reference_geometry=[[-122.5110, 37.7517 - i*0.0009] for i in range(19)],
    ))

    return trips


# Global evaluation trips cache
_evaluation_trips: list[EvaluationTripDetail] | None = None


def get_evaluation_trips() -> list[EvaluationTripDetail]:
    """Get all evaluation trips."""
    global _evaluation_trips
    if _evaluation_trips is None:
        _evaluation_trips = create_mock_evaluation_trips()
    return _evaluation_trips


def get_evaluation_trip(trip_id: str) -> EvaluationTripDetail | None:
    """Get a specific evaluation trip by ID."""
    trips = get_evaluation_trips()
    for trip in trips:
        if trip.trip_id == trip_id:
            return trip
    return None


def get_evaluation_summaries() -> list[EvaluationTripSummary]:
    """Get summaries of all evaluation trips."""
    trips = get_evaluation_trips()
    return [
        EvaluationTripSummary(
            trip_id=trip.trip_id,
            name=trip.name,
            description=trip.description,
            point_count=trip.gps_points.__len__(),
            bounds=compute_bounds(trip.gps_points),
            reference_distance_m=trip.reference_route.total_distance_m,
            reference_segment_count=trip.reference_route.segment_count,
        )
        for trip in trips
    ]


def compute_bounds(points: list[GPSPoint]) -> dict[str, float]:
    """Compute bounding box of GPS points."""
    lats = [p.latitude for p in points]
    lons = [p.longitude for p in points]
    return {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons),
    }


def calculate_accuracy_metrics(
    predicted: ReconstructedRoute,
    reference: ReconstructedRoute,
) -> dict[str, float]:
    """
    Calculate accuracy metrics between predicted and reference routes.

    Metrics:
    - segment_accuracy: Fraction of correctly matched segments
    - distance_error_pct: Percentage error in total distance
    - geometry_similarity: Fréchet distance or similar (simplified)
    """
    # Segment-level accuracy
    pred_arcs = set(predicted.arc_ids)
    ref_arcs = set(reference.arc_ids)

    if len(ref_arcs) == 0:
        segment_accuracy = 0.0
    else:
        correct = len(pred_arcs & ref_arcs)
        segment_accuracy = correct / len(ref_arcs)

    # Distance error
    if reference.total_distance_m == 0:
        distance_error_pct = 0.0
    else:
        distance_error_pct = abs(predicted.total_distance_m - reference.total_distance_m) / reference.total_distance_m * 100

    # Point count accuracy
    point_accuracy = 1.0 - abs(predicted.point_count - reference.point_count) / max(predicted.point_count, reference.point_count)

    # Overall score (weighted)
    overall_score = (segment_accuracy * 0.5) + ((100 - distance_error_pct) / 100 * 0.3) + (point_accuracy * 0.2)
    overall_score = max(0.0, min(1.0, overall_score))

    return {
        "segment_accuracy": round(segment_accuracy, 4),
        "distance_error_pct": round(distance_error_pct, 2),
        "point_accuracy": round(point_accuracy, 4),
        "overall_score": round(overall_score, 4),
    }