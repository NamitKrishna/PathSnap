"""
PATHSNAP GPS File Parser

Supports parsing GPS tracks from CSV, GPX, and JSON formats.
"""
import csv
import gpxpy
import gpxpy.gpx
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from backend.api.models import GPSPoint, GPSFileFormat
from backend.core.exceptions import FileFormatError, ValidationError


REQUIRED_CSV_COLUMNS = {"latitude", "longitude", "timestamp"}
OPTIONAL_CSV_COLUMNS = {"speed", "heading", "accuracy"}


def detect_format(file_path: Path) -> GPSFileFormat:
    """Detect GPS file format from extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return GPSFileFormat.CSV
    elif suffix == ".gpx":
        return GPSFileFormat.GPX
    elif suffix == ".json":
        return GPSFileFormat.JSON
    else:
        raise FileFormatError(f"Unsupported file extension: {suffix}")


def parse_csv(file_path: Path) -> list[GPSPoint]:
    """Parse GPS points from CSV file."""
    points = []

    with open(file_path, "r", newline="", encoding="utf-8") as f:
        # Sniff dialect
        sample = f.read(1024)
        f.seek(0)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        f.seek(0)

        reader = csv.DictReader(f, dialect=dialect)

        # Validate columns
        if reader.fieldnames is None:
            raise FileFormatError("CSV file has no headers")

        fieldnames_set = set(reader.fieldnames)
        missing = REQUIRED_CSV_COLUMNS - fieldnames_set
        if missing:
            raise FileFormatError(f"Missing required columns: {missing}")

        for row_num, row in enumerate(reader, start=2):  # 1-indexed + header
            try:
                point = _parse_csv_row(row, row_num)
                points.append(point)
            except (ValueError, KeyError) as e:
                raise ValidationError(f"Row {row_num}: {e}", field=f"row_{row_num}")

    if not points:
        raise ValidationError("CSV file contains no valid GPS points")

    return points


def _parse_csv_row(row: dict[str, str], row_num: int) -> GPSPoint:
    """Parse a single CSV row into GPSPoint."""
    # Required fields
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    timestamp_str = row["timestamp"]

    # Parse timestamp
    try:
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except ValueError:
        raise ValidationError(f"Invalid timestamp format: {timestamp_str}")

    # Optional fields
    speed = None
    if "speed" in row and row["speed"]:
        speed = float(row["speed"])
        if speed < 0:
            raise ValidationError("Speed cannot be negative")

    heading = None
    if "heading" in row and row["heading"]:
        heading = float(row["heading"])
        if not (0 <= heading < 360):
            raise ValidationError("Heading must be in [0, 360)")

    accuracy = None
    if "accuracy" in row and row["accuracy"]:
        accuracy = float(row["accuracy"])
        if accuracy < 0:
            raise ValidationError("Accuracy cannot be negative")

    return GPSPoint(
        latitude=lat,
        longitude=lon,
        timestamp=timestamp,
        speed=speed,
        heading=heading,
        accuracy=accuracy,
    )


def parse_gpx(file_path: Path) -> list[GPSPoint]:
    """Parse GPS points from GPX file."""
    points = []

    with open(file_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.latitude is None or point.longitude is None or point.time is None:
                    continue

                gps_point = GPSPoint(
                    latitude=point.latitude,
                    longitude=point.longitude,
                    timestamp=point.time,
                    speed=point.speed,
                    heading=None,  # GPX doesn't standardly include heading
                    accuracy=None,  # GPX doesn't standardly include accuracy
                )
                points.append(gps_point)

    if not points:
        raise ValidationError("GPX file contains no valid GPS points")

    return points


def parse_json(file_path: Path) -> list[GPSPoint]:
    """Parse GPS points from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support multiple JSON structures
    points_data = None

    if isinstance(data, list):
        # Direct array of points
        points_data = data
    elif isinstance(data, dict):
        # Object with points array
        if "points" in data and isinstance(data["points"], list):
            points_data = data["points"]
        elif "gps_points" in data and isinstance(data["gps_points"], list):
            points_data = data["gps_points"]
        elif "track" in data and isinstance(data["track"], list):
            points_data = data["track"]
        else:
            raise FileFormatError("JSON must contain a 'points', 'gps_points', or 'track' array")

    if not points_data:
        raise ValidationError("JSON file contains no GPS points")

    points = []
    for i, point_data in enumerate(points_data):
        try:
            point = GPSPoint(**point_data)
            points.append(point)
        except Exception as e:
            raise ValidationError(f"Point {i}: {e}")

    return points


def parse_gps_file(file_path: Path | str) -> tuple[list[GPSPoint], GPSFileFormat]:
    """
    Parse GPS file and return points with detected format.

    Args:
        file_path: Path to GPS file (string or Path)

    Returns:
        Tuple of (GPS points, detected format)

    Raises:
        FileFormatError: If format is unsupported or parsing fails
        ValidationError: If data is invalid
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    format_type = detect_format(file_path)

    if format_type == GPSFileFormat.CSV:
        points = parse_csv(file_path)
    elif format_type == GPSFileFormat.GPX:
        points = parse_gpx(file_path)
    elif format_type == GPSFileFormat.JSON:
        points = parse_json(file_path)
    else:
        raise FileFormatError(f"Unsupported format: {format_type}")

    # Validate point count
    if len(points) < 2:
        raise ValidationError("At least 2 GPS points required for reconstruction")

    # Sort by timestamp
    points.sort(key=lambda p: p.timestamp)

    # Validate timestamps are strictly increasing
    for i in range(1, len(points)):
        if points[i].timestamp <= points[i - 1].timestamp:
            raise ValidationError(f"Timestamps must be strictly increasing (point {i})")

    return points, format_type


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