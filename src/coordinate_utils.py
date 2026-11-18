"""
coordinate_utils.py
Geographical coordinate transformations, projections, and geodesic calculations.

Why Projected Coordinates?
Raw WGS84 coordinates (longitude, latitude in degrees) cannot be directly used for
Euclidean geometric operations (such as point-to-line perpendicular distance or speed),
because 1 degree of longitude varies dramatically from ~111.32 km at the equator to 0 km
at the poles (scale factor cos(latitude)). 

For each trip, this module automatically determines the optimal Universal Transverse
Mercator (UTM) projection (EPSG:326XX for Northern Hemisphere, EPSG:327XX for Southern
Hemisphere). UTM projections preserve conformal geometry (angles and local shapes) and
provide coordinates in metric units (meters), enabling fast, millimeter-to-meter accurate
spatial indexing, Euclidean perpendicular distance calculations, and bearing calculations.
Original longitude and latitude are always preserved untouched.
"""

from typing import Tuple, Union
import math
import numpy as np
import pyproj

# Standard WGS84 Geoid for ellipsoidal geodesic computations
WGS84_GEOD = pyproj.Geod(ellps="WGS84")


def get_utm_epsg(lon: float, lat: float) -> int:
    """
    Determine the EPSG code for the appropriate UTM zone given a longitude and latitude.
    
    UTM zones divide the Earth into 60 longitudinal strips of 6 degrees each.
    Zone 1 starts at 180W.
    
    Parameters
    ----------
    lon : float
        Longitude in degrees (-180 to 180).
    lat : float
        Latitude in degrees (-90 to 90).
        
    Returns
    -------
    int
        EPSG code (e.g. 32632 for UTM Zone 32N, 32716 for UTM Zone 16S).
    """
    zone = int((lon + 180.0) / 6.0) + 1
    # Handle wrap-around edge cases
    if zone > 60:
        zone = 60
    elif zone < 1:
        zone = 1

    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


class CoordinateTransformer:
    """
    High-performance coordinate transformer between WGS84 (EPSG:4326) and a local
    projected metric coordinate system (UTM).
    
    Uses pyproj.Transformer with always_xy=True so that coordinates are strictly
    passed and returned as (longitude, latitude) / (easting_x, northing_y).
    """
    
    def __init__(self, target_epsg: int, source_epsg: int = 4326):
        self.source_epsg = source_epsg
        self.target_epsg = target_epsg
        self.to_projected = pyproj.Transformer.from_crs(
            f"EPSG:{source_epsg}", f"EPSG:{target_epsg}", always_xy=True
        )
        self.to_wgs84 = pyproj.Transformer.from_crs(
            f"EPSG:{target_epsg}", f"EPSG:{source_epsg}", always_xy=True
        )

    @classmethod
    def from_center(cls, center_lon: float, center_lat: float) -> "CoordinateTransformer":
        """Instantiate transformer based on the centroid of a trip's observations."""
        target_epsg = get_utm_epsg(center_lon, center_lat)
        return cls(target_epsg=target_epsg)

    def transform_to_projected(
        self, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Transform (longitude, latitude) degrees to (x, y) meters in projected CRS.
        """
        x, y = self.to_projected.transform(lon, lat)
        return x, y

    def transform_to_wgs84(
        self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Transform (x, y) meters back to (longitude, latitude) in WGS84 degrees.
        """
        lon, lat = self.to_wgs84.transform(x, y)
        return lon, lat


def calculate_geodesic_distance(
    lon1: Union[float, np.ndarray],
    lat1: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute geodesic distance (meters) along the WGS84 ellipsoid.
    """
    _, _, dist = WGS84_GEOD.inv(lon1, lat1, lon2, lat2)
    return dist


def calculate_bearing(
    x1: Union[float, np.ndarray],
    y1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    y2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute compass bearing (azimuth) in degrees [0, 360) from (x1, y1) to (x2, y2).
    In UTM/projected coordinates:
      - 0° = North (+y)
      - 90° = East (+x)
      - 180° = South (-y)
      - 270° = West (-x)
    """
    dx = x2 - x1
    dy = y2 - y1
    angles_rad = np.arctan2(dx, dy)
    angles_deg = np.degrees(angles_rad)
    return (angles_deg + 360.0) % 360.0


def calculate_heading_difference(
    heading1: Union[float, np.ndarray],
    heading2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute the smallest angular difference in degrees [0, 180] between two headings.
    Handles wraparound at 360 degrees.
    """
    diff = np.abs(heading1 - heading2) % 360.0
    return np.minimum(diff, 360.0 - diff)
