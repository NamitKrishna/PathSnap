"""
dataset_builder.py
Orchestration pipeline for dataset exploration, validation, preprocessing,
candidate generation, and feature extraction.

Strictly enforces partition isolation to prevent data leakage:
  - 80 training trips (train/)
  - 10 student testing trips (student_test/)
  - 10 evaluator testing trips (evaluator_test/ - explicitly flagged if absent)
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import yaml
import pandas as pd
import numpy as np

# Set matplotlib cache directory to temp before importing matplotlib
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

from src.data_loader import load_trip, TripData
from src.coordinate_utils import CoordinateTransformer
from src.data_validation import validate_trip
from src.gps_preprocessing import preprocess_gps_track
from src.road_network import RoadNetwork
from src.candidate_generation import generate_candidates_for_trip
from src.feature_engineering import engineer_candidate_features


def setup_logger(log_file: str) -> logging.Logger:
    """Setup pipeline logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("PATHSNAP_Pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    return logger


def load_config(config_path: str = "config/preprocessing_config.yaml") -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DatasetBuilder:
    """
    End-to-end dataset builder and preprocessor.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("PATHSNAP_Pipeline")
        self.raw_dir = config["paths"]["raw_data_dir"]
        self.processed_dir = config["paths"]["processed_data_dir"]
        self.features_dir = config["paths"]["features_data_dir"]
        self.reports_dir = config["paths"]["reports_dir"]
        self.logs_dir = config["paths"]["logs_dir"]

    def discover_trips(self) -> Dict[str, List[str]]:
        """
        Discover available trips across partitions.
        """
        partitions = ["train", "student_test", "evaluator_test"]
        trip_map: Dict[str, List[str]] = {}

        for part in partitions:
            part_path = os.path.join(self.raw_dir, part)
            if os.path.exists(part_path):
                trips = sorted([
                    d for d in os.listdir(part_path)
                    if os.path.isdir(os.path.join(part_path, d))
                ])
                trip_map[part] = trips
            else:
                trip_map[part] = []

        return trip_map

    def process_trip(
        self, trip_id: str, partition: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        """
        Execute the complete processing pipeline on a single trip:
          1. Load raw data
          2. Run validation checks
          3. Project coordinates & preprocess GPS kinematics
          4. Construct RoadNetwork & STRtree spatial index
          5. Generate candidate road segments
          6. Engineer ML features
          7. Export processed files
        """
        trip_dir = os.path.join(self.raw_dir, partition, trip_id)
        trip_data = load_trip(trip_dir, trip_id, partition)

        # 1. Validation
        val_summary = validate_trip(trip_data, self.config)

        # 2. Coordinate Transformer (Centroid of trajectory)
        center_lon = float(trip_data.track["longitude"].mean())
        center_lat = float(trip_data.track["latitude"].mean())
        transformer = CoordinateTransformer.from_center(center_lon, center_lat)

        # 3. GPS Preprocessing
        cleaned_track_df, gps_clean_stats = preprocess_gps_track(
            trip_data.track, transformer, self.config
        )

        # 4. Road Network Construction
        road_network = RoadNetwork(trip_data.nodes, trip_data.arcs, transformer)

        # Bounding box around trajectory + margin
        margin = float(self.config["geospatial"].get("buffer_margin_meters", 500.0))
        min_x = cleaned_track_df["proj_x"].min() - margin
        max_x = cleaned_track_df["proj_x"].max() + margin
        min_y = cleaned_track_df["proj_y"].min() - margin
        max_y = cleaned_track_df["proj_y"].max() + margin
        road_network.build_spatial_index(bbox=(min_x, min_y, max_x, max_y))

        # 5. Candidate Generation
        candidates_df, cand_summary = generate_candidates_for_trip(
            trip_data, road_network, cleaned_track_df, self.config
        )

        # 6. Feature Engineering
        features_df = engineer_candidate_features(candidates_df)

        # 7. Export outputs
        # Processed data directories
        proc_trip_dir = os.path.join(self.processed_dir, partition, trip_id)
        os.makedirs(proc_trip_dir, exist_ok=True)

        cleaned_track_df.to_csv(os.path.join(proc_trip_dir, f"{trip_id}_cleaned_track.csv"), index=False)
        trip_data.nodes.to_csv(os.path.join(proc_trip_dir, f"{trip_id}_validated_nodes.csv"), index=False)
        trip_data.arcs.to_csv(os.path.join(proc_trip_dir, f"{trip_id}_validated_arcs.csv"), index=False)
        if trip_data.route is not None:
            trip_data.route.to_csv(os.path.join(proc_trip_dir, f"{trip_id}_validated_route.csv"), index=False)

        # Features directory
        feat_part_dir = os.path.join(self.features_dir, partition)
        os.makedirs(feat_part_dir, exist_ok=True)
        features_df.to_csv(os.path.join(feat_part_dir, f"{trip_id}_candidates.csv"), index=False)

        cand_summary["target_epsg"] = transformer.target_epsg
        return val_summary, cand_summary, len(features_df)

    def run_pipeline(self, max_trips_per_partition: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the end-to-end processing pipeline across all partitions.
        """
        trip_map = self.discover_trips()
        self.logger.info("Discovered Trips:")
        for part, trips in trip_map.items():
            self.logger.info(f"  Partition '{part}': {len(trips)} trips found")

        val_records = []
        cand_records = []
        successful_trips = 0
        failed_trips = []

        for part in ["train", "student_test"]:
            trips = trip_map[part]
            if max_trips_per_partition is not None:
                trips = trips[:max_trips_per_partition]

            self.logger.info(f"Processing partition '{part}' ({len(trips)} trips)...")
            for trip_id in trips:
                try:
                    val_res, cand_res, num_feats = self.process_trip(trip_id, part)
                    val_records.append(val_res)
                    cand_records.append(cand_res)
                    successful_trips += 1
                    self.logger.info(
                        f"  [OK] Trip {trip_id} ({part}): {cand_res['gps_points_count']} GPS pts, "
                        f"{num_feats} candidates, miss_rate={cand_res['candidate_miss_rate']:.3f}"
                    )
                except Exception as e:
                    self.logger.error(f"  [FAILED] Trip {trip_id} ({part}): {str(e)}", exc_info=True)
                    failed_trips.append({"trip_id": trip_id, "partition": part, "error": str(e)})

        # Export reports
        os.makedirs(self.reports_dir, exist_ok=True)
        df_val = pd.DataFrame(val_records)
        df_cand = pd.DataFrame(cand_records)

        df_val.to_csv(os.path.join(self.reports_dir, "data_quality_report.csv"), index=False)
        df_cand.to_csv(os.path.join(self.reports_dir, "candidate_generation_summary.csv"), index=False)

        summary_data = {
            "execution_timestamp": pd.Timestamp.now().isoformat(),
            "train_trips_count": len(trip_map.get("train", [])),
            "student_test_trips_count": len(trip_map.get("student_test", [])),
            "evaluator_test_trips_count": len(trip_map.get("evaluator_test", [])),
            "evaluator_test_status": "Unavailable in workspace - evaluated independently",
            "successful_trips_processed": successful_trips,
            "failed_trips_count": len(failed_trips),
            "failed_trips": failed_trips,
            "mean_candidate_hit_rate": float(df_cand["candidate_hit_rate"].dropna().mean()) if not df_cand.empty else 0.0,
            "mean_candidate_miss_rate": float(df_cand["candidate_miss_rate"].dropna().mean()) if not df_cand.empty else 0.0,
            "mean_candidates_per_point": float(df_cand["mean_candidates_per_point"].mean()) if not df_cand.empty else 0.0,
            "zero_candidate_points_total": int(df_cand["zero_candidate_points_count"].sum()) if not df_cand.empty else 0,
        }

        with open(os.path.join(self.reports_dir, "preprocessing_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        self.logger.info("Pipeline execution complete!")
        self.logger.info(f"Summary: {json.dumps(summary_data, indent=2)}")
        return summary_data


if __name__ == "__main__":
    cfg = load_config()
    logger = setup_logger("outputs/logs/preprocessing.log")
    builder = DatasetBuilder(cfg, logger)
    builder.run_pipeline()
