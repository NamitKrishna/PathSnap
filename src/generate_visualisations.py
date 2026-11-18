"""
generate_visualisations.py
Driver script to generate all required visualisations and Folium interactive maps.
"""

import os
import pandas as pd
import numpy as np
from src.visualisation import (
    plot_gps_trajectories,
    plot_road_network_and_gps_overlay,
    plot_timestamp_gap_distribution,
    plot_observation_counts,
    plot_missing_value_analysis,
    plot_road_network_connectivity,
    plot_candidate_roads,
    generate_folium_map,
)

def main():
    print("Generating visualisations...")
    inv_df = pd.read_csv("outputs/reports/dataset_inventory.csv")
    val_df = pd.read_csv("outputs/reports/data_quality_report.csv")
    cand_df = pd.read_csv("outputs/reports/candidate_generation_summary.csv")

    vis_dir = "outputs/visualisations"
    maps_dir = "outputs/maps"
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    # 1. GPS Trajectories
    sample_train = ["00000001", "00000003", "00000004"]
    plot_gps_trajectories(sample_train, "train", os.path.join(vis_dir, "gps_trajectories.png"))
    print("  [OK] gps_trajectories.png")

    # 2. Road Network and GPS Overlay
    plot_road_network_and_gps_overlay("00000001", "train", os.path.join(vis_dir, "road_network_gps_overlay.png"))
    print("  [OK] road_network_gps_overlay.png")

    # 3. Timestamp Gap Distribution
    plot_timestamp_gap_distribution(inv_df, os.path.join(vis_dir, "timestamp_gap_distribution.png"))
    print("  [OK] timestamp_gap_distribution.png")

    # 4. Observation Counts
    plot_observation_counts(inv_df, os.path.join(vis_dir, "observation_counts.png"))
    print("  [OK] observation_counts.png")

    # 5. Missing Value Analysis
    plot_missing_value_analysis(val_df, os.path.join(vis_dir, "missing_value_analysis.png"))
    print("  [OK] missing_value_analysis.png")

    # 6. Road Network Connectivity
    plot_road_network_connectivity(val_df, os.path.join(vis_dir, "road_network_connectivity.png"))
    print("  [OK] road_network_connectivity.png")

    # 7. Candidate Road Segments for Sample Observation
    plot_candidate_roads("00000001", "train", obs_id=10, output_path=os.path.join(vis_dir, "candidate_road_segments.png"))
    print("  [OK] candidate_road_segments.png")

    # 8. Interactive Folium Maps
    generate_folium_map("00000001", "train", os.path.join(maps_dir, "trip_00000001_map.html"))
    print("  [OK] trip_00000001_map.html")
    generate_folium_map("00000000", "student_test", os.path.join(maps_dir, "trip_00000000_student_test_map.html"))
    print("  [OK] trip_00000000_student_test_map.html")

    print("All visualisations and maps generated successfully!")

if __name__ == "__main__":
    main()
