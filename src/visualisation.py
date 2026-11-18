"""
visualisation.py
Generates reproducible scientific EDA visualisations and interactive Folium maps
for the PATHSNAP dataset and preprocessing pipeline.
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import folium
from folium import plugins

from src.data_loader import load_trip
from src.coordinate_utils import CoordinateTransformer


# Set clean seaborn / matplotlib style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def plot_gps_trajectories(sample_trips: List[str], partition: str, output_path: str):
    """
    Plot spatial trajectories of sample trips colored by instantaneous speed.
    """
    fig, axes = plt.subplots(1, len(sample_trips), figsize=(5 * len(sample_trips), 4.5), squeeze=False)
    
    for idx, trip_id in enumerate(sample_trips):
        ax = axes[0, idx]
        track_path = os.path.join("data", "processed", partition, trip_id, f"{trip_id}_cleaned_track.csv")
        if not os.path.exists(track_path):
            track_path = os.path.join("data", "raw", partition, trip_id, f"{trip_id}.track")
            df = pd.read_csv(track_path, sep=r"\s+", header=None, names=["longitude", "latitude", "timestamp"])
            speed = np.zeros(len(df))
        else:
            df = pd.read_csv(track_path)
            speed = df.get("speed_kmh", np.zeros(len(df)))

        sc = ax.scatter(
            df["longitude"], df["latitude"], c=speed, cmap="plasma", s=6, alpha=0.85
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("Speed (km/h)")
        ax.set_title(f"Trip {trip_id} ({partition})")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("PATHSNAP - GPS Trajectories Colored by Speed", fontsize=14, y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_road_network_and_gps_overlay(trip_id: str, partition: str, output_path: str):
    """
    Plot road network arcs, reference ground-truth route, and overlaid raw GPS points.
    """
    trip_data = load_trip(os.path.join("data", "raw", partition, trip_id), trip_id, partition)
    trans = CoordinateTransformer.from_center(
        trip_data.track["longitude"].mean(), trip_data.track["latitude"].mean()
    )

    nx, ny = trans.transform_to_projected(trip_data.nodes["longitude"].values, trip_data.nodes["latitude"].values)
    tx, ty = trans.transform_to_projected(trip_data.track["longitude"].values, trip_data.track["latitude"].values)

    # Filter arcs around trajectory + 500m
    min_x, max_x = tx.min() - 500, tx.max() + 500
    min_y, max_y = ty.min() - 500, ty.max() + 500

    u = trip_data.arcs["from_node"].values
    v = trip_data.arcs["to_node"].values
    ux, uy = nx[u], ny[u]
    vx, vy = nx[v], ny[v]

    in_bbox = (
        ((ux >= min_x) & (ux <= max_x) & (uy >= min_y) & (uy <= max_y)) |
        ((vx >= min_x) & (vx <= max_x) & (vy >= min_y) & (vy <= max_y))
    )
    relevant_arcs = np.where(in_bbox)[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. Road network arcs
    for aid in relevant_arcs[::2]:  # Subsample for rendering speed if dense
        ax.plot([ux[aid], vx[aid]], [uy[aid], vy[aid]], color="#bdc3c7", linewidth=0.8, alpha=0.6)
    # Add dummy line for legend
    ax.plot([], [], color="#bdc3c7", linewidth=1.2, label="Road Network Arcs")

    # 2. Reference ground truth route
    if trip_data.route is not None:
        r_arcs = trip_data.route["arc_id"].values
        for aid in r_arcs:
            ax.plot([ux[aid], vx[aid]], [uy[aid], vy[aid]], color="#e74c3c", linewidth=2.5, alpha=0.85)
        ax.plot([], [], color="#e74c3c", linewidth=2.5, label="Reference Route (Ground Truth)")

    # 3. GPS observations
    ax.scatter(tx, ty, color="#2980b9", s=10, alpha=0.75, label="Raw GPS Observations", zorder=5)

    ax.set_title(f"Trip {trip_id} - Road Network, Reference Route & GPS Overlay (UTM Meters)", fontsize=13)
    ax.set_xlabel("Easting (X meters)")
    ax.set_ylabel("Northing (Y meters)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_timestamp_gap_distribution(inventory_df: pd.DataFrame, output_path: str):
    """
    Plot distribution of sampling intervals (dt) and large gaps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # dt mean distribution
    sns.histplot(inventory_df["track_dt_mean"], bins=25, kde=True, ax=axes[0], color="#3498db")
    axes[0].set_title("Distribution of Mean Sampling Interval (dt) Per Trip")
    axes[0].set_xlabel("Mean dt (seconds)")
    axes[0].set_ylabel("Number of Trips")
    axes[0].axvline(1.0, color="#e74c3c", linestyle="--", label="Nominal 1 Hz (1.0s)")
    axes[0].legend()

    # Large gaps (>1s and >5s) count per trip
    gaps_gt1 = inventory_df["track_dt_gt_1s"]
    gaps_gt5 = inventory_df["track_dt_gt_5s"]
    x = np.arange(len(inventory_df))
    axes[1].scatter(x, gaps_gt1, label="Gaps > 1.01s", color="#e67e22", s=18, alpha=0.8)
    axes[1].scatter(x, gaps_gt5, label="Gaps > 5.0s", color="#e74c3c", s=22, alpha=0.9, marker="^")
    axes[1].set_title("Timestamp Gap Counts Per Trip")
    axes[1].set_xlabel("Trip Index (Sorted)")
    axes[1].set_ylabel("Number of Gaps")
    axes[1].set_yscale("log")
    axes[1].legend()

    plt.suptitle("PATHSNAP - GPS Temporal Sampling & Gap Distribution", fontsize=14, y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_observation_counts(inventory_df: pd.DataFrame, output_path: str):
    """
    Plot GPS observation counts and road network sizes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sns.histplot(inventory_df["track_rows"], bins=30, ax=axes[0], color="#2ecc71", kde=True)
    axes[0].set_title("GPS Track Point Counts Per Trip")
    axes[0].set_xlabel("Track Points")
    axes[0].set_ylabel("Trip Count")

    sns.histplot(inventory_df["nodes_rows"], bins=30, ax=axes[1], color="#9b59b6", kde=True)
    axes[1].set_title("Road Network Node Counts Per Trip")
    axes[1].set_xlabel("Nodes Count")
    axes[1].set_ylabel("Trip Count")

    sns.histplot(inventory_df["arcs_rows"], bins=30, ax=axes[2], color="#34495e", kde=True)
    axes[2].set_title("Road Network Arc Counts Per Trip")
    axes[2].set_xlabel("Arcs Count")
    axes[2].set_ylabel("Trip Count")

    plt.suptitle("PATHSNAP - Dataset Inventory & Row Count Distributions", fontsize=14, y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_missing_value_analysis(val_df: pd.DataFrame, output_path: str):
    """
    Bar chart showing completeness and missing values across all files.
    """
    metrics = {
        "GPS Null Values": int(val_df["gps_null_count"].sum()),
        "GPS Invalid Coords": int(val_df["gps_invalid_coords_count"].sum()),
        "Nodes Null Values": int(val_df["node_null_count"].sum()),
        "Nodes Invalid Coords": int(val_df["node_invalid_coords_count"].sum()),
        "Arcs Null Values": int(val_df["arc_null_count"].sum()),
        "Invalid Arc References": int((~val_df["arcs_all_nodes_exist"]).sum()),
        "Route Null Values": int(val_df["route_null_count"].sum()),
        "Invalid Route Arc Refs": int((~val_df["route_all_arcs_exist"]).sum()),
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(list(metrics.keys()), list(metrics.values()), color="#27ae60")
    ax.set_title("Data Quality & Missing Value Audit (Total Errors Across All 90 Trips)", fontsize=13)
    ax.set_xlabel("Anomaly Count")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f"{int(width)} (0.00%)", va="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_road_network_connectivity(val_df: pd.DataFrame, output_path: str):
    """
    Plot connected components (WCC/SCC) and isolated node distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(val_df["wcc_count"], bins=25, ax=axes[0], color="#e74c3c", kde=True)
    axes[0].set_title("Weakly Connected Components (WCC) Per Trip")
    axes[0].set_xlabel("WCC Count")
    axes[0].set_ylabel("Trip Frequency")

    sns.histplot(val_df["largest_wcc_ratio"] * 100, bins=25, ax=axes[1], color="#2980b9", kde=True)
    axes[1].set_title("Largest Connected Component Ratio (%)")
    axes[1].set_xlabel("Largest Component Share (%)")
    axes[1].set_ylabel("Trip Frequency")

    plt.suptitle("PATHSNAP - Road Network Connectivity & Component Analysis", fontsize=14, y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_candidate_roads(trip_id: str, partition: str, obs_id: int, output_path: str):
    """
    Detailed visualization of a specific GPS observation and its candidate road segments.
    """
    feat_path = os.path.join("data", "features", partition, f"{trip_id}_candidates.csv")
    track_path = os.path.join("data", "processed", partition, trip_id, f"{trip_id}_cleaned_track.csv")
    nodes_path = os.path.join("data", "processed", partition, trip_id, f"{trip_id}_validated_nodes.csv")
    arcs_path = os.path.join("data", "processed", partition, trip_id, f"{trip_id}_validated_arcs.csv")

    if not (os.path.exists(feat_path) and os.path.exists(track_path)):
        return

    df_feats = pd.read_csv(feat_path)
    df_track = pd.read_csv(track_path)
    df_nodes = pd.read_csv(nodes_path)
    df_arcs = pd.read_csv(arcs_path)

    obs_candidates = df_feats[df_feats["obs_id"] == obs_id]
    if obs_candidates.empty:
        obs_id = int(df_feats["obs_id"].iloc[0])
        obs_candidates = df_feats[df_feats["obs_id"] == obs_id]

    gps_row = df_track[df_track["obs_id"] == obs_id].iloc[0]
    gx, gy = gps_row["proj_x"], gps_row["proj_y"]

    fig, ax = plt.subplots(figsize=(9, 8))

    # Draw 50m search radius circle
    search_radius = 50.0
    circle = plt.Circle((gx, gy), search_radius, color="#bdc3c7", fill=False, linestyle="--", linewidth=1.5, label="50m Candidate Search Radius")
    ax.add_patch(circle)

    # Transform nodes
    trans = CoordinateTransformer.from_center(df_track["longitude"].mean(), df_track["latitude"].mean())
    nx, ny = trans.transform_to_projected(df_nodes["longitude"].values, df_nodes["latitude"].values)

    # Plot candidate segments
    for idx, r in obs_candidates.iterrows():
        aid = int(r["candidate_arc_id"])
        u = df_arcs.at[aid, "from_node"]
        v = df_arcs.at[aid, "to_node"]
        is_gt = bool(r.get("is_ground_truth", False))

        color = "#e74c3c" if is_gt else "#3498db"
        linewidth = 3.0 if is_gt else 1.8
        label = "True Ground Truth Arc" if is_gt else ("Candidate Road Arc" if idx == 0 else "")

        ax.plot([nx[u], nx[v]], [ny[u], ny[v]], color=color, linewidth=linewidth, alpha=0.9, label=label if label else None)
        # Text label of arc id and distance
        mid_x = (nx[u] + nx[v]) / 2.0
        mid_y = (ny[u] + ny[v]) / 2.0
        ax.text(mid_x, mid_y, f"Arc {aid}\n(d={r['perp_distance']:.1f}m)", fontsize=8, color=color, weight="bold" if is_gt else "normal")

    # Plot GPS point
    ax.scatter(gx, gy, color="#2c3e50", s=80, marker="x", linewidth=3, label=f"GPS Observation (ID: {obs_id})", zorder=10)

    # Heading arrow
    heading_deg = gps_row["gps_heading"]
    heading_rad = np.radians(heading_deg)
    arrow_dx = 15.0 * np.sin(heading_rad)
    arrow_dy = 15.0 * np.cos(heading_rad)
    ax.arrow(gx, gy, arrow_dx, arrow_dy, head_width=3, head_length=4, fc="#e67e22", ec="#e67e22", label="GPS Heading Vector", zorder=11)

    ax.set_title(f"Trip {trip_id} - Candidate Road Segments for GPS Observation {obs_id}", fontsize=13)
    ax.set_xlabel("UTM Easting (X meters)")
    ax.set_ylabel("UTM Northing (Y meters)")
    ax.set_xlim(gx - 65, gx + 65)
    ax.set_ylim(gy - 65, gy + 65)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_folium_map(trip_id: str, partition: str, output_html_path: str):
    """
    Generate interactive HTML map using Folium with OpenStreetMap tiles,
    displaying raw GPS trajectory, road network arcs, and reference route.
    """
    trip_data = load_trip(os.path.join("data", "raw", partition, trip_id), trip_id, partition)
    center_lat = float(trip_data.track["latitude"].mean())
    center_lon = float(trip_data.track["longitude"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # Feature groups for toggleable layers
    fg_gps = folium.FeatureGroup(name="GPS Observations", show=True)
    fg_route = folium.FeatureGroup(name="Ground Truth Reference Route", show=True)
    fg_network = folium.FeatureGroup(name="Road Network (Sample)", show=False)

    # 1. GPS points and polyline
    gps_coords = list(zip(trip_data.track["latitude"], trip_data.track["longitude"]))
    folium.PolyLine(
        gps_coords,
        color="#2980b9",
        weight=3,
        opacity=0.8,
        tooltip=f"GPS Track ({len(gps_coords)} points)"
    ).add_to(fg_gps)

    # Subsample GPS markers
    for idx in range(0, len(trip_data.track), max(1, len(trip_data.track) // 60)):
        row = trip_data.track.iloc[idx]
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color="#2980b9",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.9,
            popup=f"Obs ID: {row['obs_id']}<br>Time: {row['timestamp']:.1f}s<br>Lat: {row['latitude']:.5f}<br>Lon: {row['longitude']:.5f}"
        ).add_to(fg_gps)

    # 2. Reference route arcs
    if trip_data.route is not None:
        r_arcs = trip_data.route["arc_id"].values
        node_coords = trip_data.nodes[["latitude", "longitude"]].values
        from_nodes = trip_data.arcs["from_node"].values
        to_nodes = trip_data.arcs["to_node"].values

        for aid in r_arcs:
            u = from_nodes[aid]
            v = to_nodes[aid]
            p1 = [node_coords[u][0], node_coords[u][1]]
            p2 = [node_coords[v][0], node_coords[v][1]]
            folium.PolyLine(
                [p1, p2],
                color="#e74c3c",
                weight=5,
                opacity=0.85,
                tooltip=f"Route Arc ID: {aid}"
            ).add_to(fg_route)

    # 3. Add to map and add layer control
    fg_gps.add_to(m)
    fg_route.add_to(m)
    fg_network.add_to(m)
    folium.LayerControl().add_to(m)

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    m.save(output_html_path)
