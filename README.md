# PATHSNAP – Distributed Map-Matching & Route Reconstruction
## Complete Data Exploration, Quality Validation, and Feature Preprocessing Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-15%2F15%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## 1. Project Overview

**PATHSNAP** is an end-to-end, modular, and leakage-free data exploration and preprocessing pipeline developed for large-scale distributed map-matching and vehicle route reconstruction. 

The pipeline ingests raw trajectories sampled at ~1 Hz, road networks of directed edges, and reference ground-truth route sequences. It enforces strict structural invariants, performs high-precision metric projections (Universal Transverse Mercator - UTM), indexes spatial road networks using Shapely `STRtree`, generates candidate road segments, and extracts kinematic and topological features for downstream machine learning models (e.g., Hidden Markov Models, Graph Neural Networks, and Gradient Boosted Ranking).

---

## 2. Dataset Inventory & Partitions

The dataset comprises trips across different metropolitan and regional environments worldwide (longitudes ranging from -100.42° to +137.24°, latitudes from -34.65° to +61.71°).

### Partition Breakdown & Leakage Prevention Rules

| Partition | Expected Trips | Actual Discovered | Available in Workspace? | Usage Policy |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 80 trips | **80 trips** | **Yes** | Model training, learned preprocessing, feature selection |
| **Student Test** | 10 trips | **10 trips** | **Yes** | Local evaluation & model validation only |
| **Evaluator Test** | 10 trips | **0 trips** | **No** (Held out by evaluators) | **Unavailable in workspace**; never fabricated |

> **Strict Data Leakage Controls**:
> - Train and student-test trips are kept strictly isolated in disjoint directories.
> - Evaluator-test trips are held-out by hackathon evaluators; no dummy records or fabricated data are generated.
> - Feature engineering computes metrics purely within each trip; zero cross-trip or cross-partition information leakage is permitted.

---

## 3. File Formats & Core Invariants

Each trip directory (`<trip_id>/`) contains four key files:

1. **`.track` – GPS Trajectory Observations**:
   - 3 whitespace-separated columns without header: `longitude`, `latitude`, `timestamp`.
   - Nominal sampling frequency: 1 Hz.
   - Original row index is preserved as `obs_id`. Observations are never reordered.
2. **`.nodes` – Road Network Nodes**:
   - 2 whitespace-separated columns without header: `longitude`, `latitude`.
   - **Crucial Invariant**: The 0-based row index strictly defines the `node_id`. Rows are never sorted, dropped, or re-indexed.
3. **`.arcs` – Directed Road Network Edges**:
   - 2 whitespace-separated columns without header: `from_node`, `to_node`.
   - **Crucial Invariant**: The 0-based row index strictly defines the `arc_id`. Referenced by ground-truth route files.
4. **`.route` – Ground-Truth Reference Route**:
   - 1 integer column without header: sequence of traversed `arc_id` values.
   - **Crucial Invariant**: Route row indices **do not** correspond 1-to-1 with GPS observation row indices. Instead, a vehicle traverses arcs over continuous time intervals.

---

## 4. Geographical Processing & Coordinate System

Raw geographic degrees ($^\circ$) cannot be used for Euclidean distance calculations because 1 degree of longitude spans ~111 km at the equator but shrinks with $\cos(\text{latitude})$ towards the poles.

- **Automated Metric Projection**: The pipeline inspects each trip's trajectory centroid and automatically selects the optimal Universal Transverse Mercator (UTM) projection:
  $$\text{Zone} = \left\lfloor \frac{\text{longitude} + 180^\circ}{6^\circ} \right\rfloor + 1$$
  - EPSG: $32600 + \text{Zone}$ (Northern Hemisphere)
  - EPSG: $32700 + \text{Zone}$ (Southern Hemisphere)
- **High-Precision Conformal Metric Space**: Projected coordinates $(x, y)$ in meters enable exact perpendicular Euclidean distance, directional bearings, and vehicle kinematics without distortion.
- **Original Geodetic Coordinates**: Original WGS84 `longitude` and `latitude` are preserved intact.

---

## 5. Candidate Generation & Feature Engineering

For each GPS point $P_t = (x_t, y_t)$ and candidate road segment $S_a = \overline{N_u N_v}$:

1. **Spatial Indexing (`STRtree`)**: Rapid spatial bounding query within configurable radius $R = 50.0$m (tested at 30m, 50m, 75m).
2. **Perpendicular Distance ($d_{\perp}$)**: Exact Euclidean distance from point to segment geometry, with clamped projection factor $\tau \in [0, 1]$.
3. **Compass Bearing ($\theta$)**:
   $$\theta = \left(\text{atan2}(\Delta x, \Delta y) \times \frac{180}{\pi} + 360\right) \pmod{360}$$
4. **Heading Difference ($\Delta \theta$)**:
   $$\Delta \theta = \min(|\theta_{\text{GPS}} - \theta_{\text{arc}}|, 360^\circ - |\theta_{\text{GPS}} - \theta_{\text{arc}}|) \in [0^\circ, 180^\circ]$$
5. **Vehicle Kinematics**:
   - Instantaneous speed $v = \frac{\Delta s}{\Delta t}$ (m/s and km/h)
   - Acceleration $a = \frac{\Delta v}{\Delta t}$ ($\text{m/s}^2$)
   - Diagnostic anomaly flags: `is_large_time_gap`, `is_unrealistic_speed`, `is_stationary`
6. **Emission & Structural ML Features**:
   - Gaussian emission probability $P(d) = \frac{1}{\sqrt{2\pi}\sigma_z} \exp\left(-\frac{d_{\perp}^2}{2\sigma_z^2}\right)$ ($\sigma_z = 4.07$m)
   - Directional likelihood $\exp\left(-\frac{\Delta \theta^2}{2\sigma_\theta^2}\right)$
   - Relative candidate rank ratio $\frac{d_{\perp}}{d_{\perp,\text{min}}}$
   - Arc topological degrees: `from_node_out_degree`, `to_node_in_degree`
7. **Candidate Miss Rate**:
   $$\text{Candidate Miss Rate} = \frac{\text{Observations where ground-truth arc } \notin \text{Candidates}}{\text{Total evaluated observations}}$$
   - **Achieved Hit Rate**: **95.98%** (Mean miss rate: **4.02%** at 50m radius).

---

## 6. Directory Structure

```text
PATHSNAP/
├── config/
│   └── preprocessing_config.yaml         # Configurable parameters (radii, thresholds, CRS)
├── data/
│   ├── raw/                              # Pristine extracted raw data
│   │   ├── train/                        # 80 training trips (00000001, ..., 00000099)
│   │   └── student_test/                 # 10 student testing trips
│   ├── processed/                        # Cleaned tracks and validated network files
│   │   ├── train/<trip_id>/
│   │   └── student_test/<trip_id>/
│   └── features/                         # Per-trip ML candidate feature tables
│       ├── train/<trip_id>_candidates.csv
│       └── student_test/<trip_id>_candidates.csv
├── notebooks/
│   ├── dataset_exploration.ipynb         # Interactive EDA and network analysis
│   └── preprocessing_analysis.ipynb      # Candidate miss-rate & feature validation
├── outputs/
│   ├── reports/
│   │   ├── dataset_inventory.csv         # Full file inventory, row counts, coordinates
│   │   ├── data_quality_report.csv       # Topological integrity & quality audit
│   │   ├── candidate_generation_summary.csv # Hit/miss rates per trip
│   │   └── preprocessing_summary.json    # Pipeline execution metadata
│   ├── visualisations/                   # High-resolution EDA figures
│   │   ├── gps_trajectories.png
│   │   ├── road_network_gps_overlay.png
│   │   ├── timestamp_gap_distribution.png
│   │   ├── observation_counts.png
│   │   ├── missing_value_analysis.png
│   │   ├── road_network_connectivity.png
│   │   └── candidate_road_segments.png
│   ├── maps/                             # Interactive Folium HTML maps
│   │   ├── trip_00000001_map.html
│   │   └── trip_00000000_student_test_map.html
│   └── logs/
│       └── preprocessing.log             # Timestamped execution log
├── src/
│   ├── __init__.py
│   ├── coordinate_utils.py               # UTM projections, bearings, geodesic tools
│   ├── data_loader.py                    # Zero-index preserving loaders
│   ├── data_validation.py                # Quality, topology, & continuity checkers
│   ├── gps_preprocessing.py              # Kinematics and anomaly flagging
│   ├── road_network.py                   # STRtree spatial index & arc geometry
│   ├── candidate_generation.py           # Spatial candidate query & route alignment
│   ├── feature_engineering.py            # Normalized ML features
│   ├── dataset_builder.py                # Pipeline orchestrator
│   └── generate_visualisations.py        # Figure generator
├── tests/
│   ├── test_coordinate_utils.py
│   ├── test_data_loader.py
│   ├── test_data_validation.py
│   └── test_candidate_generation.py
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## 7. How to Run the Pipeline

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Unit Tests
```bash
pytest tests/ -v
```

### Step 3: Run Full Preprocessing Pipeline
```bash
python -m src.dataset_builder
```

### Step 4: Generate EDA Visualisations & Interactive Maps
```bash
python -m src.generate_visualisations
```

---

## 8. Summary of Experimental Results & Key Insights

1. **Total Trips Processed**: 90 trips successfully processed (80 training, 10 student testing; 0 failed).
2. **Missing Values**: 0 null or missing values across all raw tracks, nodes, arcs, and routes.
3. **Duplicate Arcs**: 1,903 multi-edges detected in road networks (e.g. parallel lanes/carriageways). Original arc IDs were strictly preserved.
4. **Route Discontinuities**: 11 trips have minor edge discontinuities (e.g. 1-2 missing edge transitions out of hundreds of hops), handled gracefully during candidate evaluation.
5. **Candidate Performance**:
   - Average candidates per GPS point: **11.65**
   - Mean candidate hit rate: **95.98%**
   - Mean candidate miss rate: **4.02%**
