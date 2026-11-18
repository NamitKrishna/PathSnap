# PathSnap – Distributed Map-Matching & Route Reconstruction

Map-matching and route reconstruction pipeline designed to clean noisy GPS trajectory logs and align them to road networks using machine learning scoring, Hidden Markov Models (HMM), and Viterbi sequence decoding.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-15%2F15%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## Overview

GPS traces collected from mobile devices often contain noise, satellite drift, and sampling gaps, especially in dense urban environments. **PathSnap** is a Python-based pipeline created for a college machine learning hackathon to clean, evaluate, and match raw GPS points to road segments using machine learning models, spatial indexing (`STRtree`), Hidden Markov Models (HMM), and Viterbi sequence decoding.

---

## How It Works

1. **Data Preprocessing & Kinematics**: Raw GPS readings (latitude, longitude, timestamp, speed) are parsed, validated for coordinate bounds and timestamp monotonicity, and projected to local metric coordinates (Universal Transverse Mercator - UTM) for accurate distance measurements without spherical distortion.
2. **Spatial Indexing & Candidate Generation**: Spatial indexing (`Shapely STRtree`) retrieves road segments within a candidate radius (e.g., 50m) of each GPS point, calculating perpendicular distance to segment geometries.
3. **Feature Engineering & ML Scoring**: Features including perpendicular distance, heading difference, vehicle velocity, and topological node connectivity are computed. Machine learning classifiers (e.g., XGBoost / Random Forest) score candidate segments.
4. **HMM & Viterbi Decoding**: The route is formulated as a Hidden Markov Model where candidate road segments represent hidden states and GPS observations represent emissions. The Viterbi algorithm selects the sequence of road segments that best balances observation likelihood with road network transition connectivity.
5. **Bayesian Optimization**: Bayesian optimization via Optuna optimizes parameters such as candidate search radius, emission variances, and transition penalties.
6. **API & Visualization**: A FastAPI interface accepts trajectory inputs, while Folium and Matplotlib generate interactive maps and publication-quality diagnostic plots.

---

## Dataset & Leakage Prevention Rules

The dataset comprises GPS tracks sampled at ~1 Hz across global metropolitan regions, along with road network nodes and arcs:

| Partition | Expected Trips | Status | Usage Policy |
| :--- | :--- | :--- | :--- |
| **Train** | 80 trips | Available in `data/raw/train` | Model training, learned preprocessing, feature selection |
| **Student Test** | 10 trips | Available in `data/raw/student_test` | Local evaluation & model validation only |
| **Evaluator Test** | 10 trips | **Held out by evaluators** | Kept separate; never fabricated |

> **Strict Data Leakage Controls**:
> - Train and student-test trips are kept strictly isolated in disjoint directories.
> - Evaluator-test trips are held-out by hackathon evaluators; no dummy records or fabricated data are generated.
> - Feature engineering computes metrics purely within each trip; zero cross-trip or cross-partition information leakage is permitted.

---

## Core File Formats & Invariants

Each trip directory (`<trip_id>/`) contains four key files:

1. **`.track` – GPS Trajectory Observations**:
   - 3 whitespace-separated columns without header: `longitude`, `latitude`, `timestamp`.
   - Nominal sampling frequency: 1 Hz. Original row index is preserved as `obs_id`.
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

## Geographical Processing & Metric Projections

Raw geographic degrees ($^\circ$) cannot be used for Euclidean distance calculations because 1 degree of longitude spans ~111 km at the equator but shrinks with $\cos(\text{latitude})$ towards the poles.

- **Automated Metric Projection**: The pipeline inspects each trip's trajectory centroid and automatically selects the optimal Universal Transverse Mercator (UTM) projection:
  $$\text{Zone} = \left\lfloor \frac{\text{longitude} + 180^\circ}{6^\circ} \right\rfloor + 1$$
  - EPSG: $32600 + \text{Zone}$ (Northern Hemisphere)
  - EPSG: $32700 + \text{Zone}$ (Southern Hemisphere)
- **High-Precision Conformal Metric Space**: Projected coordinates $(x, y)$ in meters enable exact perpendicular Euclidean distance, directional bearings, and vehicle kinematics without distortion. Original WGS84 `longitude` and `latitude` are preserved intact.

---

## Repository Structure

```text
PathSnap/
├── README.md               # Project documentation and setup guide
├── .gitignore              # Ignored files (data, models, virtual environments, caches)
├── requirements.txt        # Python package dependencies
├── config/
│   └── preprocessing_config.yaml # Preprocessing & candidate generation configuration
├── data/
│   ├── raw/                # Unprocessed GPS logs and geographic data (untracked)
│   ├── processed/          # Cleaned and normalized trajectory datasets (untracked)
│   └── features/           # Extracted candidate feature tables (untracked)
├── notebooks/              # Jupyter notebooks for data analysis and experiments
│   ├── dataset_exploration.ipynb
│   └── preprocessing_analysis.ipynb
├── src/                    # Source code modules
│   ├── coordinate_utils.py # UTM projections, bearings, geodesic calculations
│   ├── data_loader.py      # Zero-index preserving loaders
│   ├── data_validation.py  # Quality, topology, and continuity checkers
│   ├── gps_preprocessing.py # Kinematics and anomaly flagging
│   ├── road_network.py     # STRtree spatial index & arc geometry
│   ├── candidate_generation.py # Spatial candidate query & route alignment
│   ├── feature_engineering.py  # Normalized ML features
│   ├── dataset_builder.py  # Pipeline orchestrator
│   ├── audit_pipeline.py   # Comprehensive hackathon requirements audit
│   ├── generate_visualisations.py # Figure generator
│   ├── preprocessing/      # Trajectory filtering, smoothing, and projection
│   ├── candidate_generation/ # Spatial query and nearest road candidate search
│   ├── candidates/         # Helper functions for candidate segments
│   ├── features/           # Spatial and kinematic feature engineering
│   ├── models/             # Machine learning scoring models
│   ├── optimization/       # Hyperparameter tuning with Optuna
│   ├── decoder/            # HMM graph construction and Viterbi decoding
│   ├── evaluation/         # Performance metrics and routing accuracy checks
│   └── api/                # FastAPI application routes
├── backend/                # Backend API service entry points
├── frontend/               # UI components and static assets
├── tests/                  # Automated unit, preprocessing, and smoke tests
│   ├── test_coordinate_utils.py
│   ├── test_data_loader.py
│   ├── test_data_validation.py
│   ├── test_candidate_generation.py
│   └── test_smoke.py
├── results/                # Output metrics, benchmark logs, and evaluation reports
├── outputs/
│   ├── reports/            # Data inventory, audit reports, and quality summaries
│   ├── visualisations/     # Diagnostic charts and trajectory overlays
│   ├── maps/               # Interactive Folium HTML maps
│   └── logs/               # Pipeline execution logs
├── visualizations/         # Folium HTML map visualizations
├── screenshots/            # Visual previews and architecture diagrams
├── report/                 # Hackathon documentation and project presentation
└── artifacts/              # Exported models and processing artifacts
```

---

## Setup Instructions

### Linux Mint / Ubuntu

#### 1. System Requirements
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

#### 2. Virtual Environment Setup
```bash
cd PathSnap
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Windows 11

#### 1. Prerequisites
- **Python 3.10+**: Install from [python.org](https://www.python.org/downloads/) *(check "Add python.exe to PATH")*.
- **Git for Windows**: Install from [git-scm.com](https://git-scm.com/).

#### 2. Virtual Environment Setup
```powershell
cd PathSnap
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Project

### 1. Running Tests
Execute unit and preprocessing test suites:
```bash
pytest tests/ -v
```

### 2. Running Preprocessing Pipeline
Execute end-to-end data validation, metric projection, spatial indexing, and candidate generation:
```bash
python -m src.dataset_builder
```

### 3. Generating Visualizations & Interactive Maps
```bash
python -m src.generate_visualisations
```

### 4. Running the API Server
Launch the FastAPI backend service with Uvicorn:
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive API docs can be viewed in your browser at `http://127.0.0.1:8000/docs`.

### 5. Prototyping with Notebooks
```bash
jupyter notebook
```

---

## Experimental Results & Preprocessing Insights

1. **Total Trips Processed**: 90 trips successfully processed (80 training, 10 student testing; 0 failed).
2. **Missing Values**: 0 null or missing values across all raw tracks, nodes, arcs, and routes.
3. **Duplicate Arcs**: 1,903 multi-edges detected in road networks (e.g. parallel lanes/carriageways). Original arc IDs were strictly preserved.
4. **Route Discontinuities**: 11 trips have minor edge discontinuities (e.g. 1-2 missing edge transitions out of hundreds of hops), handled gracefully during candidate evaluation.
5. **Candidate Performance**:
   - Average candidates per GPS point: **11.65**
   - Mean candidate hit rate: **95.98%**
   - Mean candidate miss rate: **4.02%**

---

## Team

- **Namit Krishna Renjith** ([@NamitKrishna](https://github.com/NamitKrishna))
- **Prathyush P Rao** ([@Prathyush172](https://github.com/Prathyush172))
- **Prasanna** ([@prasanna19205](https://github.com/prasanna19205))
- **Nethra Manoj** ([@nethramanoj](https://github.com/nethramanoj))
