# PathSnap – Distributed Map-Matching & Route Reconstruction

Map-matching and route reconstruction pipeline designed to clean noisy GPS trajectory logs and align them to road networks using machine learning scoring, Hidden Markov Models (HMM), and Viterbi sequence decoding.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-51%2F51%20Passing-brightgreen.svg)]()
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

## Model Training & Experimentation

### 1. Problem Formulation
Candidate-level supervised classification and ranking for map matching. Each GPS observation has multiple candidate road segments retrieved within a spatial search radius (mean 11.65 candidates/observation). 
- **Binary Target**: `is_ground_truth ∈ {0, 1}` indicating whether candidate $c$ matches the true traversed road segment.
- **Dataset Imbalance**: ~9.86% positive class across 1,840,910 training candidate rows; handled via inverse-frequency class weighting (`class_weight='balanced'` in scikit-learn, `scale_pos_weight=9.14` in XGBoost).
- **Leakage Prevention**: 5-fold `GroupKFold` cross-validation grouped strictly by `trip_id` on the 80 training trips. The 10 student-test trips are held out strictly for independent test evaluation. The 10 evaluator-test trips are fully quarantined.

### 2. Evaluated Models
1. **Logistic Regression (L2)**: Linear baseline with standard feature scaling.
2. **K-Nearest Neighbors (KNN)**: Non-parametric distance-weighted candidate ranking with indexed reference subsampling (`FastKNN`).
3. **Decision Tree**: Non-linear tree model (`max_depth=12`, `min_samples_leaf=30`, balanced class weighting).
4. **Random Forest**: Ensemble of 100 balanced decision trees with `max_samples=0.20` subsampling.
5. **XGBoost (Hist)**: Gradient boosted trees with histogram-based binning (`tree_method='hist'`, `n_estimators=250`, `learning_rate=0.08`, `scale_pos_weight=9.14`).
6. **Optimized Random Forest**: Bayesian hyperparameter tuning via Optuna (TPE sampler).

### 3. Model Benchmark & Evaluation Results

All models evaluated across 80 training trips (1,840,910 candidates, 16,096 observations) and 10 student-test trips (207,609 candidates, 3,958 observations):

| Model | 5-Fold CV F1 | Test F1 | Test Precision | Test Recall | Test ROC-AUC | Observation Top-1 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Optimized)** | **0.7292** | **0.7349** | **0.5993** | **0.9498** | **0.9835** | **79.34%** |
| **Random Forest (Baseline)** | 0.7285 ± 0.038 | 0.7340 | 0.5985 | 0.9489 | **0.9835** | 79.32% |
| **XGBoost** | 0.7253 ± 0.034 | 0.7257 | 0.5859 | **0.9531** | 0.9834 | **79.54%** |
| **Decision Tree** | 0.7122 ± 0.034 | 0.7171 | 0.5751 | 0.9522 | 0.9789 | 79.48% |
| **KNN (Subsampled)** | 0.6655 ± 0.052 | 0.6941 | 0.7605 | 0.6384 | 0.9550 | 73.81% |
| **Logistic Regression** | 0.5875 ± 0.016 | 0.5829 | 0.4247 | 0.9291 | 0.9669 | 74.92% |

### 4. Bayesian Hyperparameter Optimization (Optuna)
Optuna TPE optimization on Random Forest yielded optimal parameters:
- `n_estimators`: 80
- `max_depth`: 14
- `min_samples_leaf`: 22
- `max_features`: `sqrt`
- `max_samples`: 0.1988 (19.88% bootstrap subsampling)

### 5. Feature Importance Analysis
Top predictive features identified by tree ensembles:
1. `relative_dist_diff`: Relative distance difference to nearest candidate ($Δd / d_{min}$) – dominant signal across all models.
2. `distance_rank_ratio`: Candidate distance relative to candidate set span.
3. `perp_distance`: Orthogonal Euclidean distance from GPS point to road segment.
4. `emission_prob_gaussian`: Gaussian emission probability density.
5. `norm_proj_pos`: Normalized position along the candidate arc $[0, 1]$.
6. `heading_diff_cos` & `heading_prob_gaussian`: Alignment between vehicle motion heading and road direction.

### 6. Downstream HMM Prediction Format
Candidate predictions are exported to `outputs/predictions/student_test_predictions.csv` with standard schema for emission likelihood in the HMM Viterbi decoder:
- `trip_id`, `obs_id`, `candidate_arc_id`, `timestamp`, `candidate_rank`, `score`, `rank_by_score`, `model`, `is_ground_truth`

### 7. How to Run Training
```bash
# Run full training pipeline with 10 Optuna HPO trials
python -m src.model_training.run_training --trials 10

# Fast debug run on a subset of trips
python -m src.model_training.run_training --debug --trials 3

# Run unit tests
pytest tests/test_model_training.py -v
```

---

## HMM & Viterbi Route Reconstruction

### 1. Architecture
Located in [`src/decoder/`](file:///C:/Users/raopr/Downloads/PathSnap/src/decoder), the sequence decoding engine converts candidate-level ML probability scores into optimal, topologically continuous road routes:
- **`emission_model.py`**: Safely normalizes Random Forest candidate classification scores into conditional emission distributions $P(z_t \mid c_{t, k})$ with $\epsilon$-smoothing.
- **`transition_model.py`**: Models physical road network transitions $P(c_t \mid c_{t-1})$ using directed 1-hop and 2-hop graph connectivity, exponential distance discrepancy penalties $|d_{net} - \Delta s_{gps}|$, and time gap relaxations ($\Delta t \ge 30$s).
- **`viterbi.py`**: Log-space dynamic programming sequence decoder with backpointer traceback and fallback handling for missing candidates, disconnected components, and non-finite costs.
- **`route_reconstruction.py`**: End-to-end benchmark pipeline evaluating all 10 student-test trips.

### 2. Comparative Benchmark (10 Student-Test Trips, 19,160 Observations)

| Metric | Nearest Candidate | Random Forest ML-Only | RF + HMM + Viterbi | Absolute Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Observation Top-1 Accuracy** | 56.22% | 82.31% | **89.88%** | **+7.57%** |
| **Route Continuity (% Connected)** | 80.53% | 89.47% | **98.66%** | **+9.19%** |
| **Invalid Transition Count** | 3,729 | 1,959 | **215** | **-89.0% reduction** |
| **Decoding Time (All 10 Trips)** | ~0.08s | ~0.10s | **4.28s** | ~0.22 ms / obs |

### 3. Per-Trip Results Summary

| Trip ID | Observations | Nearest Top-1 | ML Top-1 | Viterbi Top-1 | ML Continuity | Viterbi Continuity | Invalid Transitions | Decoding Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00** | 2,503 | 47.42% | 92.61% | **96.72%** | 95.16% | **99.84%** | 4 | 0.44s |
| **07** | 1,354 | 53.55% | 91.21% | **99.48%** | 99.19% | **99.93%** | 1 | 0.24s |
| **09** | 1,049 | 42.61% | 81.03% | **86.65%** | 83.68% | **97.52%** | 26 | 0.29s |
| **40** | 2,995 | 51.39% | 88.75% | **96.89%** | 94.89% | **99.63%** | 11 | 0.58s |
| **52** | 262 | 80.53% | 94.27% | **98.86%** | 93.10% | **96.55%** | 9 | 0.05s |
| **56** | 3,958 | 65.03% | 78.85% | **86.13%** | 85.62% | **97.19%** | 111 | 0.97s |
| **65** | 768 | 71.74% | 77.47% | **81.12%** | 88.27% | **98.83%** | 9 | 0.17s |
| **76** | 1,486 | 45.56% | 78.20% | **91.72%** | 83.43% | **98.25%** | 26 | 0.45s |
| **77** | 3,911 | 42.34% | 58.78% | **72.85%** | 89.13% | **99.74%** | 10 | 0.88s |
| **91** | 874 | 62.01% | 81.92% | **88.33%** | 82.25% | **99.08%** | 8 | 0.22s |

### 4. Running Route Reconstruction
```bash
# Execute route reconstruction benchmark across all 10 student trips
python -m src.decoder.route_reconstruction

# Run unit tests (including decoder suite)
pytest tests/test_decoder.py -v
```

---

## Team

- **Namit Krishna Renjith** ([@NamitKrishna](https://github.com/NamitKrishna))
- **Prathyush P Rao** ([@Prathyush172](https://github.com/Prathyush172))
- **Prasanna** ([@prasanna19205](https://github.com/prasanna19205))
- **Nethra Manoj** ([@nethramanoj](https://github.com/nethramanoj))
