# PathSnap

Map-matching and route reconstruction tool designed to process noisy GPS trajectory logs and align them to realistic road networks.

## Overview

GPS traces collected from mobile devices often contain noise, satellite drift, and sampling gaps, especially in dense urban environments. **PathSnap** is a Python-based pipeline created for a college machine learning hackathon to clean, evaluate, and match raw GPS points to road segments using machine learning models, Hidden Markov Models (HMM), and Viterbi sequence decoding.

## Repository Structure

```text
PathSnap/
├── README.md               # Project documentation and setup guide
├── .gitignore              # Ignored files (data, models, virtual environments, caches)
├── requirements.txt        # Python package dependencies
├── data/
│   ├── raw/                # Unprocessed GPS logs and geographic data (untracked)
│   └── processed/          # Cleaned and normalized trajectory datasets (untracked)
├── notebooks/              # Jupyter notebooks for data analysis and experiments
├── src/                    # Source code modules
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
├── tests/                  # Automated unit and smoke tests
├── results/                # Output metrics, benchmark logs, and evaluation reports
├── visualizations/         # Folium HTML map visualizations
├── screenshots/            # Visual previews and architecture diagrams
├── report/                 # Hackathon documentation and project presentation
└── artifacts/              # Exported models and processing artifacts
```

## How It Works

1. **Data Preprocessing**: Raw GPS readings (latitude, longitude, timestamp, speed) are parsed, cleaned of extreme outliers, and projected to local metric coordinates for accurate distance measurements.
2. **Candidate Road Identification**: Spatial indexing (such as R-tree) finds road segments within a candidate radius of each GPS point.
3. **Feature Engineering & ML Scoring**: Features including distance from the segment, difference in direction, and road classification are computed. Classifiers (e.g., XGBoost / Random Forest) score the likelihood of candidate segments.
4. **HMM & Viterbi Decoding**: The route is formulated as a Hidden Markov Model where candidate road segments represent hidden states and GPS points represent observations. The Viterbi algorithm selects the sequence of road segments that best balances observation likelihood with road network connectivity.
5. **Optimization**: Bayesian optimization via Optuna helps determine optimal parameters such as candidate search radius and transition penalties.
6. **API & Visualization**: A FastAPI interface accepts trajectory inputs, while Folium generates interactive maps comparing raw coordinates against reconstructed paths.

## Setup Instructions

### Linux Mint / Ubuntu

#### 1. System Requirements

Ensure Python 3, `pip`, `venv`, and Git are installed:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

#### 2. Virtual Environment Setup

Create and activate a virtual environment in the project directory:

```bash
cd PathSnap
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Windows 11

#### 1. Prerequisites

- **Python 3.10+**: Install from [python.org](https://www.python.org/downloads/) or run:
  ```powershell
  winget install Python.Python.3.11
  ```
  *(Ensure you check the box **"Add python.exe to PATH"** during installation).*
- **Git for Windows**: Install from [git-scm.com](https://git-scm.com/) or run:
  ```powershell
  winget install Git.Git
  ```

#### 2. Virtual Environment Setup

Open **PowerShell** or **Command Prompt** in the `PathSnap` folder:

```powershell
cd PathSnap

# Create a virtual environment
python -m venv .venv

# Activate in PowerShell:
.venv\Scripts\Activate.ps1

# Or activate in Command Prompt (cmd.exe):
.venv\Scripts\activate.bat
```

> **PowerShell Note**: If you encounter an execution policy restriction (`cannot be loaded because running scripts is disabled`), run this once in PowerShell:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project

### Running Tests

Execute unit and smoke tests using `pytest`:

```bash
pytest
```

For detailed test logs:

```bash
pytest -v
```

### Running the API Server

Launch the FastAPI backend service with Uvicorn:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Once running, interactive API docs can be viewed in your browser at `http://127.0.0.1:8000/docs`.

### Prototyping with Notebooks

Launch Jupyter to explore datasets or test algorithms:

```bash
jupyter notebook
```

## Development Environments

- **Linux Mint**:
  - Edit project files using Xed, nano, or any preferred editor (`xed src/preprocessing/__init__.py &`).
  - Code::Blocks project files (`.cbp`, `.layout`, `.depend`) are safely excluded by `.gitignore`.
- **Windows 11**:
  - Teammates can use VS Code, PyCharm, or any text editor of their choice.
  - Windows-specific files (`Thumbs.db`, `desktop.ini`, `$RECYCLE.BIN/`) are safely excluded by `.gitignore`.

## Project Status

This repository provides the foundational architecture and organized structure for the college ML hackathon. Preprocessing routines, model scoring, and decoding components are under active development by the team.

## Team

- **Namit Krishna Renjith** ([@NamitKrishna](https://github.com/NamitKrishna))
- Hackathon Project Collaborators
