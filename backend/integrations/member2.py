"""
PATHSNAP Member 2 Integration Adapter

Interface for ML model predictions on candidate road segments.
Replace mock implementation with actual Member 2 trained model when available.
"""
import time
import joblib
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from backend.api.models import CandidateSegment
from backend.integrations.member1 import CandidateSet
from backend.core.exceptions import ModelPredictionError, ModelNotLoadedError
from backend.core.logging import get_logger
from backend.config import settings

logger = get_logger(__name__)


class MLModel(ABC):
    """Abstract ML model interface for candidate scoring."""

    @abstractmethod
    def predict(self, candidates: list[CandidateSet]) -> list[CandidateSet]:
        """
        Predict probabilities for candidate road segments.

        Args:
            candidates: List of CandidateSet, each containing a GPS point and its candidates

        Returns:
            Same structure with probability field populated on each CandidateSegment
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names expected by the model."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata (version, training date, metrics, etc.)."""
        pass


# --- Mock Implementation (Replace with Member 2's actual model) ---

class MockMLModel(MLModel):
    """
    Mock ML model for development.

    Member 2 should implement:
    - Load trained model (XGBoost, Random Forest, etc.)
    - Feature engineering pipeline matching training
    - Probability calibration
    - Feature name/order validation
    - Batch prediction for efficiency
    """

    def __init__(self, model_path: Optional[Path] = None, config: Optional[dict] = None):
        self.model_path = model_path or settings.model_path
        self.config = config or {}
        self._model = None
        self._scaler = None
        self._feature_names = None
        self._model_info = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load model artifact. Replace with Member 2's loading logic."""
        try:
            if self.model_path.exists():
                logger.info("Loading model from disk", extra={"path": str(self.model_path)})
                artifact = joblib.load(self.model_path)

                # Handle different artifact formats
                if isinstance(artifact, dict):
                    self._model = artifact.get("model")
                    self._scaler = artifact.get("scaler")
                    self._feature_names = artifact.get("feature_names", [])
                    self._model_info = artifact.get("metadata", {})
                else:
                    self._model = artifact
                    self._feature_names = getattr(artifact, "feature_names_in_", [])

                logger.info("Model loaded successfully")
            else:
                logger.warning("Model file not found, using mock", extra={"path": str(self.model_path)})
                self._init_mock_model()
        except Exception as e:
            logger.warning("Failed to load model, using mock", extra={"error": str(e)})
            self._init_mock_model()

    def _init_mock_model(self) -> None:
        """Initialize a mock model for development."""
        self._model = None
        self._scaler = None
        self._feature_names = [
            "distance_m",
            "heading_diff_deg",
            "speed",
            "speed_smoothed",
            "heading",
            "heading_smoothed",
            "accuracy",
            "dt",
            "acceleration",
            "road_class_primary",
            "road_class_secondary",
            "road_class_tertiary",
            "road_class_residential",
            "road_class_unclassified",
        ]
        self._model_info = {
            "type": "mock",
            "version": "dev-0.1",
            "features": len(self._feature_names),
            "note": "Replace with Member 2's trained model",
        }

    def predict(self, candidates: list[CandidateSet]) -> list[CandidateSet]:
        start_time = time.perf_counter()
        total_candidates = sum(len(cs.candidates) for cs in candidates)

        logger.info("Running model prediction", extra={"candidate_sets": len(candidates), "total_candidates": total_candidates})

        if total_candidates == 0:
            logger.warning("No candidates to predict")
            return candidates

        # Extract features for all candidates
        feature_matrix = []
        candidate_refs = []  # (candidate_set_idx, candidate_idx)

        for cs_idx, candidate_set in enumerate(candidates):
            for cand_idx, candidate in enumerate(candidate_set.candidates):
                features = self._extract_features(candidate, candidate_set.gps_point)
                feature_matrix.append(features)
                candidate_refs.append((cs_idx, cand_idx))

        X = np.array(feature_matrix)

        # Predict probabilities
        if self._model is not None:
            try:
                # Apply scaler if available
                if self._scaler is not None:
                    X = self._scaler.transform(X)

                # Get probabilities (assuming binary classification or multiclass)
                if hasattr(self._model, "predict_proba"):
                    probs = self._model.predict_proba(X)
                    # For binary, take positive class; for multiclass, take max
                    if probs.shape[1] == 2:
                        prob_values = probs[:, 1]
                    else:
                        prob_values = probs.max(axis=1)
                else:
                    # Fallback for models without predict_proba
                    prob_values = self._model.predict(X)
                    # Normalize to [0, 1] if needed
                    prob_values = np.clip(prob_values, 0, 1)

            except Exception as e:
                logger.error("Model prediction failed, using fallback", extra={"error": str(e)})
                prob_values = self._mock_predict(X)
        else:
            prob_values = self._mock_predict(X)

        # Assign probabilities back to candidates
        for (cs_idx, cand_idx), prob in zip(candidate_refs, prob_values):
            candidates[cs_idx].candidates[cand_idx].probability = float(prob)

        # Sort candidates by probability (descending) within each set
        for cs in candidates:
            cs.candidates.sort(key=lambda c: c.probability, reverse=True)

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Model prediction complete", extra={"duration_ms": duration_ms})

        return candidates

    def _mock_predict(self, X: np.ndarray) -> np.ndarray:
        """Mock prediction based on distance and heading difference."""
        # X columns: distance_m, heading_diff_deg, speed, speed_smoothed, heading, heading_smoothed, accuracy, dt, acceleration, road_class_*
        distance = X[:, 0] if X.shape[1] > 0 else np.zeros(len(X))
        heading_diff = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))

        # Probability decreases with distance and heading difference
        # Distance: exponential decay, Heading: cosine similarity
        dist_score = np.exp(-distance / 20.0)  # 20m characteristic distance
        heading_score = np.cos(np.radians(np.clip(heading_diff, 0, 180)))
        heading_score = (heading_score + 1) / 2  # Map to [0, 1]

        # Road class boost (last 5 columns are one-hot)
        road_class_boost = 0
        if X.shape[1] >= 14:
            # Primary/secondary get boost
            road_class_boost = 0.1 * X[:, 9] + 0.05 * X[:, 10]  # primary, secondary

        prob = 0.6 * dist_score + 0.3 * heading_score + 0.1 * road_class_boost
        prob = np.clip(prob + np.random.normal(0, 0.05, len(prob)), 0.01, 0.99)

        return prob

    def _extract_features(self, candidate: CandidateSegment, gps_point) -> list[float]:
        """Extract feature vector for a single candidate."""
        features = []

        # Distance feature
        features.append(candidate.distance_m or 0.0)

        # Heading difference feature
        features.append(candidate.heading_diff_deg or 180.0)

        # GPS point features
        gp = gps_point
        features.append(gp.original.speed or 0.0)
        features.append(gp.speed_smoothed or 0.0)
        features.append(gp.original.heading or 0.0)
        features.append(gp.heading_smoothed or 0.0)
        features.append(gp.original.accuracy or 0.0)
        features.append(gp.features.get("dt", 1.0))
        features.append(gp.features.get("acceleration", 0.0))

        # Road class one-hot encoding
        road_class = (candidate.road_class or "unclassified").lower()
        class_map = {
            "primary": [1, 0, 0, 0, 0],
            "secondary": [0, 1, 0, 0, 0],
            "tertiary": [0, 0, 1, 0, 0],
            "residential": [0, 0, 0, 1, 0],
            "unclassified": [0, 0, 0, 0, 1],
        }
        features.extend(class_map.get(road_class, [0, 0, 0, 0, 1]))

        # Ensure consistent length
        while len(features) < len(self._feature_names):
            features.append(0.0)

        return features[:len(self._feature_names)]

    def get_feature_names(self) -> list[str]:
        return self._feature_names.copy()

    def get_model_info(self) -> dict[str, Any]:
        return self._model_info.copy()


# --- Factory Function ---

def create_ml_model(model_path: Optional[Path] = None, config: Optional[dict] = None) -> MLModel:
    """Create ML model instance. Replace with Member 2's implementation."""
    # TODO: Replace with actual Member 2 implementation
    # from member2.models import ProductionMLModel
    # return ProductionMLModel(model_path, config)
    return MockMLModel(model_path, config)