"""
test_decoder.py – Unit tests for HMM & Viterbi route reconstruction module.
"""

import numpy as np
import pandas as pd
import pytest

from src.decoder.emission_model import EmissionModel
from src.decoder.transition_model import TransitionModel
from src.decoder.viterbi import ViterbiDecoder, ViterbiResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_arcs_df():
    """Simple linear and branching road network: 0 -> 1 -> 2 -> 3."""
    return pd.DataFrame({
        "arc_id": [0, 1, 2, 3],
        "from_node": [10, 11, 12, 13],
        "to_node": [11, 12, 13, 14],
    })


@pytest.fixture
def sample_trajectory_candidates():
    """Toy trajectory: 3 observations, 2 candidates each."""
    # Obs 0: candidates 0, 1
    # Obs 1: candidates 1, 2
    # Obs 2: candidates 2, 3
    data = [
        {"obs_id": 0, "candidate_arc_id": 0, "candidate_rank": 0, "score": 0.8,
         "norm_proj_pos": 0.5, "arc_length": 50.0, "delta_t": 1.0, "delta_s": 10.0, "is_ground_truth": 1},
        {"obs_id": 0, "candidate_arc_id": 3, "candidate_rank": 1, "score": 0.2,
         "norm_proj_pos": 0.5, "arc_length": 50.0, "delta_t": 1.0, "delta_s": 10.0, "is_ground_truth": 0},

        {"obs_id": 1, "candidate_arc_id": 1, "candidate_rank": 0, "score": 0.9,
         "norm_proj_pos": 0.3, "arc_length": 50.0, "delta_t": 1.0, "delta_s": 15.0, "is_ground_truth": 1},
        {"obs_id": 1, "candidate_arc_id": 3, "candidate_rank": 1, "score": 0.1,
         "norm_proj_pos": 0.3, "arc_length": 50.0, "delta_t": 1.0, "delta_s": 15.0, "is_ground_truth": 0},

        {"obs_id": 2, "candidate_arc_id": 2, "candidate_rank": 0, "score": 0.85,
         "norm_proj_pos": 0.4, "arc_length": 50.0, "delta_t": 1.0, "delta_s": 15.0, "is_ground_truth": 1},
        {"obs_id": 2, "candidate_arc_id": 0, "candidate_rank": 1, "score": 0.15,
         "norm_proj_pos": 0.4, "arc_length": 50.0, "delta_t": 1.0, "delta_s": 15.0, "is_ground_truth": 0},
    ]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Emission Model Tests
# ---------------------------------------------------------------------------

class TestEmissionModel:
    def test_normalization_sums_to_one(self):
        model = EmissionModel()
        scores = np.array([0.2, 0.5, 0.3])
        probs, log_probs = model.normalize_scores(scores)
        assert np.isclose(np.sum(probs), 1.0)
        assert np.all(np.isfinite(log_probs))

    def test_zero_scores_handled_gracefully(self):
        model = EmissionModel()
        scores = np.array([0.0, 0.0, 0.0])
        probs, log_probs = model.normalize_scores(scores)
        assert np.isclose(np.sum(probs), 1.0)
        assert np.all(np.isclose(probs, 1.0 / 3))

    def test_nan_scores_handled_without_crash(self):
        model = EmissionModel()
        scores = np.array([np.nan, 0.8, -0.1])
        probs, log_probs = model.normalize_scores(scores)
        assert np.all(np.isfinite(probs))
        assert np.isclose(np.sum(probs), 1.0)

    def test_log_probs_respect_floor(self):
        model = EmissionModel(log_floor=-50.0)
        scores = np.array([1e-15, 1.0])
        _, log_probs = model.normalize_scores(scores)
        assert np.all(log_probs >= -50.0)


# ---------------------------------------------------------------------------
# Transition Model Tests
# ---------------------------------------------------------------------------

class TestTransitionModel:
    def test_graph_adjacency_built_correctly(self, sample_arcs_df):
        model = TransitionModel(sample_arcs_df)
        # arc 0 (10->11) connects to arc 1 (11->12)
        assert model.is_connected(0, 1)
        assert model.is_connected(1, 2)
        assert model.is_connected(2, 3)
        # arc 0 to 2 is 2-hop
        assert not model.is_connected(0, 2)
        assert 2 in model.succ_2[0]

    def test_same_arc_is_connected(self, sample_arcs_df):
        model = TransitionModel(sample_arcs_df)
        assert model.is_connected(0, 0)
        assert model.is_connected(1, 1)

    def test_disconnected_arcs_penalized(self, sample_arcs_df):
        model = TransitionModel(sample_arcs_df)
        prev_arcs = np.array([0])
        prev_pos = np.array([0.5])
        prev_len = np.array([50.0])

        curr_arcs = np.array([1, 3])  # 1 is connected, 3 is disconnected
        curr_pos = np.array([0.5, 0.5])
        curr_len = np.array([50.0, 50.0])

        log_trans = model.compute_transition_matrix(
            prev_arcs, prev_pos, prev_len, curr_arcs, curr_pos, curr_len,
            delta_t=1.0, delta_s=15.0
        )
        assert log_trans.shape == (1, 2)
        # Connected transition (0 -> 1) must have strictly higher log probability than disconnected (0 -> 3)
        assert log_trans[0, 0] > log_trans[0, 1]

    def test_large_time_gap_relaxes_topology(self, sample_arcs_df):
        model = TransitionModel(sample_arcs_df, max_time_gap=30.0)
        prev_arcs = np.array([0])
        prev_pos = np.array([0.5])
        prev_len = np.array([50.0])

        curr_arcs = np.array([3])  # disconnected
        curr_pos = np.array([0.5])
        curr_len = np.array([50.0])

        # Over 45 seconds gap
        log_trans = model.compute_transition_matrix(
            prev_arcs, prev_pos, prev_len, curr_arcs, curr_pos, curr_len,
            delta_t=45.0, delta_s=50.0
        )
        assert np.isfinite(log_trans[0, 0])
        assert log_trans[0, 0] > model.log_floor


# ---------------------------------------------------------------------------
# Viterbi Decoder Tests
# ---------------------------------------------------------------------------

class TestViterbiDecoder:
    def test_end_to_end_viterbi_decoding(self, sample_arcs_df, sample_trajectory_candidates):
        emission = EmissionModel()
        transition = TransitionModel(sample_arcs_df)
        decoder = ViterbiDecoder(emission, transition)

        result: ViterbiResult = decoder.decode_trajectory(sample_trajectory_candidates, trip_id=1)

        assert result.total_observations == 3
        assert len(result.chosen_arcs) == 3
        # Ground truth sequence is 0 -> 1 -> 2
        assert result.chosen_arcs == [0, 1, 2]
        assert result.route_continuity == 1.0
        assert result.invalid_transition_count == 0
        assert np.isfinite(result.log_likelihood)

    def test_viterbi_corrects_noisy_ml_outlier(self, sample_arcs_df):
        """
        If at t=1, an off-route candidate has a momentarily higher ML score,
        Viterbi's transition model should override it in favor of the topologically connected arc.
        """
        emission = EmissionModel()
        transition = TransitionModel(sample_arcs_df)
        decoder = ViterbiDecoder(emission, transition)

        # t=0: arc 0 (score 0.9)
        # t=1: arc 3 (score 0.7 - disconnected outlier!), arc 1 (score 0.65 - connected!)
        # t=2: arc 2 (score 0.9)
        data = [
            {"obs_id": 0, "candidate_arc_id": 0, "candidate_rank": 0, "score": 0.9,
             "norm_proj_pos": 0.5, "arc_length": 50.0, "delta_t": 1.0, "delta_s": 10.0},
            {"obs_id": 1, "candidate_arc_id": 3, "candidate_rank": 0, "score": 0.70,
             "norm_proj_pos": 0.5, "arc_length": 50.0, "delta_t": 1.0, "delta_s": 10.0},
            {"obs_id": 1, "candidate_arc_id": 1, "candidate_rank": 1, "score": 0.65,
             "norm_proj_pos": 0.5, "arc_length": 50.0, "delta_t": 1.0, "delta_s": 10.0},
            {"obs_id": 2, "candidate_arc_id": 2, "candidate_rank": 0, "score": 0.90,
             "norm_proj_pos": 0.5, "arc_length": 50.0, "delta_t": 1.0, "delta_s": 10.0},
        ]
        df = pd.DataFrame(data)
        result = decoder.decode_trajectory(df, trip_id=99)

        # Viterbi should pick 0 -> 1 -> 2, ignoring disconnected arc 3 despite higher emission score
        assert result.chosen_arcs == [0, 1, 2]
        assert result.route_continuity == 1.0

    def test_empty_candidates_handled_gracefully(self, sample_arcs_df):
        emission = EmissionModel()
        transition = TransitionModel(sample_arcs_df)
        decoder = ViterbiDecoder(emission, transition)

        df = pd.DataFrame(columns=["obs_id", "candidate_arc_id", "candidate_rank", "score", "norm_proj_pos", "arc_length", "delta_t", "delta_s"])
        result = decoder.decode_trajectory(df, trip_id=0)
        assert result.total_observations == 0
        assert len(result.chosen_arcs) == 0
