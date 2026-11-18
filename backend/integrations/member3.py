"""
PATHSNAP Member 3 Integration Adapter

Interface for HMM/Viterbi route reconstruction.
Replace mock implementation with actual Member 3 code when available.
"""
import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from backend.api.models import CandidateSegment, ReconstructedRoute
from backend.integrations.member1 import CandidateSet
from backend.data.road_graph import RoadGraph
from backend.core.exceptions import DecodingError
from backend.core.logging import get_logger
from backend.config import settings

logger = get_logger(__name__)


@dataclass
class ViterbiResult:
    """Result of Viterbi decoding."""
    best_path: list[str]  # arc_ids
    best_score: float  # log probability
    path_probabilities: list[float]  # per-step probabilities
    backpointers: list[list[int]]  # for debugging


class HMVDecoder(ABC):
    """Abstract HMM/Viterbi decoder interface."""

    @abstractmethod
    def decode(
        self,
        candidate_sets: list[CandidateSet],
        road_graph: RoadGraph,
        transition_penalty: float = 1.0,
    ) -> ViterbiResult:
        """
        Run Viterbi decoding to find best route.

        Args:
            candidate_sets: Candidates with probabilities for each GPS point
            road_graph: Road network for transition probabilities
            transition_penalty: Penalty factor for unlikely transitions

        Returns:
            ViterbiResult with best path and diagnostics
        """
        pass


# --- Mock Implementation (Replace with Member 3's actual HMM/Viterbi) ---

class MockHMVDecoder(HMVDecoder):
    """
    Mock HMM/Viterbi decoder for development.

    Member 3 should implement:
    - Emission probabilities from Member 2's model outputs
    - Transition probabilities from road network connectivity
    - Viterbi algorithm in log-space
    - Handling of disconnected transitions
    - Numerical stability (log-sum-exp, etc.)
    - Bayesian optimization integration
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.transition_penalty = self.config.get("transition_penalty", settings.hmm_transition_penalty)
        self.use_log_space = self.config.get("use_log_space", settings.viterbi_use_log_space)

    def decode(
        self,
        candidate_sets: list[CandidateSet],
        road_graph: RoadGraph,
        transition_penalty: float = None,
    ) -> ViterbiResult:
        start_time = time.perf_counter()

        penalty = transition_penalty or self.transition_penalty

        logger.info(
            "Running Viterbi decoding",
            extra={
                "observations": len(candidate_sets),
                "penalty": penalty,
            }
        )

        if not candidate_sets:
            raise DecodingError("No candidate sets provided")

        # Filter out observations with no candidates
        valid_sets = [cs for cs in candidate_sets if cs.candidates]
        if not valid_sets:
            raise DecodingError("No candidates available for any observation")

        if len(valid_sets) != len(candidate_sets):
            logger.warning(
                "Some observations have no candidates",
                extra={"total": len(candidate_sets), "valid": len(valid_sets)}
            )

        # Build emission matrix (log probabilities)
        # Shape: (n_observations, max_candidates_per_obs)
        max_cands = max(len(cs.candidates) for cs in valid_sets)
        n_obs = len(valid_sets)

        # Pad with -inf for missing candidates
        emissions = np.full((n_obs, max_cands), -np.inf)
        arc_ids = []  # arc_ids[obs_idx][cand_idx] = arc_id

        for obs_idx, cs in enumerate(valid_sets):
            obs_arc_ids = []
            for cand_idx, candidate in enumerate(cs.candidates):
                # Emission = model probability (convert to log)
                prob = max(candidate.probability, 1e-10)
                emissions[obs_idx, cand_idx] = math.log(prob)
                obs_arc_ids.append(candidate.arc_id)
            arc_ids.append(obs_arc_ids)

        # Build transition matrix
        # For simplicity, use a simplified transition model
        # Real implementation would use road_graph connectivity
        transitions = self._build_transition_matrix(arc_ids, road_graph, penalty)

        # Run Viterbi
        result = self._viterbi(emissions, transitions)

        # Map back to arc_ids
        best_arc_ids = [arc_ids[t][result.best_path[t]] for t in range(n_obs)]

        # Get route geometry from road graph
        coordinates = road_graph.get_route_geometry(best_arc_ids)

        # Compute total distance
        total_distance = 0.0
        for arc_id in best_arc_ids:
            edge = road_graph.get_edge_by_arc_id(arc_id)
            if edge and "length_m" in edge:
                total_distance += edge["length_m"]

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Viterbi decoding complete",
            extra={
                "duration_ms": duration_ms,
                "path_length": len(best_arc_ids),
                "best_score": result.best_score,
            }
        )

        return ViterbiResult(
            best_path=best_arc_ids,
            best_score=result.best_score,
            path_probabilities=result.path_probabilities,
            backpointers=result.backpointers,
        )

    def _build_transition_matrix(
        self,
        arc_ids: list[list[str]],
        road_graph: RoadGraph,
        penalty: float,
    ) -> list[np.ndarray]:
        """Build transition log-probability matrices between consecutive observations."""
        n_obs = len(arc_ids)
        transitions = []

        for t in range(n_obs - 1):
            n_curr = len(arc_ids[t])
            n_next = len(arc_ids[t + 1])

            # Transition matrix: (n_curr, n_next)
            trans = np.full((n_curr, n_next), -np.inf)

            for i, arc_curr in enumerate(arc_ids[t]):
                for j, arc_next in enumerate(arc_ids[t + 1]):
                    # Check connectivity
                    connected = self._are_connected(arc_curr, arc_next, road_graph)

                    if connected:
                        # Base transition probability
                        trans[i, j] = math.log(0.8)  # High prob for connected
                    else:
                        # Penalty for disconnected
                        trans[i, j] = math.log(0.01 * penalty)

            transitions.append(trans)

        return transitions

    def _are_connected(self, arc_1: str, arc_2: str, road_graph: RoadGraph) -> bool:
        """Check if two arcs are topologically connected."""
        edge_1 = road_graph.get_edge_by_arc_id(arc_1)
        edge_2 = road_graph.get_edge_by_arc_id(arc_2)

        if not edge_1 or not edge_2:
            return False

        # For mock: check if they share a node
        # Real implementation would use actual graph topology
        return True  # Simplified

    def _viterbi(self, emissions: np.ndarray, transitions: list[np.ndarray]) -> ViterbiResult:
        """
        Viterbi algorithm in log space.

        Args:
            emissions: (n_obs, n_states) log emission probabilities
            transitions: list of (n_states_t, n_states_t+1) log transition probabilities

        Returns:
            ViterbiResult
        """
        n_obs, max_states = emissions.shape

        # DP tables
        # delta[t, s] = max log prob of path ending in state s at time t
        delta = np.full((n_obs, max_states), -np.inf)
        # psi[t, s] = best previous state
        psi = np.full((n_obs, max_states), -1, dtype=int)

        # Initialization (t=0)
        delta[0, :] = emissions[0, :]

        # Recursion
        for t in range(1, n_obs):
            n_curr = emissions.shape[1]  # max_states
            n_prev = delta[t - 1, :].shape[0]  # max_states (same)

            trans = transitions[t - 1] if t - 1 < len(transitions) else None

            for s in range(n_curr):
                if emissions[t, s] == -np.inf:
                    continue

                # Find best previous state
                best_score = -np.inf
                best_prev = -1

                for s_prev in range(n_prev):
                    if delta[t - 1, s_prev] == -np.inf:
                        continue

                    # Transition score
                    if trans is not None and s_prev < trans.shape[0] and s < trans.shape[1]:
                        trans_score = trans[s_prev, s]
                    else:
                        trans_score = math.log(0.01)  # Default penalty

                    score = delta[t - 1, s_prev] + trans_score

                    if score > best_score:
                        best_score = score
                        best_prev = s_prev

                if best_prev >= 0:
                    delta[t, s] = best_score + emissions[t, s]
                    psi[t, s] = best_prev

        # Termination
        best_last_state = np.argmax(delta[n_obs - 1, :])
        best_score = delta[n_obs - 1, best_last_state]

        # Backtrack
        best_path = [best_last_state]
        for t in range(n_obs - 1, 0, -1):
            prev = psi[t, best_path[-1]]
            best_path.append(prev if prev >= 0 else 0)

        best_path.reverse()

        # Path probabilities (for diagnostics)
        path_probs = []
        for t, s in enumerate(best_path):
            if t < n_obs and s < emissions.shape[1]:
                path_probs.append(math.exp(emissions[t, s]) if emissions[t, s] > -np.inf else 0.0)
            else:
                path_probs.append(0.0)

        return ViterbiResult(
            best_path=best_path,
            best_score=float(best_score),
            path_probabilities=path_probs,
            backpointers=psi.tolist(),
        )


# --- Factory Function ---

def create_hmm_decoder(config: Optional[dict] = None) -> HMVDecoder:
    """Create HMM/Viterbi decoder instance. Replace with Member 3's implementation."""
    # TODO: Replace with actual Member 3 implementation
    # from member3.decoder import ProductionHMVDecoder
    # return ProductionHMVDecoder(config)
    return MockHMVDecoder(config)