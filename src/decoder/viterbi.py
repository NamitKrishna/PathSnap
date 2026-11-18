"""
viterbi.py – Log-space Viterbi sequence decoding for PATHSNAP map-matching.

Finds the most likely sequence of candidate road segments that jointly
maximizes emission probabilities (ML scores) and transition probabilities
(directed graph connectivity and kinematic displacement).
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd

from src.decoder.emission_model import EmissionModel
from src.decoder.transition_model import TransitionModel


@dataclass
class ViterbiResult:
    """Container for Viterbi decoding outputs and diagnostic metrics."""
    trip_id: Any
    chosen_arcs: List[int]
    chosen_candidate_ranks: List[int]
    observation_ids: List[int]
    log_likelihood: float
    total_observations: int
    fallback_count: int
    invalid_transition_count: int
    route_continuity: float
    decoding_time_s: float


class ViterbiDecoder:
    """
    Decodes the optimal sequence of road segments along a GPS trajectory
    using the log-space Viterbi algorithm with backpointer traceback.
    """

    def __init__(self, emission_model: EmissionModel, transition_model: TransitionModel):
        self.emission_model = emission_model
        self.transition_model = transition_model

    def decode_trajectory(
        self,
        observations_candidate_df: pd.DataFrame,
        trip_id: Any = None,
    ) -> ViterbiResult:
        """
        Execute Viterbi route reconstruction on candidate data for a single trip.

        Parameters
        ----------
        observations_candidate_df : DataFrame containing candidate predictions for one trip,
            grouped by obs_id, with columns:
            ['obs_id', 'candidate_arc_id', 'candidate_rank', 'score', 'norm_proj_pos',
             'arc_length', 'delta_t', 'delta_s']
        trip_id : identifier for the trip

        Returns
        -------
        ViterbiResult containing chosen sequence, metrics, and diagnostics.
        """
        start_time = time.perf_counter()

        # Sort strictly by obs_id and candidate_rank
        df = observations_candidate_df.sort_values(["obs_id", "candidate_rank"]).copy()
        obs_groups = [group for _, group in df.groupby("obs_id", sort=True)]
        T = len(obs_groups)

        if T == 0:
            return ViterbiResult(
                trip_id=trip_id,
                chosen_arcs=[],
                chosen_candidate_ranks=[],
                observation_ids=[],
                log_likelihood=0.0,
                total_observations=0,
                fallback_count=0,
                invalid_transition_count=0,
                route_continuity=1.0,
                decoding_time_s=0.0,
            )

        fallback_count = 0
        obs_ids = []

        # Process each observation's candidate set
        obs_candidates = []
        for g in obs_groups:
            cdata = self.emission_model.process_observation_candidates(g)
            dt = float(g["delta_t"].iloc[0]) if "delta_t" in g.columns and len(g) > 0 else 1.0
            ds = float(g["delta_s"].iloc[0]) if "delta_s" in g.columns and len(g) > 0 else 0.0
            cdata["delta_t"] = dt
            cdata["delta_s"] = ds
            cdata["obs_id"] = int(g["obs_id"].iloc[0])
            cdata["ranks"] = g["candidate_rank"].values if "candidate_rank" in g.columns else np.arange(len(g))
            obs_ids.append(cdata["obs_id"])
            obs_candidates.append(cdata)

        # Trellis tables
        # V[t] stores log-likelihood for each candidate at time t
        # backpointers[t] stores index of best candidate at time t-1
        V: List[np.ndarray] = []
        backpointers: List[np.ndarray] = []

        # -------------------------------------------------------------
        # Step 0: Initialization
        # -------------------------------------------------------------
        c0 = obs_candidates[0]
        K0 = len(c0["candidate_arc_ids"])
        if K0 == 0:
            # Fallback: empty candidate set at t=0
            fallback_count += 1
            V.append(np.array([0.0]))
            backpointers.append(np.array([0]))
        else:
            V.append(c0["log_probs"].copy())
            backpointers.append(np.zeros(K0, dtype=np.int32))

        # -------------------------------------------------------------
        # Step 1 to T-1: Forward Viterbi recursion
        # -------------------------------------------------------------
        for t in range(1, T):
            c_prev = obs_candidates[t - 1]
            c_curr = obs_candidates[t]
            K_curr = len(c_curr["candidate_arc_ids"])

            # Fallback 1: current observation has 0 candidates
            if K_curr == 0:
                fallback_count += 1
                # Inherit previous state
                V.append(V[-1].copy())
                backpointers.append(np.arange(len(V[-1]), dtype=np.int32))
                continue

            K_prev = len(c_prev["candidate_arc_ids"])
            # Fallback 2: previous observation had 0 candidates
            if K_prev == 0:
                fallback_count += 1
                V.append(c_curr["log_probs"].copy())
                backpointers.append(np.zeros(K_curr, dtype=np.int32))
                continue

            # Compute log transition matrix (K_prev, K_curr)
            log_trans = self.transition_model.compute_transition_matrix(
                prev_arcs=c_prev["candidate_arc_ids"],
                prev_pos=c_prev["norm_proj_pos"],
                prev_len=c_prev["arc_length"],
                curr_arcs=c_curr["candidate_arc_ids"],
                curr_pos=c_curr["norm_proj_pos"],
                curr_len=c_curr["arc_length"],
                delta_t=c_curr["delta_t"],
                delta_s=c_curr["delta_s"],
            )

            # Combined cost: V_prev (K_prev, 1) + log_trans (K_prev, K_curr)
            cost_matrix = V[-1][:, None] + log_trans

            # Fallback 3: check for non-finite costs (underflow)
            if not np.any(np.isfinite(cost_matrix)):
                fallback_count += 1
                # Fallback to pure emission probabilities
                best_prev = np.zeros(K_curr, dtype=np.int32)
                best_cost = c_curr["log_probs"].copy()
            else:
                best_prev = np.argmax(cost_matrix, axis=0)
                best_cost = np.max(cost_matrix, axis=0) + c_curr["log_probs"]

            V.append(best_cost)
            backpointers.append(best_prev)

        # -------------------------------------------------------------
        # Step T: Termination & Traceback
        # -------------------------------------------------------------
        final_V = V[-1]
        best_last_idx = int(np.argmax(final_V))
        total_log_likelihood = float(final_V[best_last_idx])

        # Backtrack pointers from T-1 down to 0
        chosen_indices = [best_last_idx]
        for t in range(T - 1, 0, -1):
            curr_idx = chosen_indices[-1]
            prev_idx = int(backpointers[t][curr_idx])
            chosen_indices.append(prev_idx)

        chosen_indices.reverse()

        # Extract chosen arc IDs and candidate ranks
        chosen_arcs: List[int] = []
        chosen_ranks: List[int] = []

        for t in range(T):
            c_info = obs_candidates[t]
            idx = chosen_indices[t]
            if len(c_info["candidate_arc_ids"]) > idx:
                chosen_arcs.append(int(c_info["candidate_arc_ids"][idx]))
                chosen_ranks.append(int(c_info["ranks"][idx]))
            elif len(c_info["candidate_arc_ids"]) > 0:
                chosen_arcs.append(int(c_info["candidate_arc_ids"][0]))
                chosen_ranks.append(int(c_info["ranks"][0]))
            else:
                # Missing candidate fallback
                chosen_arcs.append(-1)
                chosen_ranks.append(-1)

        # -------------------------------------------------------------
        # Diagnostic Metrics: Continuity & Invalid Transitions
        # -------------------------------------------------------------
        invalid_transitions = 0
        valid_transitions = 0
        total_transitions = max(1, T - 1)

        for t in range(T - 1):
            a1 = chosen_arcs[t]
            a2 = chosen_arcs[t + 1]
            if a1 == -1 or a2 == -1:
                invalid_transitions += 1
            elif self.transition_model.is_connected(a1, a2):
                valid_transitions += 1
            else:
                invalid_transitions += 1

        route_continuity = valid_transitions / total_transitions
        decoding_time = time.perf_counter() - start_time

        return ViterbiResult(
            trip_id=trip_id,
            chosen_arcs=chosen_arcs,
            chosen_candidate_ranks=chosen_ranks,
            observation_ids=obs_ids,
            log_likelihood=total_log_likelihood,
            total_observations=T,
            fallback_count=fallback_count,
            invalid_transition_count=invalid_transitions,
            route_continuity=route_continuity,
            decoding_time_s=decoding_time,
        )
