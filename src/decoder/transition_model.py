"""
transition_model.py – Transition probability model for PATHSNAP HMM.

Models topological and kinematic state transitions between candidate road segments:
    P(c_{t, j} | c_{t-1, i})

Accounts for:
- Directed road network connectivity (1-hop, 2-hop successors)
- Same-arc transitions and progression along the segment
- GPS displacement (delta_s) vs topological network distance (d_net)
- Time gaps (delta_t >= 30s) where strict topology is relaxed
- Disconnected transitions penalized with safe non-zero fallbacks
"""

from typing import Dict, Set, List, Optional
from collections import defaultdict
import numpy as np
import pandas as pd


class TransitionModel:
    """
    Computes transition probabilities between candidate road segments.
    """

    def __init__(
        self,
        arcs_df: pd.DataFrame,
        beta: float = 15.0,
        p_same: float = 0.95,
        p_conn1: float = 0.90,
        p_conn2: float = 0.50,
        p_fallback: float = 1e-4,
        max_time_gap: float = 30.0,
        log_floor: float = -100.0,
    ):
        """
        Parameters
        ----------
        arcs_df : DataFrame with ['arc_id', 'from_node', 'to_node']
        beta : scale parameter for exponential distance error (meters)
        p_same : weight for remaining on the same arc
        p_conn1 : weight for direct 1-hop connected arc
        p_conn2 : weight for 2-hop connected arc
        p_fallback : penalty for disconnected arcs
        max_time_gap : time gap (seconds) beyond which strict topology is relaxed
        log_floor : minimum allowed log-probability
        """
        self.beta = beta
        self.p_same = p_same
        self.p_conn1 = p_conn1
        self.p_conn2 = p_conn2
        self.p_fallback = p_fallback
        self.max_time_gap = max_time_gap
        self.log_floor = log_floor

        # Build graph adjacency
        self._build_graph(arcs_df)

    def _build_graph(self, arcs_df: pd.DataFrame):
        """Precompute directed 1-hop and 2-hop road network successors."""
        u_vals = arcs_df["from_node"].values
        v_vals = arcs_df["to_node"].values
        arc_ids = arcs_df["arc_id"].values if "arc_id" in arcs_df.columns else np.arange(len(arcs_df))

        node_out = defaultdict(list)
        for aid, u in zip(arc_ids, u_vals):
            node_out[u].append(aid)

        # 1-hop successors: arcs starting from head node v
        self.succ_1: Dict[int, Set[int]] = {}
        for aid, v in zip(arc_ids, v_vals):
            if v in node_out:
                self.succ_1[aid] = set(node_out[v])

        # 2-hop successors: arcs starting from head nodes of 1-hop successors
        self.succ_2: Dict[int, Set[int]] = {}
        for aid, direct_succs in self.succ_1.items():
            s2 = set()
            for s1 in direct_succs:
                if s1 in self.succ_1:
                    s2.update(self.succ_1[s1])
            s2.difference_update(direct_succs)
            s2.discard(aid)
            if s2:
                self.succ_2[aid] = s2

    def is_connected(self, arc_1: int, arc_2: int) -> bool:
        """Return True if arc_1 equals arc_2 or is directly 1-hop connected to arc_2."""
        if arc_1 == arc_2:
            return True
        return arc_2 in self.succ_1.get(arc_1, ())

    def compute_transition_matrix(
        self,
        prev_arcs: np.ndarray,
        prev_pos: np.ndarray,
        prev_len: np.ndarray,
        curr_arcs: np.ndarray,
        curr_pos: np.ndarray,
        curr_len: np.ndarray,
        delta_t: float,
        delta_s: float,
    ) -> np.ndarray:
        """
        Compute the log transition probability matrix of shape (K_prev, K_curr).

        Parameters
        ----------
        prev_arcs, curr_arcs : 1D arrays of candidate arc IDs
        prev_pos, curr_pos : 1D arrays of normalized positions along arc [0, 1]
        prev_len, curr_len : 1D arrays of arc lengths (meters)
        delta_t : elapsed time between observations (seconds)
        delta_s : GPS displacement between observations (meters)

        Returns
        -------
        log_trans : 2D array of shape (K_prev, K_curr) with log transition probabilities
        """
        K_prev = len(prev_arcs)
        K_curr = len(curr_arcs)

        if K_prev == 0 or K_curr == 0:
            return np.empty((K_prev, K_curr), dtype=np.float64)

        # Large GPS time gap (tunnel, signal dropout, etc.): relax strict topology
        if delta_t >= self.max_time_gap:
            # Soft transition based on expected travel velocity
            expected_v = delta_s / max(1.0, delta_t)
            # Velocity plausibility (< 40 m/s or 144 km/h)
            prob_gap = max(self.p_fallback, np.exp(-max(0.0, expected_v - 30.0) / 10.0))
            return np.full((K_prev, K_curr), np.log(max(prob_gap, 1e-12)))

        # Pairwise candidate comparison
        # same arc mask: (K_prev, K_curr)
        same_mask = (prev_arcs[:, None] == curr_arcs[None, :])

        # Precompute boolean masks for 1-hop and 2-hop
        conn1_mask = np.zeros((K_prev, K_curr), dtype=bool)
        conn2_mask = np.zeros((K_prev, K_curr), dtype=bool)

        for i, a1 in enumerate(prev_arcs):
            succs1 = self.succ_1.get(a1, ())
            succs2 = self.succ_2.get(a1, ())
            if succs1:
                for j, a2 in enumerate(curr_arcs):
                    if a2 in succs1:
                        conn1_mask[i, j] = True
            if succs2:
                for j, a2 in enumerate(curr_arcs):
                    if a2 in succs2:
                        conn2_mask[i, j] = True

        # Network distances:
        # Same arc: forward progression (pos2 - pos1) * L1
        pos_diff = curr_pos[None, :] - prev_pos[:, None]
        dnet_same = np.maximum(0.0, pos_diff * prev_len[:, None])

        # 1-hop: remaining on arc 1 + entry into arc 2
        dnet_conn1 = (1.0 - prev_pos[:, None]) * prev_len[:, None] + curr_pos[None, :] * curr_len[None, :]

        # 2-hop: remaining on arc 1 + nominal mid-arc length (30m) + entry into arc 2
        dnet_conn2 = (1.0 - prev_pos[:, None]) * prev_len[:, None] + 30.0 + curr_pos[None, :] * curr_len[None, :]

        # Compute probabilities
        probs = np.full((K_prev, K_curr), self.p_fallback * np.exp(-delta_s / 100.0))

        # Backward motion on same one-way arc is penalized
        forward_same = same_mask & (pos_diff >= -0.15)
        backward_same = same_mask & (pos_diff < -0.15)

        if np.any(forward_same):
            probs[forward_same] = self.p_same * np.exp(-np.abs(dnet_same[forward_same] - delta_s) / self.beta)
        if np.any(backward_same):
            probs[backward_same] = (self.p_same * 0.05) * np.exp(-np.abs(dnet_same[backward_same] - delta_s) / self.beta)

        if np.any(conn1_mask):
            probs[conn1_mask] = self.p_conn1 * np.exp(-np.abs(dnet_conn1[conn1_mask] - delta_s) / self.beta)

        if np.any(conn2_mask):
            probs[conn2_mask] = self.p_conn2 * np.exp(-np.abs(dnet_conn2[conn2_mask] - delta_s) / self.beta)

        # Floor and log
        log_trans = np.log(np.maximum(probs, 1e-12))
        return np.maximum(log_trans, self.log_floor)
