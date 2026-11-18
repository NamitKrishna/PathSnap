"""
route_reconstruction.py – Full Route Reconstruction & Benchmarking Pipeline.

Evaluates and compares:
A. Nearest Candidate Baseline
B. Random Forest ML-Only Predictions
C. Random Forest + HMM + Viterbi Reconstructed Routes

Across all 10 student-testing trips, measuring:
- Observation-level Top-1 accuracy
- Route continuity
- Invalid transition counts
- Fallback counts
- Execution/decoding time
"""

import os
import glob
import json
import time
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.decoder.emission_model import EmissionModel
from src.decoder.transition_model import TransitionModel
from src.decoder.viterbi import ViterbiDecoder, ViterbiResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class RouteReconstructionPipeline:
    """
    Orchestrates end-to-end HMM Viterbi route reconstruction
    and comparative benchmark on the student-test partition.
    """

    def __init__(
        self,
        predictions_path: str = "outputs/predictions/student_test_predictions.csv",
        features_dir: str = "data/features/student_test",
        processed_dir: str = "data/processed/student_test",
        output_dir: str = "outputs/decoder",
        beta: float = 15.0,
        p_same: float = 0.95,
        p_conn1: float = 0.90,
        p_conn2: float = 0.50,
        p_fallback: float = 1e-4,
    ):
        self.predictions_path = predictions_path
        self.features_dir = features_dir
        self.processed_dir = processed_dir
        self.output_dir = output_dir

        self.beta = beta
        self.p_same = p_same
        self.p_conn1 = p_conn1
        self.p_conn2 = p_conn2
        self.p_fallback = p_fallback

        os.makedirs(self.output_dir, exist_ok=True)

    def load_predictions(self) -> pd.DataFrame:
        """Load Random Forest candidate predictions for student-test partition."""
        if not os.path.exists(self.predictions_path):
            raise FileNotFoundError(f"Predictions not found at {self.predictions_path}. Run training first.")
        logger.info(f"Loading predictions from {self.predictions_path} ...")
        df = pd.read_csv(self.predictions_path)
        return df

    def run_trip(
        self, trip_id: int, trip_preds: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run Nearest, ML-Only, and HMM+Viterbi on a single trip.
        """
        trip_str = f"{int(trip_id):08d}"
        
        # 1. Load spatial candidate metadata
        cands_path = os.path.join(self.features_dir, f"{trip_str}_candidates.csv")
        if not os.path.exists(cands_path):
            raise FileNotFoundError(f"Candidate features not found: {cands_path}")
        
        cands_cols = [
            "obs_id", "candidate_arc_id", "perp_distance", "norm_proj_pos",
            "arc_length", "delta_t", "delta_s", "timestamp"
        ]
        cands_df = pd.read_csv(cands_path, usecols=lambda c: c in cands_cols)

        # Merge candidate predictions with spatial features
        merged = pd.merge(
            trip_preds,
            cands_df,
            on=["obs_id", "candidate_arc_id"],
            how="inner",
            suffixes=("", "_feat")
        )

        # 2. Load road network topology
        arcs_path = os.path.join(self.processed_dir, trip_str, f"{trip_str}_validated_arcs.csv")
        if not os.path.exists(arcs_path):
            raise FileNotFoundError(f"Validated arcs not found: {arcs_path}")
        arcs_df = pd.read_csv(arcs_path)

        # 3. Instantiate decoder components
        emission_model = EmissionModel()
        transition_model = TransitionModel(
            arcs_df=arcs_df,
            beta=self.beta,
            p_same=self.p_same,
            p_conn1=self.p_conn1,
            p_conn2=self.p_conn2,
            p_fallback=self.p_fallback,
        )
        decoder = ViterbiDecoder(emission_model, transition_model)

        # 4. Extract Ground Truth per observation
        # Identify ground truth arc for each obs_id
        gt_map = {}
        for obs_id, group in merged.groupby("obs_id"):
            gt_row = group[group["is_ground_truth"] == 1]
            if len(gt_row) > 0:
                gt_map[int(obs_id)] = int(gt_row["candidate_arc_id"].iloc[0])
            else:
                gt_map[int(obs_id)] = -1

        sorted_obs_ids = sorted(gt_map.keys())
        N_obs = len(sorted_obs_ids)

        # -----------------------------------------------------------------
        # Strategy A: Nearest Candidate Baseline (minimum perp_distance / candidate_rank==0)
        # -----------------------------------------------------------------
        t0_nearest = time.perf_counter()
        nearest_map = {}
        for obs_id, group in merged.groupby("obs_id"):
            # candidate_rank 0 is the nearest candidate
            nearest_row = group.sort_values("candidate_rank").iloc[0]
            nearest_map[int(obs_id)] = int(nearest_row["candidate_arc_id"])
        nearest_time = time.perf_counter() - t0_nearest

        nearest_arcs = [nearest_map[oid] for oid in sorted_obs_ids]
        nearest_correct = sum(nearest_arcs[i] == gt_map[sorted_obs_ids[i]] for i in range(N_obs))
        nearest_top1 = nearest_correct / max(1, N_obs)

        nearest_invalid = sum(
            not transition_model.is_connected(nearest_arcs[i], nearest_arcs[i + 1])
            for i in range(N_obs - 1)
        )
        nearest_continuity = (N_obs - 1 - nearest_invalid) / max(1, N_obs - 1)

        # -----------------------------------------------------------------
        # Strategy B: Random Forest ML-Only Predictions (highest ML score)
        # -----------------------------------------------------------------
        t0_ml = time.perf_counter()
        ml_map = {}
        for obs_id, group in merged.groupby("obs_id"):
            best_ml_row = group.sort_values("score", ascending=False).iloc[0]
            ml_map[int(obs_id)] = int(best_ml_row["candidate_arc_id"])
        ml_time = time.perf_counter() - t0_ml

        ml_arcs = [ml_map[oid] for oid in sorted_obs_ids]
        ml_correct = sum(ml_arcs[i] == gt_map[sorted_obs_ids[i]] for i in range(N_obs))
        ml_top1 = ml_correct / max(1, N_obs)

        ml_invalid = sum(
            not transition_model.is_connected(ml_arcs[i], ml_arcs[i + 1])
            for i in range(N_obs - 1)
        )
        ml_continuity = (N_obs - 1 - ml_invalid) / max(1, N_obs - 1)

        # -----------------------------------------------------------------
        # Strategy C: Random Forest + HMM + Viterbi
        # -----------------------------------------------------------------
        viterbi_res: ViterbiResult = decoder.decode_trajectory(merged, trip_id=trip_id)
        viterbi_arcs = viterbi_res.chosen_arcs

        viterbi_correct = sum(viterbi_arcs[i] == gt_map[sorted_obs_ids[i]] for i in range(N_obs))
        viterbi_top1 = viterbi_correct / max(1, N_obs)

        # Per-observation records for detailed export
        trip_records = []
        for i, oid in enumerate(sorted_obs_ids):
            a_near = nearest_arcs[i]
            a_ml = ml_arcs[i]
            a_vit = viterbi_arcs[i]
            a_gt = gt_map[oid]

            is_connected = True
            if i < N_obs - 1:
                is_connected = transition_model.is_connected(a_vit, viterbi_arcs[i + 1])

            trip_records.append({
                "trip_id": trip_id,
                "obs_id": oid,
                "nearest_arc_id": a_near,
                "ml_arc_id": a_ml,
                "viterbi_arc_id": a_vit,
                "ground_truth_arc_id": a_gt,
                "viterbi_is_correct": int(a_vit == a_gt),
                "ml_is_correct": int(a_ml == a_gt),
                "nearest_is_correct": int(a_near == a_gt),
                "viterbi_is_connected": int(is_connected),
            })

        metrics = {
            "trip_id": trip_id,
            "n_observations": N_obs,
            # Nearest
            "nearest_top1_acc": nearest_top1,
            "nearest_continuity": nearest_continuity,
            "nearest_invalid_trans": nearest_invalid,
            "nearest_time_s": nearest_time,
            # ML Only
            "ml_top1_acc": ml_top1,
            "ml_continuity": ml_continuity,
            "ml_invalid_trans": ml_invalid,
            "ml_time_s": ml_time,
            # Viterbi HMM
            "viterbi_top1_acc": viterbi_top1,
            "viterbi_continuity": viterbi_res.route_continuity,
            "viterbi_invalid_trans": viterbi_res.invalid_transition_count,
            "viterbi_fallbacks": viterbi_res.fallback_count,
            "viterbi_time_s": viterbi_res.decoding_time_s,
        }

        logger.info(
            f"Trip {trip_str}: "
            f"Nearest Top1={nearest_top1:.4f} (Cont={nearest_continuity:.4f}) | "
            f"ML Top1={ml_top1:.4f} (Cont={ml_continuity:.4f}) | "
            f"Viterbi Top1={viterbi_top1:.4f} (Cont={viterbi_res.route_continuity:.4f}, inv={viterbi_res.invalid_transition_count}) | "
            f"Time={viterbi_res.decoding_time_s:.2f}s"
        )

        return {
            "metrics": metrics,
            "records": trip_records,
            "viterbi_result": viterbi_res,
        }

    def run_all_trips(self) -> pd.DataFrame:
        """Execute route reconstruction across all student-test trips."""
        preds = self.load_predictions()
        trip_ids = sorted(preds["trip_id"].unique())
        logger.info(f"Running route reconstruction on {len(trip_ids)} student-test trips: {trip_ids}")

        all_metrics = []
        all_records = []

        for tid in trip_ids:
            df_trip = preds[preds["trip_id"] == tid]
            res = self.run_trip(int(tid), df_trip)
            all_metrics.append(res["metrics"])
            all_records.extend(res["records"])

        # 1. Summary DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        summary_path = os.path.join(self.output_dir, "reconstruction_results.csv")
        metrics_df.to_csv(summary_path, index=False)
        logger.info(f"Saved reconstruction metrics: {summary_path}")

        # 2. Detailed route records
        records_df = pd.DataFrame(all_records)
        routes_path = os.path.join(self.output_dir, "student_test_reconstructed_routes.csv")
        records_df.to_csv(routes_path, index=False)
        logger.info(f"Saved reconstructed routes: {routes_path}")

        # 3. Save config
        config = {
            "beta": self.beta,
            "p_same": self.p_same,
            "p_conn1": self.p_conn1,
            "p_conn2": self.p_conn2,
            "p_fallback": self.p_fallback,
            "n_trips": len(trip_ids),
            "total_observations": int(metrics_df["n_observations"].sum()),
            "mean_nearest_top1": float(metrics_df["nearest_top1_acc"].mean()),
            "mean_ml_top1": float(metrics_df["ml_top1_acc"].mean()),
            "mean_viterbi_top1": float(metrics_df["viterbi_top1_acc"].mean()),
            "mean_nearest_continuity": float(metrics_df["nearest_continuity"].mean()),
            "mean_ml_continuity": float(metrics_df["ml_continuity"].mean()),
            "mean_viterbi_continuity": float(metrics_df["viterbi_continuity"].mean()),
            "total_ml_invalid_trans": int(metrics_df["ml_invalid_trans"].sum()),
            "total_viterbi_invalid_trans": int(metrics_df["viterbi_invalid_trans"].sum()),
        }
        config_path = os.path.join(self.output_dir, "decoder_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # 4. Generate comparison plots
        self._plot_results(metrics_df)
        self._plot_trajectory_sample(records_df)

        return metrics_df

    def _plot_results(self, df: pd.DataFrame):
        """Generate publication-quality comparison bar plots."""
        trips = [f"Trip {int(t):02d}" for t in df["trip_id"]]
        x = np.arange(len(trips))
        width = 0.28

        # --- Plot 1: Top-1 Accuracy ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        ax1.bar(x - width, df["nearest_top1_acc"], width, label="Nearest Candidate", color="#95a5a6", alpha=0.9)
        ax1.bar(x, df["ml_top1_acc"], width, label="Random Forest ML-Only", color="#3498db", alpha=0.9)
        ax1.bar(x + width, df["viterbi_top1_acc"], width, label="RF + HMM + Viterbi", color="#2ecc71", alpha=0.95)

        ax1.set_ylabel("Observation Top-1 Accuracy", fontsize=12, fontweight="bold")
        ax1.set_title("PATHSNAP: Observation Top-1 Accuracy Comparison across Student-Testing Trips", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(trips, fontsize=11)
        ax1.set_ylim(0.0, 1.05)
        ax1.legend(loc="lower right", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

        # Annotate mean values
        mean_near = df["nearest_top1_acc"].mean()
        mean_ml = df["ml_top1_acc"].mean()
        mean_vit = df["viterbi_top1_acc"].mean()
        ax1.axhline(mean_vit, color="#27ae60", linestyle=":", linewidth=2, label=f"Viterbi Mean ({mean_vit:.1%})")
        ax1.axhline(mean_ml, color="#2980b9", linestyle=":", linewidth=2, label=f"ML Mean ({mean_ml:.1%})")

        # --- Plot 2: Route Continuity ---
        ax2.bar(x - width, df["nearest_continuity"], width, label="Nearest Candidate", color="#95a5a6", alpha=0.9)
        ax2.bar(x, df["ml_continuity"], width, label="Random Forest ML-Only", color="#e74c3c", alpha=0.9)
        ax2.bar(x + width, df["viterbi_continuity"], width, label="RF + HMM + Viterbi", color="#27ae60", alpha=0.95)

        ax2.set_ylabel("Route Continuity (% Valid Transitions)", fontsize=12, fontweight="bold")
        ax2.set_title("PATHSNAP: Route Topological Continuity Comparison (% Valid Transitions in Directed Graph)", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(trips, fontsize=11)
        ax2.set_ylim(0.7, 1.02)
        ax2.legend(loc="lower left", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "route_reconstruction_comparison.png")
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved comparison plot: {plot_path}")

    def _plot_trajectory_sample(self, records_df: pd.DataFrame):
        """Generate sequence plot illustrating discontinuity elimination."""
        # Select trip 0, observations 200 to 350
        sample = records_df[(records_df["trip_id"] == 0) & (records_df["obs_id"] >= 150) & (records_df["obs_id"] <= 260)]
        if len(sample) == 0:
            return

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(sample["obs_id"], sample["ground_truth_arc_id"], "k-", linewidth=2.5, label="Ground Truth Arc ID", alpha=0.8)
        ax.plot(sample["obs_id"], sample["viterbi_arc_id"], "g--", linewidth=2.0, label="Viterbi Reconstructed Arc ID", alpha=0.9)
        ax.scatter(sample["obs_id"], sample["ml_arc_id"], color="red", s=18, label="ML-Only Prediction (Notice Jitter)", alpha=0.6)

        ax.set_xlabel("Observation ID (Time Step)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Traversed Arc ID", fontsize=11, fontweight="bold")
        ax.set_title("PATHSNAP: Sequence Reconstruction Window (Trip 00) – Viterbi Smoothness vs ML Jitter", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "trajectory_sample_smoothing.png")
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved trajectory sequence plot: {plot_path}")


def main():
    pipeline = RouteReconstructionPipeline()
    results = pipeline.run_all_trips()
    print("\n============================================================")
    print("ROUTE RECONSTRUCTION BENCHMARK SUMMARY (10 STUDENT-TEST TRIPS)")
    print("============================================================")
    print(results[["trip_id", "nearest_top1_acc", "ml_top1_acc", "viterbi_top1_acc",
                   "nearest_continuity", "ml_continuity", "viterbi_continuity",
                   "viterbi_invalid_trans", "viterbi_time_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
