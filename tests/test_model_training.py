"""
test_model_training.py – Unit tests for PATHSNAP model training pipeline.

Run with:
    pytest tests/test_model_training.py -v
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from src.model_training.data_loader_ml import (
    FEATURE_COLS, ID_COLS, LABEL_COL,
    load_feature_file, get_X_y_groups,
)
from src.model_training.leakage_guard import (
    EVALUATOR_TEST_TRIP_IDS, verify_no_leakage,
    get_train_trips, get_student_test_trips,
)
from src.model_training.metrics import (
    compute_candidate_metrics,
    compute_obs_level_top1_accuracy,
    candidate_miss_rate,
)
from src.model_training.predict import generate_predictions, top1_predictions


# ---------------------------------------------------------------------------
# Helpers – build synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_feature_df(
    n_obs: int = 50,
    n_candidates: int = 5,
    seed: int = 42,
    trip_id: int = 1,
    all_gt_present: bool = True,
) -> pd.DataFrame:
    """Create a synthetic candidate feature DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for obs_id in range(n_obs):
        gt_arc = rng.integers(0, n_candidates)
        for cand_rank in range(n_candidates):
            arc_id = cand_rank
            is_gt = int(arc_id == gt_arc) if all_gt_present else 0
            row = {
                "trip_id":               trip_id,
                "partition":             "train",
                "obs_id":                obs_id,
                "timestamp":             float(obs_id),
                "candidate_arc_id":      arc_id,
                "candidate_rank":        cand_rank,
                "perp_distance":         rng.uniform(0, 50),
                "norm_proj_pos":         rng.uniform(0, 1),
                "heading_diff":          rng.uniform(0, 180),
                "gps_heading":           rng.uniform(0, 360),
                "arc_bearing":           rng.uniform(0, 360),
                "speed_kmh":             rng.uniform(0, 80),
                "delta_t":               1.0,
                "delta_s":               rng.uniform(0, 30),
                "is_stationary":         0,
                "arc_length":            rng.uniform(10, 200),
                "from_node":             rng.integers(0, 100),
                "to_node":               rng.integers(0, 100),
                "from_node_out_degree":  rng.integers(1, 5),
                "to_node_in_degree":     rng.integers(1, 5),
                LABEL_COL:               is_gt,
                "log_perp_distance":     np.log1p(rng.uniform(0, 50)),
                "emission_prob_gaussian":rng.uniform(0, 1),
                "heading_diff_cos":      np.cos(np.radians(rng.uniform(0, 180))),
                "heading_prob_gaussian": rng.uniform(0, 1),
                "is_heading_aligned":    int(rng.random() > 0.5),
                "is_heading_opposed":    int(rng.random() > 0.5),
                "log_arc_length":        np.log1p(rng.uniform(10, 200)),
                "relative_dist_diff":    rng.uniform(0, 1),
                "distance_rank_ratio":   cand_rank / n_candidates,
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: feature loading
# ---------------------------------------------------------------------------

class TestFeatureLoading:
    def test_load_feature_file(self, tmp_path):
        """load_feature_file returns DataFrame with correct columns and dtypes."""
        df = _make_feature_df()
        csv_path = str(tmp_path / "00000001_candidates.csv")
        df.to_csv(csv_path, index=False)

        loaded = load_feature_file(csv_path)
        assert loaded is not None
        assert LABEL_COL in loaded.columns
        assert loaded[LABEL_COL].dtype in (int, np.int64, np.int32)
        for col in FEATURE_COLS:
            assert col in loaded.columns, f"Missing feature column: {col}"

    def test_nan_label_rows_dropped(self, tmp_path):
        """Rows with NaN is_ground_truth are dropped during loading."""
        df = _make_feature_df()
        # Inject NaN labels
        df.loc[:4, LABEL_COL] = np.nan
        csv_path = str(tmp_path / "00000002_candidates.csv")
        df.to_csv(csv_path, index=False)

        loaded = load_feature_file(csv_path)
        assert loaded[LABEL_COL].isna().sum() == 0
        assert len(loaded) == len(df) - 5

    def test_get_X_y_groups(self):
        """get_X_y_groups returns correct shapes and types."""
        df = _make_feature_df(n_obs=20, n_candidates=5)
        X, y, groups = get_X_y_groups(df)
        assert X.shape == (100, len(FEATURE_COLS))
        assert len(y) == 100
        assert set(y.unique()).issubset({0, 1})
        assert groups.name == "trip_id"

    def test_feature_columns_complete(self):
        """All 19 expected feature columns are present in FEATURE_COLS."""
        expected_features = [
            "perp_distance", "norm_proj_pos", "heading_diff", "speed_kmh",
            "delta_t", "delta_s", "is_stationary", "arc_length",
            "from_node_out_degree", "to_node_in_degree",
            "log_perp_distance", "emission_prob_gaussian", "heading_diff_cos",
            "heading_prob_gaussian", "is_heading_aligned", "is_heading_opposed",
            "log_arc_length", "relative_dist_diff", "distance_rank_ratio",
        ]
        assert set(expected_features) == set(FEATURE_COLS), (
            f"FEATURE_COLS mismatch. Missing: {set(expected_features) - set(FEATURE_COLS)}, "
            f"Extra: {set(FEATURE_COLS) - set(expected_features)}"
        )


# ---------------------------------------------------------------------------
# Test: label generation
# ---------------------------------------------------------------------------

class TestLabelGeneration:
    def test_label_distribution_reasonable(self):
        """Positive class fraction should be ~1/n_candidates."""
        n_candidates = 8
        df = _make_feature_df(n_obs=200, n_candidates=n_candidates)
        pos_rate = df[LABEL_COL].mean()
        # Expected ~1/8 = 0.125; allow generous tolerance
        assert 0.05 < pos_rate < 0.35, f"Unexpected positive rate: {pos_rate:.3f}"

    def test_at_most_one_gt_per_obs(self):
        """Each GPS observation should have at most one ground-truth candidate."""
        df = _make_feature_df(n_obs=100, n_candidates=5)
        gt_per_obs = df.groupby(["trip_id", "obs_id"])[LABEL_COL].sum()
        assert (gt_per_obs > 1).sum() == 0, (
            f"{(gt_per_obs > 1).sum()} observations have multiple GT candidates"
        )

    def test_label_is_integer(self):
        """Labels must be integer 0 or 1 after loading."""
        df = _make_feature_df()
        X, y, _ = get_X_y_groups(df)
        assert set(y.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Test: leakage guard
# ---------------------------------------------------------------------------

class TestLeakageGuard:
    def test_disjoint_partitions_pass(self):
        """verify_no_leakage passes when partitions are disjoint."""
        verify_no_leakage([1, 2, 3], [4, 5, 6], [7, 8, 9])

    def test_overlapping_partitions_fail(self):
        """verify_no_leakage raises AssertionError on overlap."""
        with pytest.raises(AssertionError, match="DATA LEAKAGE"):
            verify_no_leakage([1, 2, 3], [3, 4, 5])

    def test_evaluator_trips_blocked(self):
        """verify_no_leakage raises if evaluator trips appear in any partition."""
        eval_id = next(iter(EVALUATOR_TEST_TRIP_IDS))
        with pytest.raises(AssertionError, match="DATA LEAKAGE"):
            verify_no_leakage([eval_id, 1, 2], [4, 5])

    def test_evaluator_ids_not_in_train_or_test(self):
        """Train and student-test trip IDs from filesystem must exclude evaluator IDs."""
        raw_dir = os.path.join(_ROOT, "data", "raw")
        if not os.path.isdir(raw_dir):
            pytest.skip("data/raw not found – skipping filesystem test")

        train = set(get_train_trips(raw_dir))
        stest = set(get_student_test_trips(raw_dir))
        overlap_train = train & EVALUATOR_TEST_TRIP_IDS
        overlap_test  = stest & EVALUATOR_TEST_TRIP_IDS
        assert not overlap_train, f"Evaluator trips in train: {overlap_train}"
        assert not overlap_test,  f"Evaluator trips in student_test: {overlap_test}"

    def test_no_leakage_between_folds(self):
        """GroupKFold never shares trip_id between train and val."""
        from sklearn.model_selection import GroupKFold
        df = _make_feature_df(n_obs=20, n_candidates=5, trip_id=1)
        for tid in range(2, 10):
            extra = _make_feature_df(n_obs=20, n_candidates=5, trip_id=tid, seed=tid)
            df = pd.concat([df, extra], ignore_index=True)

        X, y, groups = get_X_y_groups(df)
        gkf = GroupKFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            train_trips = set(groups.iloc[train_idx].unique())
            val_trips   = set(groups.iloc[val_idx].unique())
            overlap = train_trips & val_trips
            assert not overlap, (
                f"Fold {fold}: trip_ids in both train and val: {overlap}"
            )


# ---------------------------------------------------------------------------
# Test: metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_compute_candidate_metrics(self):
        """compute_candidate_metrics returns expected keys."""
        rng = np.random.default_rng(1)
        y_true  = rng.integers(0, 2, 100)
        y_pred  = rng.integers(0, 2, 100)
        y_proba = rng.random(100)
        metrics = compute_candidate_metrics(y_true, y_pred, y_proba)
        for key in ("f1", "precision", "recall", "roc_auc", "avg_precision"):
            assert key in metrics, f"Missing metric: {key}"

    def test_top1_accuracy_perfect(self):
        """compute_obs_level_top1_accuracy = 1.0 when best score always matches GT."""
        df_rows = []
        for obs_id in range(20):
            for cand in range(5):
                is_gt = int(cand == 0)
                df_rows.append({
                    "trip_id": 1, "obs_id": obs_id,
                    "candidate_arc_id": cand,
                    "score": 1.0 - cand * 0.1,  # cand 0 always has highest score
                    "is_ground_truth": is_gt,
                })
        df = pd.DataFrame(df_rows)
        acc = compute_obs_level_top1_accuracy(df)
        assert abs(acc - 1.0) < 1e-6

    def test_top1_accuracy_random(self):
        """Top-1 accuracy is in [0, 1] for random scores."""
        rng = np.random.default_rng(42)
        df = _make_feature_df(n_obs=50, n_candidates=5)
        df["score"] = rng.random(len(df))
        df = df.rename(columns={LABEL_COL: "is_ground_truth"})
        acc = compute_obs_level_top1_accuracy(df)
        assert 0.0 <= acc <= 1.0

    def test_candidate_miss_rate_zero(self):
        """Miss rate is 0 when every observation has at least one GT candidate."""
        df = _make_feature_df(n_obs=30, n_candidates=5)
        df["score"] = np.random.rand(len(df))
        df = df.rename(columns={LABEL_COL: "is_ground_truth"})
        rate = candidate_miss_rate(df)
        assert rate == 0.0


# ---------------------------------------------------------------------------
# Test: model save / load
# ---------------------------------------------------------------------------

class TestModelSaveLoad:
    def test_save_and_load_model(self, tmp_path):
        """Saved model can be reloaded and produces identical predictions."""
        from sklearn.ensemble import RandomForestClassifier
        from src.model_training.trainer import save_model, load_model

        df = _make_feature_df(n_obs=50, n_candidates=5)
        X, y, _ = get_X_y_groups(df)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X.values, y.values)

        # Save
        models_dir = str(tmp_path / "models")
        save_model(rf, "test_rf", {"test": True}, models_dir)

        # Load
        loaded = load_model("test_rf", models_dir)
        pred_orig   = rf.predict(X.values)
        pred_loaded = loaded.predict(X.values)
        assert np.array_equal(pred_orig, pred_loaded), (
            "Predictions differ after save/load"
        )

    def test_config_json_saved(self, tmp_path):
        """Config JSON is written alongside model.pkl."""
        from sklearn.tree import DecisionTreeClassifier
        from src.model_training.trainer import save_model

        df = _make_feature_df(n_obs=20, n_candidates=3)
        X, y, _ = get_X_y_groups(df)

        dt = DecisionTreeClassifier(max_depth=3)
        dt.fit(X.values, y.values)

        models_dir = str(tmp_path / "models")
        save_model(dt, "test_dt", {"max_depth": 3, "note": "unit test"}, models_dir)

        config_path = os.path.join(models_dir, "test_dt", "config.json")
        assert os.path.exists(config_path)
        with open(config_path) as f:
            cfg = json.load(f)
        assert cfg["max_depth"] == 3


# ---------------------------------------------------------------------------
# Test: prediction output format
# ---------------------------------------------------------------------------

class TestPredictionFormat:
    def test_generate_predictions_format(self):
        """generate_predictions returns correct columns and ordering."""
        from sklearn.ensemble import RandomForestClassifier

        df = _make_feature_df(n_obs=20, n_candidates=5)
        X, y, _ = get_X_y_groups(df)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X.values, y.values)

        preds = generate_predictions(rf, df, model_name="test_rf")

        required = {"trip_id", "obs_id", "candidate_arc_id", "score",
                    "rank_by_score", "model"}
        assert required.issubset(set(preds.columns)), (
            f"Missing columns: {required - set(preds.columns)}"
        )
        assert len(preds) == len(df)

    def test_rank_by_score_starts_at_1(self):
        """The minimum rank_by_score for each observation is 1."""
        from sklearn.ensemble import RandomForestClassifier

        df = _make_feature_df(n_obs=20, n_candidates=5)
        X, y, _ = get_X_y_groups(df)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X.values, y.values)

        preds = generate_predictions(rf, df, model_name="test_rf")
        min_rank_per_obs = preds.groupby(["trip_id", "obs_id"])["rank_by_score"].min()
        assert (min_rank_per_obs == 1).all(), (
            "Some observations have minimum rank_by_score != 1"
        )

    def test_top1_predictions_one_per_obs(self):
        """top1_predictions returns exactly one row per (trip_id, obs_id)."""
        from sklearn.ensemble import RandomForestClassifier

        df = _make_feature_df(n_obs=25, n_candidates=5)
        X, y, _ = get_X_y_groups(df)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X.values, y.values)

        preds  = generate_predictions(rf, df, model_name="test_rf")
        top1   = top1_predictions(preds)
        n_obs  = df.groupby(["trip_id", "obs_id"]).ngroups
        assert len(top1) == n_obs, (
            f"top1_predictions returned {len(top1)} rows, expected {n_obs}"
        )


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_end_to_end_mini_training(self, tmp_path):
        """
        Smoke test: build multi-trip dataset, train Random Forest with 3-fold CV,
        verify results dict has expected keys.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.base import clone
        from sklearn.model_selection import GroupKFold
        from src.model_training.cross_validation import run_cv, summarise_cv

        # Build 6-trip dataset
        frames = []
        for tid in range(6):
            frames.append(_make_feature_df(n_obs=30, n_candidates=5, trip_id=tid, seed=tid))
        df = pd.concat(frames, ignore_index=True)
        X, y, groups = get_X_y_groups(df)

        rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42,
                                    class_weight="balanced")
        cv_results = run_cv(rf, X, y, groups, n_splits=3)
        summary = summarise_cv(cv_results)

        assert "f1_mean" in summary
        assert 0.0 <= summary["f1_mean"] <= 1.0

    def test_actual_feature_files_exist(self):
        """Verify that at least 1 actual feature file exists in data/features/train."""
        import glob
        pattern = os.path.join(_ROOT, "data", "features", "train", "*_candidates.csv")
        files = glob.glob(pattern)
        assert len(files) >= 1, (
            "No feature files found in data/features/train/. "
            "Run dataset_builder.py first."
        )
