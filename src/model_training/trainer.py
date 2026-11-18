"""
trainer.py – Main training loop for PATHSNAP map-matching models.

Workflow:
  1. Load all training features (80 trips).
  2. Verify partition integrity (leakage guard).
  3. Compute class-imbalance weights.
  4. For each model: run 5-fold GroupKFold CV, record metrics, timing.
  5. Evaluate each model on 10 student-test trips (no refitting).
  6. Save model artifacts (joblib) and results CSV.
  7. Identify best model by CV F1.
  8. Run Bayesian HPO on best model.
  9. Save optimised model as models/best_model/.
 10. Generate predictions CSV for future HMM stage.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone

logger = logging.getLogger(__name__)

# Project root (resolved relative to this file's location)
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _resolve(path: str) -> str:
    """Resolve a path relative to project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def _setup_dirs():
    dirs = [
        "models/logistic_regression",
        "models/knn",
        "models/decision_tree",
        "models/random_forest",
        "models/xgboost",
        "models/best_model",
        "outputs/model_comparison",
        "outputs/confusion_matrices",
        "outputs/feature_importance",
        "outputs/optimization",
        "outputs/training_logs",
        "outputs/predictions",
    ]
    for d in dirs:
        os.makedirs(_resolve(d), exist_ok=True)


# ---------------------------------------------------------------------------
# Model save / load helpers
# ---------------------------------------------------------------------------

def save_model(model, model_name: str, config: dict, models_dir: str = "models"):
    """Save model artifact, config JSON."""
    out_dir = _resolve(os.path.join(models_dir, model_name))
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.pkl")
    joblib.dump(model, model_path)

    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Saved {model_name} → {model_path}")
    return model_path


def load_model(model_name: str, models_dir: str = "models"):
    """Load a saved model artifact."""
    model_path = _resolve(os.path.join(models_dir, model_name, "model.pkl"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Single-model training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    obs_ids_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    obs_ids_test: pd.Series,
    groups_test: pd.Series,
    n_splits: int = 5,
) -> dict:
    """
    Train one model: CV on train, evaluate on test, return metrics dict.
    """
    from .cross_validation import run_cv, summarise_cv
    from .metrics import (
        compute_candidate_metrics, compute_obs_level_top1_accuracy,
        candidate_miss_rate,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {model_name}")
    logger.info(f"{'='*60}")

    # --- Cross-validation ---
    t0 = time.time()
    cv_results = run_cv(
        model, X_train, y_train, groups_train,
        obs_ids=obs_ids_train, n_splits=n_splits
    )
    cv_time = time.time() - t0
    cv_summary = summarise_cv(cv_results)
    logger.info(f"CV F1 = {cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f} "
                f"| CV time = {cv_time:.1f}s")

    # --- Full training fit ---
    t0 = time.time()
    model_fitted = clone(model)
    model_fitted.fit(X_train, y_train)
    fit_time = time.time() - t0
    logger.info(f"Full fit time: {fit_time:.1f}s")

    # --- Student-test evaluation ---
    t0 = time.time()
    if hasattr(model_fitted, "predict_proba"):
        y_proba_test = model_fitted.predict_proba(X_test)[:, 1]
    else:
        y_proba_test = model_fitted.decision_function(X_test)

    y_pred_test = (y_proba_test >= 0.5).astype(int)
    test_cand_metrics = compute_candidate_metrics(
        y_test.values, y_pred_test, y_proba_test
    )
    eval_time = time.time() - t0

    # Observation-level Top-1
    test_pred_df = pd.DataFrame({
        "trip_id": groups_test.values,
        "obs_id": obs_ids_test.values,
        "score": y_proba_test,
        "is_ground_truth": y_test.values,
        "candidate_arc_id": np.zeros(len(y_test), dtype=int),  # placeholder
    })
    # Try to get actual arc IDs if passed through
    test_top1 = compute_obs_level_top1_accuracy(test_pred_df)
    logger.info(
        f"Test candidate F1={test_cand_metrics['f1']:.4f} | "
        f"Test Top-1 Acc={test_top1:.4f}"
    )

    result = {
        "model_name":        model_name,
        "cv_f1_mean":        cv_summary["f1_mean"],
        "cv_f1_std":         cv_summary["f1_std"],
        "cv_roc_auc_mean":   cv_summary.get("roc_auc_mean", float("nan")),
        "cv_avg_prec_mean":  cv_summary.get("avg_prec_mean", float("nan")),
        "cv_obs_top1_mean":  cv_summary.get("obs_top1_acc_mean", float("nan")),
        "test_f1":           test_cand_metrics["f1"],
        "test_precision":    test_cand_metrics["precision"],
        "test_recall":       test_cand_metrics["recall"],
        "test_roc_auc":      test_cand_metrics.get("roc_auc", float("nan")),
        "test_avg_prec":     test_cand_metrics.get("avg_precision", float("nan")),
        "test_top1_acc":     test_top1,
        "cv_time_s":         cv_time,
        "fit_time_s":        fit_time,
        "eval_time_s":       eval_time,
        "n_train_rows":      len(X_train),
        "n_test_rows":       len(X_test),
        "n_train_trips":     groups_train.nunique(),
        "n_test_trips":      groups_test.nunique(),
    }

    return result, model_fitted


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_model_comparison(results_df: pd.DataFrame, output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    models = results_df["model_name"].tolist()
    metrics = ["cv_f1_mean", "test_f1", "test_top1_acc", "test_roc_auc"]
    labels  = ["CV F1", "Test F1", "Test Top-1 Acc", "Test ROC-AUC"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = results_df[metric].fillna(0).tolist()
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("PATHSNAP – Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(output_dir, "model_comparison_bar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved model comparison plot: {out}")


def _plot_cv_boxplot(cv_all: dict, output_dir: str):
    """Box plot of per-fold F1 scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    model_names = list(cv_all.keys())
    f1_per_model = [cv_all[m]["test_f1"] for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(f1_per_model, labels=model_names, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for patch, color in zip(bp["boxes"], colors[:len(model_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("CV F1 Score (per fold)")
    ax.set_title("5-Fold GroupKFold CV F1 Distribution", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(output_dir, "cv_f1_boxplot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved CV box plot: {out}")


def _plot_confusion_matrix(model, model_name, X_test, y_test, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not GT", "GT"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"{model_name} – Confusion Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix: {out}")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training(
    feature_dir: str = "data/features",
    models_dir: str = "models",
    output_dir: str = "outputs",
    n_splits: int = 5,
    run_hpo: bool = True,
    hpo_n_trials: int = 50,
    max_train_trips: int | None = None,  # debug: limit trips
):
    """
    Full training pipeline.

    Parameters
    ----------
    feature_dir     : root directory for feature CSV files
    models_dir      : where to save model artifacts
    output_dir      : where to save reports, plots
    n_splits        : GroupKFold splits
    run_hpo         : whether to run Bayesian HPO on best model
    hpo_n_trials    : number of Optuna trials
    max_train_trips : optional limit (for debugging)
    """
    _setup_dirs()

    from .data_loader_ml import load_all_features, get_X_y_groups, FEATURE_COLS
    from .leakage_guard import (
        get_train_trips, get_student_test_trips,
        verify_no_leakage, log_partition_summary,
    )
    from .models import get_models, compute_scale_pos_weight
    from .feature_importance import save_all_importances

    feature_dir  = _resolve(feature_dir)
    models_dir   = _resolve(models_dir)

    # ------------------------------------------------------------------
    # 1. Discover and verify partitions
    # ------------------------------------------------------------------
    logger.info("Discovering partitions ...")
    raw_dir = _resolve("data/raw")
    train_trips = get_train_trips(raw_dir)
    stest_trips = get_student_test_trips(raw_dir)
    log_partition_summary(train_trips, stest_trips)

    # ------------------------------------------------------------------
    # 2. Load features
    # ------------------------------------------------------------------
    logger.info("Loading training features ...")
    train_df = load_all_features(
        "train", feature_dir,
        max_trips=max_train_trips,
        trip_ids=train_trips,
    )
    logger.info("Loading student-test features ...")
    stest_df = load_all_features("student_test", feature_dir, trip_ids=stest_trips)

    # ------------------------------------------------------------------
    # 3. Prepare X, y, groups
    # ------------------------------------------------------------------
    X_train, y_train, groups_train = get_X_y_groups(train_df)
    obs_ids_train = train_df["obs_id"]

    X_test, y_test, groups_test = get_X_y_groups(stest_df)
    obs_ids_test = stest_df["obs_id"]

    # Verify no leakage
    verify_no_leakage(
        train_trips=groups_train.unique().tolist(),
        test_trips=groups_test.unique().tolist(),
    )

    # ------------------------------------------------------------------
    # 4. Compute class weight / scale_pos_weight
    # ------------------------------------------------------------------
    spw = compute_scale_pos_weight(y_train)

    # ------------------------------------------------------------------
    # 5. Train all models
    # ------------------------------------------------------------------
    model_registry = get_models(scale_pos_weight=spw)
    all_results = []
    fitted_models = {}
    cv_raw = {}  # model_name → raw CV arrays for box plot

    for model_name, model in model_registry.items():
        try:
            result, fitted = train_and_evaluate_model(
                model_name=model_name,
                model=model,
                X_train=X_train, y_train=y_train,
                groups_train=groups_train, obs_ids_train=obs_ids_train,
                X_test=X_test, y_test=y_test,
                obs_ids_test=obs_ids_test, groups_test=groups_test,
                n_splits=n_splits,
            )
            all_results.append(result)
            fitted_models[model_name] = fitted

            # Save model
            config = {k: v for k, v in result.items()
                      if k not in ("model_name",)}
            config["features"] = FEATURE_COLS
            save_model(fitted, model_name, config, models_dir)

            # Confusion matrix
            _plot_confusion_matrix(
                fitted, model_name, X_test, y_test,
                os.path.join(_resolve(output_dir), "confusion_matrices")
            )

        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}", exc_info=True)

    if not all_results:
        raise RuntimeError("All models failed to train.")

    # ------------------------------------------------------------------
    # 6. Save results table
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_results).sort_values("cv_f1_mean", ascending=False)
    results_df.reset_index(drop=True, inplace=True)

    results_path = _resolve(os.path.join(output_dir, "training_logs", "training_results.csv"))
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nTraining results saved: {results_path}")
    logger.info(f"\n{results_df[['model_name','cv_f1_mean','test_f1','test_top1_acc']].to_string(index=False)}")

    # ------------------------------------------------------------------
    # 7. Comparison plots
    # ------------------------------------------------------------------
    _plot_model_comparison(results_df, _resolve(os.path.join(output_dir, "model_comparison")))

    # ------------------------------------------------------------------
    # 8. Feature importance for tree-based and LR models
    # ------------------------------------------------------------------
    save_all_importances(
        fitted_models, FEATURE_COLS,
        _resolve(os.path.join(output_dir, "feature_importance")),
        X_sample=X_test[:5000] if len(X_test) > 5000 else X_test,
        y_sample=y_test[:5000] if len(y_test) > 5000 else y_test,
    )

    # ------------------------------------------------------------------
    # 9. Best model identification
    # ------------------------------------------------------------------
    best_row = results_df.iloc[0]
    best_model_name = best_row["model_name"]
    best_model = fitted_models[best_model_name]
    logger.info(f"\nBest model: {best_model_name} (CV F1={best_row['cv_f1_mean']:.4f})")

    # ------------------------------------------------------------------
    # 10. Bayesian HPO on best model (if enabled and optuna available)
    # ------------------------------------------------------------------
    optimized_model = None
    if run_hpo and best_model_name in ("xgboost", "random_forest"):
        try:
            import optuna
            from .bayesian_optimization import (
                run_optuna_optimization, build_optimized_model
            )
            hpo_result = run_optuna_optimization(
                model_name=best_model_name,
                X_train=X_train, y_train=y_train,
                groups_train=groups_train,
                scale_pos_weight=spw,
                n_trials=hpo_n_trials,
                output_dir=_resolve(os.path.join(output_dir, "optimization")),
            )
            # Refit on full training set with best params
            optimized_model = build_optimized_model(
                best_model_name, hpo_result["best_params"], spw
            )
            optimized_model.fit(X_train, y_train)

            # Evaluate optimised model on student-test
            from .metrics import compute_candidate_metrics, compute_obs_level_top1_accuracy
            y_proba_opt = optimized_model.predict_proba(X_test)[:, 1]
            y_pred_opt  = (y_proba_opt >= 0.5).astype(int)
            opt_cand = compute_candidate_metrics(y_test.values, y_pred_opt, y_proba_opt)
            opt_top1_df = pd.DataFrame({
                "trip_id": groups_test.values,
                "obs_id": obs_ids_test.values,
                "score": y_proba_opt,
                "is_ground_truth": y_test.values,
                "candidate_arc_id": np.zeros(len(y_test), dtype=int),
            })
            opt_top1 = compute_obs_level_top1_accuracy(opt_top1_df)
            logger.info(
                f"Optimised {best_model_name}: "
                f"Test F1={opt_cand['f1']:.4f} | Top-1={opt_top1:.4f} | "
                f"Best CV F1={hpo_result['best_value']:.4f}"
            )

            # Save optimised model
            opt_config = {
                "model_name": f"{best_model_name}_optimized",
                "best_params": hpo_result["best_params"],
                "best_cv_f1": hpo_result["best_value"],
                "test_f1": opt_cand["f1"],
                "test_top1_acc": opt_top1,
                "features": FEATURE_COLS,
            }
            save_model(optimized_model, "best_model", opt_config, models_dir)
            # Also save under its named directory
            save_model(optimized_model, f"{best_model_name}_optimized", opt_config, models_dir)

            # Append to results
            opt_row = pd.DataFrame([{
                "model_name": f"{best_model_name}_optimized",
                "cv_f1_mean": hpo_result["best_value"],
                "test_f1": opt_cand["f1"],
                "test_precision": opt_cand["precision"],
                "test_recall": opt_cand["recall"],
                "test_roc_auc": opt_cand.get("roc_auc", float("nan")),
                "test_top1_acc": opt_top1,
            }])
            results_df = pd.concat([results_df, opt_row], ignore_index=True)
            results_df.to_csv(results_path, index=False)

        except ImportError:
            logger.warning("optuna not available – skipping Bayesian HPO.")
            # Copy best model as best_model
            save_model(best_model, "best_model", {}, models_dir)
    else:
        # Copy best model as best_model
        save_model(best_model, "best_model", {}, models_dir)

    # ------------------------------------------------------------------
    # 11. Generate prediction scores for HMM stage
    # ------------------------------------------------------------------
    logger.info("Generating prediction scores for HMM stage ...")
    final_model = optimized_model if optimized_model is not None else best_model

    from .predict import generate_predictions
    # Train predictions
    train_pred_df = generate_predictions(final_model, train_df,
                                         model_name=best_model_name)
    train_pred_path = _resolve(os.path.join(output_dir, "predictions",
                                             "train_predictions.csv"))
    train_pred_df.to_csv(train_pred_path, index=False)
    logger.info(f"Train predictions saved: {train_pred_path}")

    # Test predictions
    test_pred_df = generate_predictions(final_model, stest_df,
                                        model_name=best_model_name)
    test_pred_path = _resolve(os.path.join(output_dir, "predictions",
                                            "student_test_predictions.csv"))
    test_pred_df.to_csv(test_pred_path, index=False)
    logger.info(f"Student-test predictions saved: {test_pred_path}")

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Results: {results_path}")

    return results_df, fitted_models, final_model
