"""
run_training.py – CLI entry point for PATHSNAP model training.

Usage:
    python -m src.model_training.run_training [--debug] [--no-hpo] [--trials N]

Arguments:
    --debug     : Load only 5 training trips (fast test run)
    --no-hpo    : Skip Bayesian hyperparameter optimisation
    --trials N  : Number of Optuna trials (default: 50)
    --splits N  : Number of CV folds (default: 5)
"""

import os
import sys
import argparse
import logging

# Ensure project root is on sys.path regardless of how the script is invoked
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Fix matplotlib config dir issue on Windows
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())


def _configure_logging(level=logging.INFO):
    os.makedirs(os.path.join(_PROJECT_ROOT, "outputs", "training_logs"), exist_ok=True)
    log_path = os.path.join(_PROJECT_ROOT, "outputs", "training_logs", "training.log")

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    # Suppress noisy third-party loggers
    for noisy in ("matplotlib", "PIL", "optuna", "xgboost"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="PATHSNAP Model Training")
    parser.add_argument("--debug", action="store_true",
                        help="Load only 5 training trips for a fast smoke test")
    parser.add_argument("--no-hpo", dest="no_hpo", action="store_true",
                        help="Skip Bayesian hyperparameter optimisation")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna HPO trials (default: 50)")
    parser.add_argument("--splits", type=int, default=5,
                        help="Number of GroupKFold CV splits (default: 5)")
    args = parser.parse_args()

    _configure_logging(logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("run_training")

    logger.info("=" * 60)
    logger.info("PATHSNAP – Model Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"HPO enabled: {not args.no_hpo}")
    logger.info(f"HPO trials: {args.trials}")
    logger.info(f"CV splits: {args.splits}")

    from src.model_training.trainer import run_training

    results_df, fitted_models, best_model = run_training(
        feature_dir="data/features",
        models_dir="models",
        output_dir="outputs",
        n_splits=args.splits,
        run_hpo=not args.no_hpo,
        hpo_n_trials=args.trials,
        max_train_trips=5 if args.debug else None,
    )

    logger.info("\nFinal results summary:")
    cols = ["model_name", "cv_f1_mean", "cv_f1_std", "test_f1", "test_top1_acc"]
    available = [c for c in cols if c in results_df.columns]
    logger.info(f"\n{results_df[available].to_string(index=False)}")
    logger.info("\nTraining pipeline finished successfully.")


if __name__ == "__main__":
    main()
