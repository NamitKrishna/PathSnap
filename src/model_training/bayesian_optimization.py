"""
bayesian_optimization.py – Hyperparameter optimisation for PATHSNAP.

Uses Optuna (Bayesian optimisation with TPE sampler) when available,
falling back to scikit-learn RandomizedSearchCV.

The objective is 5-fold GroupKFold F1 on the 80 training trips.
Best parameters are refitted on the full training set and evaluated
on the student-test partition.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.base import clone

logger = logging.getLogger(__name__)

N_TRIALS = 50
N_SPLITS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# XGBoost search space
# ---------------------------------------------------------------------------

def _xgb_objective(trial, X, y, groups, scale_pos_weight):
    from xgboost import XGBClassifier
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 60, 150),
        "max_depth":         trial.suggest_int("max_depth", 4, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 15),
        "scale_pos_weight":  scale_pos_weight,
        "random_state":      RANDOM_STATE,
        "eval_metric":       "logloss",
        "tree_method":       "hist",
        "verbosity":         0,
        "n_jobs":            -1,
    }
    model = XGBClassifier(**params)
    gkf = GroupKFold(n_splits=3)
    scores = cross_val_score(
        model, X, y, cv=gkf, groups=groups,
        scoring="f1", n_jobs=1, error_score=0.0
    )
    score = scores.mean()
    logger.info(f"  [Optuna Trial {trial.number}] XGBoost F1 = {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Random Forest search space
# ---------------------------------------------------------------------------

def _rf_objective(trial, X, y, groups):
    from sklearn.ensemble import RandomForestClassifier
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 50, 100),
        "max_depth":       trial.suggest_int("max_depth", 8, 14),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 20, 50),
        "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "max_samples":     trial.suggest_float("max_samples", 0.15, 0.30),
        "class_weight":    "balanced",
        "random_state":    RANDOM_STATE,
        "n_jobs":          -1,
    }
    model = RandomForestClassifier(**params)
    gkf = GroupKFold(n_splits=3)
    scores = cross_val_score(
        model, X, y, cv=gkf, groups=groups,
        scoring="f1", n_jobs=1, error_score=0.0
    )
    score = scores.mean()
    logger.info(f"  [Optuna Trial {trial.number}] Random Forest F1 = {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Main Optuna optimisation
# ---------------------------------------------------------------------------

def run_optuna_optimization(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    scale_pos_weight: float = 11.0,
    n_trials: int = N_TRIALS,
    output_dir: str = "outputs/optimization",
) -> dict:
    """
    Run Optuna Bayesian optimisation for the specified model.

    Parameters
    ----------
    model_name       : 'xgboost' or 'random_forest'
    X_train, y_train : training features and labels
    groups_train     : trip_id groups for GroupKFold
    scale_pos_weight : for XGBoost
    n_trials         : number of Optuna trials
    output_dir       : directory for saving trial CSV and best params JSON

    Returns
    -------
    dict with keys: best_params, best_value, trials_df
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    os.makedirs(output_dir, exist_ok=True)

    if model_name == "xgboost":
        objective = lambda trial: _xgb_objective(
            trial, X_train.values, y_train.values, groups_train.values, scale_pos_weight
        )
    elif model_name == "random_forest":
        objective = lambda trial: _rf_objective(
            trial, X_train.values, y_train.values, groups_train.values
        )
    else:
        raise ValueError(f"HPO not configured for model: {model_name}")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    logger.info(f"Starting Optuna HPO for {model_name}: {n_trials} trials ...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    best_value  = study.best_trial.value
    logger.info(f"Best CV F1 for {model_name}: {best_value:.4f} | params: {best_params}")

    # Save best params
    params_path = os.path.join(output_dir, f"{model_name}_best_params.json")
    with open(params_path, "w") as f:
        json.dump({"model": model_name, "best_cv_f1": best_value,
                   "params": best_params}, f, indent=2)

    # Save all trials
    trials_df = study.trials_dataframe()
    trials_df["model"] = model_name
    trials_path = os.path.join(output_dir, f"{model_name}_optuna_trials.csv")
    trials_df.to_csv(trials_path, index=False)

    # Plot optimization history
    _plot_optuna_history(study, model_name, output_dir)

    return {
        "best_params": best_params,
        "best_value":  best_value,
        "trials_df":   trials_df,
        "study":       study,
    }


def _plot_optuna_history(study, model_name: str, output_dir: str):
    """Save optimization history plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    best_so_far = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(values, "o", alpha=0.4, markersize=4, label="Trial F1", color="steelblue")
    ax.plot(best_so_far, "-", linewidth=2, label="Best so far", color="darkorange")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV F1 Score")
    ax.set_title(f"Optuna Optimization History – {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{model_name}_optuna_history.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Optuna history plot: {out_path}")


# ---------------------------------------------------------------------------
# Fallback: RandomizedSearchCV
# ---------------------------------------------------------------------------

def run_randomized_search(
    model,
    param_distributions: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_iter: int = 30,
    output_dir: str = "outputs/optimization",
    model_name: str = "model",
) -> dict:
    """
    Fallback HPO using RandomizedSearchCV with GroupKFold.

    Returns dict with best_params, best_score, results_df.
    """
    from sklearn.model_selection import RandomizedSearchCV

    os.makedirs(output_dir, exist_ok=True)
    gkf = GroupKFold(n_splits=N_SPLITS)

    search = RandomizedSearchCV(
        model, param_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=gkf,
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
        error_score=0.0,
    )
    logger.info(f"Running RandomizedSearchCV ({n_iter} iterations) for {model_name} ...")
    search.fit(X_train, y_train, groups=groups_train)

    best_params = search.best_params_
    best_score  = search.best_score_
    logger.info(f"RandomizedSearchCV best F1 for {model_name}: {best_score:.4f}")

    results_df = pd.DataFrame(search.cv_results_)
    results_df.to_csv(
        os.path.join(output_dir, f"{model_name}_randomsearch_results.csv"), index=False
    )

    params_path = os.path.join(output_dir, f"{model_name}_best_params.json")
    with open(params_path, "w") as f:
        json.dump({"model": model_name, "best_cv_f1": best_score,
                   "params": best_params}, f, indent=2)

    return {
        "best_params": best_params,
        "best_score":  best_score,
        "results_df":  results_df,
        "best_model":  search.best_estimator_,
    }


def build_optimized_model(model_name: str, best_params: dict, scale_pos_weight: float = 11.0):
    """Instantiate a model with best hyperparameters."""
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        params = {**best_params,
                  "scale_pos_weight": scale_pos_weight,
                  "random_state": RANDOM_STATE,
                  "eval_metric": "logloss",
                  "tree_method": "hist",
                  "n_jobs": -1,
                  "verbosity": 0}
        return XGBClassifier(**params)

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        params = {**best_params,
                  "class_weight": "balanced",
                  "n_jobs": -1,
                  "random_state": RANDOM_STATE}
        return RandomForestClassifier(**params)

    raise ValueError(f"Unknown model for HPO: {model_name}")
