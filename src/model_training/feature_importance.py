"""
feature_importance.py – Feature importance analysis for PATHSNAP.

Supports three importance types:
1. Tree-based built-in importance (Random Forest, XGBoost, Decision Tree).
2. Logistic Regression coefficient magnitude (after scaling).
3. Permutation importance (model-agnostic, slower).

All results are saved as CSV and PNG plots.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)


def _get_feature_names_from_pipeline(model, feature_names):
    """Extract feature names accounting for Pipeline wrappers."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        return feature_names   # StandardScaler doesn't change names
    return feature_names


def extract_importances(model, feature_names: list[str]) -> pd.DataFrame | None:
    """
    Extract built-in feature importances.

    Returns a DataFrame with columns [feature, importance] sorted descending,
    or None if the model type is unsupported.
    """
    from sklearn.pipeline import Pipeline

    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps.get("clf", model[-1])

    # Tree-based: feature_importances_
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # Logistic Regression: coefficient magnitude
    if hasattr(estimator, "coef_"):
        coef = np.abs(estimator.coef_[0])
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": coef,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    logger.warning(f"Cannot extract built-in importance for {type(estimator).__name__}")
    return None


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance (model-agnostic).

    Returns a DataFrame with columns [feature, importance, std].
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import f1_score

    logger.info("Computing permutation importance ...")
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        scoring=lambda m, Xv, yv: f1_score(yv, m.predict(Xv), zero_division=0),
        random_state=random_state,
        n_jobs=1,
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    top_n: int = 19,
    title_suffix: str = "",
) -> str:
    """
    Plot horizontal bar chart of feature importances.

    Returns path to saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = importance_df.head(top_n).copy()
    df = df.sort_values("importance", ascending=True)   # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.42)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df)))
    bars = ax.barh(df["feature"], df["importance"], color=colors, edgecolor="white")

    # Add error bars if std column present
    if "std" in df.columns:
        ax.barh(df["feature"], df["importance"],
                xerr=df["std"], color="none", ecolor="gray",
                capsize=3, linewidth=0.8)

    ax.set_xlabel("Importance", fontsize=11)
    title = f"{model_name} – Feature Importances (Top {top_n}){title_suffix}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved feature importance plot: {out_path}")
    return out_path


def save_all_importances(
    models_dict: dict,
    feature_names: list[str],
    output_dir: str,
    X_sample: pd.DataFrame | None = None,
    y_sample: pd.Series | None = None,
    use_permutation: bool = False,
) -> pd.DataFrame:
    """
    Extract and save importances for all models in models_dict.

    Parameters
    ----------
    models_dict : {model_name: fitted model}
    feature_names : list of feature column names
    output_dir : directory for PNG and CSV outputs
    X_sample / y_sample : used only if use_permutation=True
    use_permutation : compute permutation importance (slow) instead of built-in

    Returns
    -------
    pd.DataFrame with all importances stacked (columns: model, feature, importance)
    """
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    for model_name, model in models_dict.items():
        logger.info(f"Extracting importance for {model_name} ...")
        if use_permutation and X_sample is not None and y_sample is not None:
            df = compute_permutation_importance(model, X_sample, y_sample, feature_names)
        else:
            df = extract_importances(model, feature_names)

        if df is None:
            logger.warning(f"No importance available for {model_name}")
            continue

        # Save CSV
        csv_path = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
        df.to_csv(csv_path, index=False)

        # Plot
        plot_feature_importance(df, model_name, output_dir)

        df["model"] = model_name
        all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(
            os.path.join(output_dir, "all_feature_importances.csv"), index=False
        )
        return combined

    return pd.DataFrame()
