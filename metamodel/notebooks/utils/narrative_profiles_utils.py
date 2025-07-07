"""narrative_profiles_utils.py
================================================
Standalone helpers for SHAP‑based *directional* diagnostics.

Typical usage
-------------
>>> import joblib, pandas as pd
>>> from utils.narrative_profiles_utils import plot_shap_diagnostics
>>> model = joblib.load("multi_output_model.pkl")
>>> X_test = pd.read_parquet("X_test.parquet")
>>> y_test = pd.read_parquet("y_test.parquet")
>>> targets = ["CO2", "CH4", "N2O"]
>>> plot_shap_diagnostics(model, X_test, y_test, targets, log_transform=True)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

__all__ = ["plot_shap_diagnostics"]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100)

# ---------------------------------------------------------------------------
# Public: plot_shap_diagnostics
# ---------------------------------------------------------------------------

def plot_shap_diagnostics(
    pipeline: Union[str, Path, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    targets: Sequence[str],
    *,
    log_transform: bool = False,
    top_n: int = 10,
) -> None:
    """Create residual/parity, gain‑importance, directional‑consistency and
    beeswarm SHAP plots for **multi‑output** XGBoost pipelines.

    Parameters
    ----------
    pipeline : str | pathlib.Path | sklearn.pipeline.Pipeline
        A fitted scikit‑learn pipeline *or* a path to a pickled one.
    X_test, y_test : pd.DataFrame
        Hold‑out features and targets.
    targets : Sequence[str]
        Names of the target columns (order must match estimator order).
    log_transform : bool, default ``False``
        Whether the targets were log‑transformed during training.  The transform
        is automatically reversed for plotting.
    top_n : int, default ``10``
        Number of top features to show in bar charts and beeswarm plots.
    """
    # 0) Load pipeline if path given ------------------------------------------------
    if isinstance(pipeline, (str, Path)):
        pipeline = joblib.load(pipeline)  # type: ignore[assignment]
    if not isinstance(pipeline, Pipeline):
        raise TypeError("`pipeline` must be a scikit‑learn Pipeline or path to one.")

    if "model" not in pipeline.named_steps:
        raise ValueError("Pipeline must contain a 'model' step.")

    model_step = pipeline.named_steps["model"]
    if not isinstance(model_step, MultiOutputRegressor):
        raise TypeError("Currently only MultiOutputRegressor‑wrapped estimators are supported.")

    # 1) Predictions & residuals ----------------------------------------------------
    y_pred = model_step.predict(X_test)
    if log_transform:
        y_pred = np.expm1(y_pred)
    residuals = y_test.values - y_pred

    feature_names = X_test.columns.to_list()
    n_targets = len(targets)

    # 2) Residual + Parity ---------------------------------------------------------
    fig_rp, axes_rp = plt.subplots(n_targets, 2, figsize=(10, 4 * n_targets))
    if n_targets == 1:
        axes_rp = np.expand_dims(axes_rp, 0)

    for i, tgt in enumerate(targets):
        # residuals vs predicted
        axes_rp[i, 0].scatter(y_pred[:, i], residuals[:, i], alpha=0.6)
        axes_rp[i, 0].axhline(0, linestyle="--", color="k")
        axes_rp[i, 0].set_title(f"Residuals vs Predicted – {tgt}")
        axes_rp[i, 0].set_xlabel("Predicted")
        axes_rp[i, 0].set_ylabel("Residual")

        # parity
        y_true = y_test.iloc[:, i].values
        mn, mx = min(y_true.min(), y_pred[:, i].min()), max(y_true.max(), y_pred[:, i].max())
        axes_rp[i, 1].scatter(y_true, y_pred[:, i], alpha=0.6)
        axes_rp[i, 1].plot([mn, mx], [mn, mx], "k--", linewidth=1)
        axes_rp[i, 1].set_title(f"Actual vs Predicted – {tgt}")
        axes_rp[i, 1].set_xlabel("Actual")
        axes_rp[i, 1].set_ylabel("Predicted")

    plt.tight_layout()
    plt.show()

    # 3) Gain importances + Directional consistency + Beeswarm -------------
    for i, tgt in enumerate(targets):
        est = model_step.estimators_[i]
        gains = est.feature_importances_
        idx_imp = np.argsort(gains)[-top_n:][::-1]

        # SHAP explainer for this estimator
        explainer = shap.Explainer(est, X_test)
        shap_vals = explainer(X_test).values  # (n_samples, n_features)

        # Spearman ρ between feature value & SHAP contribution
        rho = {f: spearmanr(X_test[f], shap_vals[:, j]).correlation for j, f in enumerate(feature_names)}
        rho_ser = pd.Series(rho).dropna()
        mean_abs = np.abs(shap_vals).mean(axis=0)
        summary = (
            pd.DataFrame({"rho": rho_ser, "abs_mean": mean_abs})
            .assign(abs_rho=lambda d: d["rho"].abs())
            .sort_values("abs_rho", ascending=False)
        )
        top = summary.head(top_n)
        colors = ["mediumseagreen" if r > 0 else "salmon" for r in top["rho"]]
        alphas = (top["abs_mean"] / top["abs_mean"].max()).values

        fig, (ax_gain, ax_dir) = plt.subplots(1, 2, figsize=(14, 0.6 * top_n + 3))

        # gain importances ------------------------------------------------
        ax_gain.barh([feature_names[k] for k in idx_imp][::-1], gains[idx_imp][::-1])
        ax_gain.set_title(f"Top {top_n} Gain Importances – {tgt}")
        ax_gain.set_xlabel("Gain")

        # directional consistency ----------------------------------------
        for j, (name, row) in enumerate(top.iterrows()):
            ax_dir.barh(
                y=j,
                width=row["rho"],
                color=colors[j],
                alpha=alphas[j],
                edgecolor="black",
            )
        ax_dir.set_yticks(range(top_n), top.index)
        ax_dir.axvline(0, color="black", linewidth=0.8)
        ax_dir.set_title(f"Directional Consistency – {tgt}")
        ax_dir.set_xlabel("Spearman ρ  (+ ↗ KPI  |  – ↘ KPI)")
        ax_dir.invert_yaxis()

        plt.tight_layout()
        plt.show()

        # beeswarm --------------------------------------------------------
        shap.summary_plot(
            shap_vals,
            X_test,
            feature_names=feature_names,
            show=True,
            max_display=top_n,
            plot_size=(8, 5),
            title=f"SHAP BeeSwarm – {tgt}",
        )
