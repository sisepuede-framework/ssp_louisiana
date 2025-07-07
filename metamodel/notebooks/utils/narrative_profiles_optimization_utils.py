"""
narrative_profiles_optimization_utils.py  •  v2.3
===============================================

Evolutionary optimisation helpers with:

• GA / NSGA-II + multi-seed restarts.
• Optional local L-BFGS-B polish for single-metric runs (uses SciPy if present).
• Plot helpers.

Public API
----------
optimise()             – GA / NSGA-II wrapper
plot_pareto()          – quick Pareto scatter
plot_best_features()   – horizontal bar of a policy’s top feature magnitudes
"""

from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Sequence, List

# pymoo core
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
try:
    from pymoo.algorithms.soo.genetic_algorithm import GA          # pymoo ≥0.6
except ModuleNotFoundError:                                         # pymoo 0.5
    from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# optional SciPy
try:
    from scipy.optimize import minimize as scipy_min
except ModuleNotFoundError:
    scipy_min = None   # module runs fine without SciPy – local search skipped

__all__ = ["optimise", "plot_pareto", "plot_best_features"]

# --------------------------------------------------------------------------- #
# 1. wrap a scikit-learn Pipeline as a pymoo Problem
# --------------------------------------------------------------------------- #
def _make_problem(
    pipeline,
    X_ref: pd.DataFrame,
    train_targets: Sequence[str],
    opt_targets: Sequence[str],
    goals: Sequence[str],
) -> Problem:
    lb = X_ref.min().values.astype(float)
    ub = X_ref.max().values.astype(float)
    sign = np.array([1.0 if g == "min" else -1.0 for g in goals])
    col_idx = [train_targets.index(t) for t in opt_targets]

    class _SkProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(lb), n_obj=len(opt_targets),
                             xl=lb, xu=ub, elementwise_evaluation=False)

        def _evaluate(self, X, out, **kwargs):
            preds = pipeline.predict(pd.DataFrame(X, columns=X_ref.columns))
            preds = np.asarray(preds)
            if preds.ndim == 1:
                preds = preds[:, None]
            out["F"] = preds[:, col_idx] * sign

    return _SkProblem()

# --------------------------------------------------------------------------- #
# 2. optional local L-BFGS-B refinement  (single-objective only)
# --------------------------------------------------------------------------- #
def _local_bfgs(x0: np.ndarray, prob: Problem) -> np.ndarray:
    """One L-BFGS-B step; returns x0 unchanged if SciPy absent."""
    if scipy_min is None:
        return x0

    def f(x):
        d = {}
        prob._evaluate(np.asarray([x]), d)
        return d["F"][0, 0]

    res = scipy_min(
        f, x0, method="L-BFGS-B",
        bounds=list(zip(prob.xl, prob.xu)),
        options={"maxiter": 200, "ftol": 1e-9},
    )
    return res.x if res.success else x0

# --------------------------------------------------------------------------- #
# 3. public optimise()
# --------------------------------------------------------------------------- #
def optimise(
    pipeline,
    X_ref: pd.DataFrame,
    *,
    train_targets: Sequence[str],
    targets: Sequence[str] | str,
    goals: Sequence[str] | str = "max",
    n_gen: int = 100,
    pop_size: int = 200,
    n_restarts: int = 3,
):
    """
    Evolutionary optimisation.

    Returns
    -------
    • single metric → (best_X : pd.Series, best_value : float)  
    • multi  metric → (pareto_X : pd.DataFrame, pareto_F : pd.DataFrame)
    """
    # normalise arg types
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(goals, str):
        goals = [goals]
    if len(targets) != len(goals):
        raise ValueError("targets and goals lengths differ")

    prob = _make_problem(pipeline, X_ref, train_targets, targets, goals)

    def _one_run(seed):
        algo = GA(pop_size=pop_size, eliminate_duplicates=True) if len(targets) == 1 \
               else NSGA2(pop_size=pop_size, eliminate_duplicates=True)
        return minimize(prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)

    results = [_one_run(seed) for seed in range(n_restarts)]

    # ---------------- single-objective -------------------------------------
    if len(targets) == 1:
        best_ga = min(results, key=lambda r: r.F[0])  # minimise in pymoo space
        best_x_vec = _local_bfgs(best_ga.X, prob)

        best_y = prob.evaluate(best_x_vec[None, :])[0, 0]   # ndarray slice
        if goals[0] == "max":
            best_y *= -1

        best_x = pd.Series(best_x_vec, index=X_ref.columns, name="best_policy")
        return best_x, float(best_y)

    # ---------------- multi-objective --------------------------------------
    X_all = np.vstack([r.pop.get("X") for r in results])
    F_all = np.vstack([r.pop.get("F") for r in results])

    nds = NonDominatedSorting()
    I = nds.do(F_all, only_non_dominated_front=True)

    sign = np.array([1.0 if g == "min" else -1.0 for g in goals])
    pareto_X = pd.DataFrame(X_all[I], columns=X_ref.columns)
    pareto_F = pd.DataFrame(F_all[I] * sign, columns=targets)
    return pareto_X, pareto_F

# --------------------------------------------------------------------------- #
# 4. plotting helpers
# --------------------------------------------------------------------------- #
def plot_pareto(F: pd.DataFrame | np.ndarray, *, labels: List[str] | None = None):
    """Quick 2-D / 3-D / scatter-matrix plot of objective space."""
    F = pd.DataFrame(F) if not isinstance(F, pd.DataFrame) else F.copy()
    if labels is not None:
        F.columns = labels[: F.shape[1]]

    n = F.shape[1]
    if n == 2:
        plt.scatter(F.iloc[:, 0], F.iloc[:, 1], s=20)
        plt.xlabel(F.columns[0]); plt.ylabel(F.columns[1])
        plt.title("Pareto front (2-D)")
    elif n == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        ax = plt.figure().add_subplot(111, projection="3d")
        ax.scatter(F.iloc[:, 0], F.iloc[:, 1], F.iloc[:, 2], s=20)
        ax.set_xlabel(F.columns[0]); ax.set_ylabel(F.columns[1]); ax.set_zlabel(F.columns[2])
        ax.set_title("Pareto front (3-D)")
    else:
        pd.plotting.scatter_matrix(F, figsize=(2.3 * n, 2.3 * n),
                                   diagonal="kde", s=15)
        plt.suptitle("Pareto front (scatter-matrix)", y=1.02)

    plt.tight_layout(); plt.show()


def plot_best_features(series: pd.Series, *, top_n: int = 15, title: str = ""):
    """Horizontal bar chart of the largest-magnitude features of a policy."""
    s = series.abs().nlargest(top_n).sort_values()
    colours = ["steelblue" if series[i] >= 0 else "salmon" for i in s.index]

    plt.figure(figsize=(6, 0.45 * top_n + 1))
    plt.barh(s.index, s.values, color=colours)
    plt.xlabel("Magnitude"); plt.title(title or "Top features")
    plt.tight_layout(); plt.show()
