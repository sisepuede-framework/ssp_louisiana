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

# ---------------------------------------------------------------------------
# 3. public optimise()  — general, robust for 1..K objectives
# ---------------------------------------------------------------------------
from typing import Sequence
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

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

    Notes
    -----
    - Goals are case-insensitive and may include synonyms e.g. "min", "minimize",
      "minimise", "max", "maximize".
    - For multi-objective, pareto_F values are in the ORIGINAL metric space
      (i.e., not negated for max goals).
    """
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(goals, str):
        goals = [goals]

    # --- canonicalize goals -------------------------------------------------
    _syn_min = {"min", "minimize", "minimise", "minimization", "minimisation"}
    _syn_max = {"max", "maximize", "maximise", "maximization", "maximisation"}
    goals = [g.strip().lower() for g in goals]
    goals = ["min" if g in _syn_min else "max" if g in _syn_max else g for g in goals]

    if len(targets) != len(goals):
        raise ValueError(f"targets ({len(targets)}) and goals ({len(goals)}) lengths differ")

    if any(g not in ("min", "max") for g in goals):
        bad = [g for g in goals if g not in ("min", "max")]
        raise ValueError(f"Unrecognized goals: {bad}. Use one of {{'min','max'}} (case-insensitive).")

    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")

    # construct pymoo problem in minimization space (your existing helper)
    prob = _make_problem(pipeline, X_ref, train_targets, targets, goals)

    # --- one run (single vs multi) -----------------------------------------
    def _one_run(seed: int):
        if len(targets) == 1:
            algo = GA(pop_size=pop_size, eliminate_duplicates=True)
        else:
            algo = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
        return minimize(prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)

    results = [_one_run(seed) for seed in range(n_restarts)]

    # ---------------- single-objective -------------------------------------
    if len(targets) == 1:
        # pymoo returns the best (min) value in r.F; pick the best across restarts
        def _scalar_F(res):
            f = np.asarray(res.F).reshape(-1)
            if f.size != 1:
                raise ValueError(f"Expected single-objective F with size 1, got shape {res.F.shape}")
            return float(f[0])

        best_ga = min(results, key=_scalar_F)

        # optional local refinement (keep GA if refinement fails)
        x0 = np.asarray(best_ga.X).reshape(-1)
        try:
            best_x_vec = _local_bfgs(x0, prob)  # expects (x_init, problem)
        except Exception:
            best_x_vec = x0

        # Evaluate in minimization space then convert back to original metric
        f_minspace = prob.evaluate(best_x_vec[None, :]).reshape(-1)[0]
        # If the user wanted "max", the objective was negated inside the problem.
        # Convert back to ORIGINAL metric value here.
        best_y = -f_minspace if goals[0] == "max" else f_minspace

        best_x = pd.Series(best_x_vec, index=X_ref.columns, name="best_policy")
        return best_x, float(best_y)

    # ---------------- multi-objective --------------------------------------
    # Stack all final pops, then extract the nondominated front (rank 0)
    X_all = np.vstack([np.asarray(r.pop.get("X")) for r in results])
    F_all = np.vstack([np.asarray(r.pop.get("F")) for r in results])

    # Non-dominated sort in minimization space
    nds = NonDominatedSorting()
    I = nds.do(F_all, only_non_dominated_front=True)
    X_nd = X_all[I]
    F_nd_minspace = F_all[I]

    # Convert back to ORIGINAL metric space by undoing any internal negation
    # For goals == "max", values in minimization space were negated -> re-negate
    sign = np.array([(-1.0 if g == "max" else 1.0) for g in goals], dtype=float)
    F_nd_original = F_nd_minspace * sign

    # De-duplicate across restarts (tolerant to tiny float noise)
    # Concatenate X and F for a stable duplicate check, then split back.
    XF = np.concatenate([X_nd, F_nd_original], axis=1)
    # Round to reduce floating-point jitter; adjust decimals as needed.
    XF_df = pd.DataFrame(np.round(XF, 10))
    XF_df = XF_df.drop_duplicates(ignore_index=True)
    n_x = X_ref.shape[1]
    X_unique = XF_df.iloc[:, :n_x].to_numpy()
    F_unique = XF_df.iloc[:, n_x:].to_numpy()

    pareto_X = pd.DataFrame(X_unique, columns=X_ref.columns)
    pareto_F = pd.DataFrame(F_unique, columns=list(targets))
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
