"""
File: utils/narrative_profiles_optimization_utils.py
Version: 3.1

Evolutionary optimisation helpers for sklearn Pipelines (multi-output) using pymoo.

Highlights
----------
• GA / NSGA-II with multi-seed restarts and Pareto de-duplication.
• Optional local L-BFGS-B polish for single-objective runs (SciPy optional).
• Fixed inputs: lock any subset of features (e.g., exogenous `group_*`).
• Multi-future evaluation in one run:
   - scenario_mode="aggregate": evaluate each candidate across many futures and aggregate
     objectives in *minimization* space via weighted mean or worst.
   - scenario_mode="stack": treat each (future × metric) as a separate objective.
• Bounds:
   - bounds="data" (default): per-feature [min,max] from X_ref.
   - bounds="unit": search in [0,1]^n and auto-unscale to data space before predict.
• **NEW (v3.1)**: Batched scenario prediction for huge speed-ups (one big predict call
  instead of S calls), optional chunking for memory safety, and a helper to ensure
  multi-threaded XGBoost predict.

Backwards compatibility
-----------------------
- v3.1 preserves the v2.3/v2.4/v3.0 API by default: if you do nothing, behaviour is unchanged.
  The new features are opt-in (fixed_inputs, scenario_* controls, threading helper).

Public API
----------
optimise()                    – GA / NSGA-II wrapper (single/multi objective)
plot_pareto()                 – quick Pareto scatter
plot_best_features()          – horizontal bar of a policy’s top feature magnitudes
build_scenarios_by_future()   – helper to construct scenario clamps from a DataFrame
fixed_inputs_from_row()       – helper to clamp features by name from a row
set_estimator_threads()       – ensure multi-threaded predict for XGBoost inside a Pipeline
"""

from __future__ import annotations
from typing import Any, List, Mapping, Optional, Sequence, Dict, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

# pymoo core
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting as _NDS
try:
    # pymoo ≥ 0.6
    from pymoo.algorithms.soo.genetic_algorithm import GA
except ModuleNotFoundError:  # pymoo 0.5
    from pymoo.algorithms.soo.nonconvex.ga import GA  # type: ignore
from pymoo.algorithms.moo.nsga2 import NSGA2

# optional SciPy for single-objective polish
try:
    from scipy.optimize import minimize as scipy_min  # type: ignore
except Exception:  # pragma: no cover
    scipy_min = None

# optional XGBoost presence check
try:
    import xgboost as _xgb  # type: ignore
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

__all__ = [
    "optimise",
    "plot_pareto",
    "plot_best_features",
    "build_scenarios_by_future",
    "fixed_inputs_from_row",
    "set_estimator_threads",
]

__version__ = "3.1"


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _canonical_goals(goals: Sequence[str]) -> List[str]:
    """Normalize goal strings to 'min' or 'max'. Accepts common synonyms."""
    _syn_min = {"min", "minimize", "minimise", "minimization", "minimisation"}
    _syn_max = {"max", "maximize", "maximise", "maximization", "maximisation"}
    out: List[str] = []
    for g in goals:
        gs = str(g).strip().lower()
        if gs in _syn_min:
            out.append("min")
        elif gs in _syn_max:
            out.append("max")
        else:
            out.append(gs)
    return out


def fixed_inputs_from_row(
    row: pd.Series,
    feature_cols: Sequence[str],
    *,
    prefix: str = "group_",
) -> Dict[str, float]:
    """Build a feature→value dict by **name** from a row, clamping columns that
    start with `prefix`. Useful for freezing exogenous `group_*` variables.
    """
    return {c: float(row[c]) for c in feature_cols if c.startswith(prefix)}


def build_scenarios_by_future(
    full_df: pd.DataFrame,
    feature_cols: Sequence[str],
    future_ids: Sequence[Any],
    *,
    prefix: str = "group_",
) -> List[Dict[str, float]]:
    """Construct a list of scenario dicts (feature→value clamps) for each `future_id`.

    Parameters
    ----------
    full_df : DataFrame containing at least 'future_id' and `feature_cols`.
    feature_cols : columns that the model expects (order must match training).
    future_ids : list of future IDs to include.
    prefix : only features that start with this prefix will be clamped.

    Returns
    -------
    list of dict[str,float] with one dict per future.
    """
    scenarios: List[Dict[str, float]] = []
    for fid in future_ids:
        row = full_df.loc[full_df["future_id"] == fid, feature_cols].iloc[0]
        scenarios.append(fixed_inputs_from_row(row, feature_cols, prefix=prefix))
    return scenarios


# --------------------------------------------------------------------------- #
# Optional: ensure multi-threaded predict for XGBoost inside a Pipeline
# --------------------------------------------------------------------------- #
def _iter_estimators(obj: Any) -> Iterable[Any]:
    """Walk through a sklearn Pipeline/ColumnTransformer-ish object to yield estimators."""
    if obj is None:
        return
    yield obj
    # sklearn Pipeline
    steps = getattr(obj, "steps", None)
    if steps:
        for _, step in steps:
            yield from _iter_estimators(step)
    # ColumnTransformer / FeatureUnion style
    transformers = getattr(obj, "transformers", None) or getattr(obj, "estimators", None)
    if transformers:
        for _, tr, _ in transformers:
            yield from _iter_estimators(tr)
    # GridSearchCV / RandomizedSearchCV / meta-estimators
    est = getattr(obj, "estimator", None) or getattr(obj, "best_estimator_", None)
    if est is not None:
        yield from _iter_estimators(est)


def set_estimator_threads(pipeline, n_jobs: Optional[int] = None) -> None:
    """Ensure multi-threaded predict for XGBoost estimators inside a sklearn Pipeline.

    - If XGBoost is present and any step is an XGBRegressor/XGBClassifier, set its `n_jobs`.
    - `n_jobs` defaults to all logical CPUs.
    """
    n_jobs = n_jobs or mp.cpu_count()
    if not _HAVE_XGB:
        return
    for est in _iter_estimators(pipeline):
        try:
            if isinstance(est, (_xgb.XGBRegressor, _xgb.XGBClassifier)):  # type: ignore[attr-defined]
                est.set_params(n_jobs=n_jobs)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 1) Wrap a scikit-learn Pipeline as a pymoo Problem (vectorized evaluate)
#     - **v3.1**: Batched scenario prediction (1 big predict), optional chunking.
# --------------------------------------------------------------------------- #
def _make_problem(
    pipeline,
    X_ref: pd.DataFrame,
    train_targets: Sequence[str],
    opt_targets: Sequence[str],
    goals: Sequence[str],
    *,
    fixed_inputs: Optional[Mapping[str, float]] = None,           # global clamps
    bounds: str = "data",
    scenario_sets: Optional[Sequence[Mapping[str, float]]] = None, # per-future clamps
    scenario_weights: Optional[Sequence[float]] = None,            # for aggregate mode
    scenario_mode: str = "aggregate",                              # "aggregate" | "stack"
    scenario_agg: str = "mean",                                    # "mean" | "worst"
    scenario_chunk: Optional[int] = None,                          # process scenarios in chunks
) -> Problem:
    """Create a vectorized pymoo Problem with optional **multi-future** evaluation.

    Behaviour
    ---------
    - Decision variables = all columns of `X_ref`, **except** those clamped via
      `fixed_inputs` (global) and/or per-scenario dicts in `scenario_sets`.
    - Objectives are derived from `opt_targets` with `goals` converted to
      minimization space internally.

    Multi-future modes
    -------------------
    • aggregate: For each candidate, evaluate under every scenario and aggregate
      per-objective in **minimization** space using a weighted **mean** (default)
      or **worst** (elementwise max). The number of objectives equals
      `len(opt_targets)`.

    • stack: Concatenate objectives across scenarios → number of objectives is
      `len(opt_targets) * n_scenarios` in minimization space. Useful to expose
      trade-offs across futures explicitly.

    Performance (v3.1)
    ------------------
    - Batched scenario prediction: build a single matrix of size (S·N, d) and call
      `pipeline.predict(...)` once per generation. Optional `scenario_chunk` controls
      memory by splitting S into chunks per call.
    """
    if not isinstance(X_ref, pd.DataFrame):
        raise TypeError("X_ref must be a pandas DataFrame")

    cols = X_ref.columns.to_list()

    # data-space per-feature bounds
    lb_data = X_ref.min().values.astype(float)
    ub_data = X_ref.max().values.astype(float)
    span = (ub_data - lb_data)
    span[span == 0.0] = 1.0  # guard against zero-span columns

    use_unit = (str(bounds).lower() == "unit")
    xl, xu = (np.zeros_like(lb_data), np.ones_like(lb_data)) if use_unit else (lb_data, ub_data)

    # minimization sign:  1 for min,  -1 for max
    goals = _canonical_goals(goals)
    sign = np.array([1.0 if g == "min" else -1.0 for g in goals], dtype=float)
    col_idx = [train_targets.index(t) for t in opt_targets]

    # Global fixed inputs
    fixed_inputs = dict(fixed_inputs or {})
    fixed_mask_global = np.array([c in fixed_inputs for c in cols])
    fixed_vals_global = np.array([fixed_inputs.get(c, 0.0) for c in cols], dtype=float)

    # Scenario sets (list[dict feature->value])
    scen_list = list(scenario_sets) if scenario_sets is not None else [None]
    n_scen = len(scen_list)

    scen_masks: List[Optional[np.ndarray]] = []
    scen_vals:  List[Optional[np.ndarray]] = []
    for s in scen_list:
        if s is None:
            scen_masks.append(None); scen_vals.append(None)
        else:
            sd = dict(s)
            m = np.array([c in sd for c in cols])
            v = np.array([sd.get(c, 0.0) for c in cols], dtype=float)
            scen_masks.append(m); scen_vals.append(v)

    # weights for aggregate
    if scenario_mode == "aggregate":
        if scenario_weights is None:
            w = np.ones(n_scen, float) / max(n_scen, 1)
        else:
            w = np.asarray(scenario_weights, float)
            if w.shape[0] != n_scen:
                raise ValueError("scenario_weights length must match number of scenario_sets")
            w = w / (w.sum() if w.sum() != 0 else 1.0)
    else:
        w = None

    # objectives count
    if scenario_mode == "stack":
        n_obj = len(opt_targets) * n_scen
    else:
        n_obj = len(opt_targets)

    # chunking control
    chunk = None if (scenario_chunk is None or scenario_chunk <= 0) else int(scenario_chunk)

    class _SkProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=len(cols),
                n_obj=n_obj,
                xl=xl,
                xu=xu,
                elementwise_evaluation=False,
            )

        def _evaluate(self, X, out, **kwargs):
            X = np.asarray(X, float)                # (N, d)
            N = X.shape[0]

            # unit-space → data-space
            Xd = (X * span + lb_data) if use_unit else X.copy()

            # apply global fixed inputs
            if fixed_inputs:
                Xd[:, fixed_mask_global] = fixed_vals_global[fixed_mask_global]

            # helper to evaluate a subset of scenarios in one batch
            def eval_scenarios_batch(s_idx: np.ndarray):
                # Build big batch by stacking (len(s_idx) * N, d)
                mats = []
                for k in s_idx:
                    m, v = scen_masks[k], scen_vals[k]
                    if m is None:
                        Xm = Xd
                    else:
                        Xm = Xd.copy()
                        Xm[:, m] = v[m]
                    mats.append(Xm)
                X_big = np.concatenate(mats, axis=0)

                # Single predict over all scenarios in this chunk
                preds_big = pipeline.predict(pd.DataFrame(X_big, columns=cols))
                preds_big = np.asarray(preds_big)
                if preds_big.ndim == 1:
                    preds_big = preds_big[:, None]  # (S*N, T)

                # Reshape to (N, T, S_chunk), then select objectives, flip sign → (N, n_obj, S_chunk)
                S_chunk = len(s_idx)
                preds_big = preds_big.reshape(S_chunk, N, -1).transpose(1, 2, 0)  # (N, T, S_chunk)
                F_chunk_min = preds_big[:, col_idx, :] * sign[None, :, None]
                return F_chunk_min  # (N, n_obj, S_chunk)

            # Evaluate all scenarios with optional chunking
            if n_scen == 1:
                F_all = eval_scenarios_batch(np.array([0], dtype=int))  # (N, n_obj, 1)
            else:
                if chunk is None:
                    idx_all = np.arange(n_scen, dtype=int)
                    F_all = eval_scenarios_batch(idx_all)               # (N, n_obj, S)
                else:
                    parts = []
                    for start in range(0, n_scen, chunk):
                        idx = np.arange(start, min(start + chunk, n_scen), dtype=int)
                        parts.append(eval_scenarios_batch(idx))
                    F_all = np.concatenate(parts, axis=2)               # (N, n_obj, S)

            # Aggregate or stack
            if scenario_mode == "aggregate":
                if scenario_agg == "mean":
                    # weighted mean across scenarios → (N, n_obj)
                    out["F"] = np.tensordot(F_all, w, axes=([2], [0]))
                elif scenario_agg == "worst":
                    # worst (elementwise max) in minimization space → (N, n_obj)
                    out["F"] = F_all.max(axis=2)
                else:
                    raise ValueError("scenario_agg must be 'mean' or 'worst'")
            else:  # "stack"
                out["F"] = F_all.reshape(N, -1)  # (N, n_obj * n_scen)

    return _SkProblem()


# --------------------------------------------------------------------------- #
# 2) optional local L-BFGS-B refinement (single-objective only)
# --------------------------------------------------------------------------- #
def _local_bfgs(x0: np.ndarray, prob: Problem) -> np.ndarray:
    """One L-BFGS-B pass; returns x0 unchanged if SciPy absent or fails."""
    if scipy_min is None:
        return x0

    def f(x):
        d: Dict[str, Any] = {}
        prob._evaluate(np.asarray([x]), d)  # minimization space
        return float(np.asarray(d["F"]).reshape(-1)[0])

    try:
        res = scipy_min(
            f,
            x0,
            method="L-BFGS-B",
            bounds=list(zip(prob.xl, prob.xu)),
            options={"maxiter": 200, "ftol": 1e-9},
        )
        return res.x if (res is not None and getattr(res, "success", False)) else x0
    except Exception:
        return x0


# --------------------------------------------------------------------------- #
# 3) public optimise() — robust for 1..K objectives; multi-future capable
#     - **v3.1**: threading helper hook + scenario_chunk passthrough
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
    fixed_inputs: Optional[Mapping[str, float]] = None,
    bounds: str = "data",
    # multi-future knobs (all optional)
    scenario_sets: Optional[Sequence[Mapping[str, float]]] = None,
    scenario_weights: Optional[Sequence[float]] = None,
    scenario_mode: str = "aggregate",       # "aggregate" | "stack"
    scenario_agg: str = "mean",             # used only for aggregate
    scenario_names: Optional[Sequence[str]] = None,  # naming for stack outputs
    scenario_chunk: Optional[int] = None,   # NEW: evaluate scenarios in chunks for memory
    ensure_threaded_predict: bool = True,   # NEW: set XGBoost n_jobs if available
):
    """Evolutionary optimisation (GA/NSGA-II) for sklearn Pipelines.

    Parameters
    ----------
    pipeline : sklearn Pipeline with `.predict(X) -> (n_samples, n_targets)`.
    X_ref : DataFrame of reference features; defines column order and data-bounds.
    train_targets : list of all metric names predicted by the pipeline (in order).
    targets : metric name or list of names to optimize (subset of train_targets).
    goals : "min"/"max" (synonyms OK) per target.
    n_gen, pop_size, n_restarts : genetic algorithm controls and robust restarts.

    fixed_inputs : dict[feature_name, value]
        Globally clamp these features for *all* evaluations.

    bounds : {"data","unit"}
        - "data" (default): bounds from X_ref min/max per feature.
        - "unit": search in [0,1]^n and auto-unscale to data space before predict.

    Multi-future evaluation (optional)
    ----------------------------------
    scenario_sets : list of dicts (feature→value) used as **per-scenario clamps**
        (e.g., exogenous `group_*`). Each candidate is evaluated once per scenario.
    scenario_weights : weights for aggregate mean across scenarios.
    scenario_mode : "aggregate" (default) or "stack".
    scenario_agg : "mean" (weighted) or "worst" (elementwise max in min-space).
    scenario_names : optional labels used to name stacked objectives (order must
        match scenario_sets).
    scenario_chunk : if set (e.g., 32), scenarios are processed in batches of that size
        to limit memory usage during the single batched predict.
    ensure_threaded_predict : if True, set XGBoost estimators' n_jobs to CPU count.

    Returns
    -------
    • Single objective → `(best_X : pd.Series, best_value : float)`
    • Multi objective → `(pareto_X : pd.DataFrame, pareto_F : pd.DataFrame)`
      - In aggregate mode: columns(pareto_F) = `list(targets)` (original metric space).
      - In stack mode: columns(pareto_F) = flattened labels per (scenario × target).

    Notes
    -----
    - Objectives are converted to minimization internally; returned values are in
      the **original** metric space (sign un-flipped).
    - Decision variables exclude any clamped features (global or per-scenario).
    """
    # threading (optional but recommended)
    if ensure_threaded_predict:
        try:
            set_estimator_threads(pipeline)
        except Exception:
            pass

    # normalize inputs
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(goals, str):
        goals = [goals]

    goals = _canonical_goals(goals)

    if len(targets) != len(goals):
        raise ValueError(f"targets ({len(targets)}) and goals ({len(goals)}) lengths differ")
    if any(g not in ("min", "max") for g in goals):
        bad = [g for g in goals if g not in ("min", "max")]
        raise ValueError(f"Unrecognized goals: {bad}. Use one of {{'min','max'}}.")
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")

    # build problem
    prob = _make_problem(
        pipeline,
        X_ref,
        train_targets,
        opt_targets=list(targets),
        goals=list(goals),
        fixed_inputs=fixed_inputs,
        bounds=bounds,
        scenario_sets=scenario_sets,
        scenario_weights=scenario_weights,
        scenario_mode=scenario_mode,
        scenario_agg=scenario_agg,
        scenario_chunk=scenario_chunk,   # NEW
    )

    # choose algorithm per dimensionality and run restarts
    def _one_run(seed: int):
        algo = GA(pop_size=pop_size, eliminate_duplicates=True) if len(targets) == 1 \
               else NSGA2(pop_size=pop_size, eliminate_duplicates=True)
        return minimize(prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)

    results = [_one_run(seed) for seed in range(n_restarts)]

    # ---------------- single-objective --------------------------------------
    if len(targets) == 1:
        # pick best by minimized F
        def _scalar_F(res):
            f = np.asarray(res.F).reshape(-1)
            if f.size != 1:
                raise ValueError(f"Expected single-objective F with size 1, got shape {res.F.shape}")
            return float(f[0])

        best_ga = min(results, key=_scalar_F)

        # optional local refinement
        x0 = np.asarray(best_ga.X).reshape(-1)
        try:
            best_x_vec = _local_bfgs(x0, prob)
        except Exception:
            best_x_vec = x0

        # minimization-space objective value → original metric value
        f_minspace = np.asarray(prob.evaluate(best_x_vec[None, :])).reshape(-1)[0]
        best_y = -f_minspace if goals[0] == "max" else f_minspace

        best_x = pd.Series(best_x_vec, index=X_ref.columns, name="best_policy")
        return best_x, float(best_y)

    # ---------------- multi-objective ---------------------------------------
    # Stack final populations across restarts
    X_all = np.vstack([np.asarray(r.pop.get("X")) for r in results])
    F_all = np.vstack([np.asarray(r.pop.get("F")) for r in results])  # minimization space

    # Non-dominated sort in minimization space
    nds = _NDS()
    I = nds.do(F_all, only_non_dominated_front=True)
    X_nd = X_all[I]
    F_nd_min = F_all[I]

    # convert back to ORIGINAL metric space (undo minimization sign)
    sign = np.array([1.0 if g == "min" else -1.0 for g in goals], dtype=float)

    # In stack mode, F is (n_pop, len(targets)*n_scen) in min-space; unflip per target block
    if scenario_sets is not None and str(scenario_mode).lower() == "stack":
        n_scen = len(scenario_sets)
        n_t = len(targets)
        F_sh = F_nd_min.reshape(F_nd_min.shape[0], n_t, n_scen)
        F_un = F_sh * sign[None, :, None]
        F_nd_orig = F_un.reshape(F_nd_min.shape[0], n_t * n_scen)
    else:
        F_nd_orig = F_nd_min * sign

    # de-duplicate across restarts (tolerant to tiny float noise)
    XF = np.concatenate([X_nd, F_nd_orig], axis=1)
    XF_df = pd.DataFrame(np.round(XF, 10))
    XF_df = XF_df.drop_duplicates(ignore_index=True)

    n_x = X_ref.shape[1]
    X_unique = XF_df.iloc[:, :n_x].to_numpy()
    F_unique = XF_df.iloc[:, n_x:].to_numpy()

    pareto_X = pd.DataFrame(X_unique, columns=X_ref.columns)

    # Name objective columns
    if scenario_sets is not None and str(scenario_mode).lower() == "stack":
        n_scen = len(scenario_sets)
        n_t = len(targets)
        if scenario_names is None:
            scenario_names = [f"scen{i}" for i in range(n_scen)]
        if len(scenario_names) != n_scen:
            raise ValueError("scenario_names length must match scenario_sets")
        cols = [f"{targets[j]}@{scenario_names[k]}" for k in range(n_scen) for j in range(n_t)]
    else:
        cols = list(targets)

    pareto_F = pd.DataFrame(F_unique, columns=cols)
    return pareto_X, pareto_F


# --------------------------------------------------------------------------- #
# 4) plotting helpers
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
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = plt.figure().add_subplot(111, projection="3d")
        ax.scatter(F.iloc[:, 0], F.iloc[:, 1], F.iloc[:, 2], s=20)
        ax.set_xlabel(F.columns[0]); ax.set_ylabel(F.columns[1]); ax.set_zlabel(F.columns[2])
        ax.set_title("Pareto front (3-D)")
    else:
        pd.plotting.scatter_matrix(F, figsize=(2.3 * n, 2.3 * n), diagonal="kde", s=15)
        plt.suptitle("Pareto front (scatter-matrix)", y=1.02)

    plt.tight_layout(); plt.show()


def plot_best_features(series: pd.Series, *, top_n: int = 15, title: str = ""):
    """Horizontal bar chart of the largest-magnitude features of a policy."""
    s = series.abs().nlargest(top_n).sort_values()
    colours = ["steelblue" if float(series[i]) >= 0 else "salmon" for i in s.index]

    plt.figure(figsize=(6, 0.45 * top_n + 1))
    plt.barh(s.index, s.values, color=colours)
    plt.xlabel("Magnitude"); plt.title(title or "Top features")
    plt.tight_layout(); plt.show()




# """
# File: utils/narrative_profiles_optimization_utils.py
# Version: 3.0

# Evolutionary optimisation helpers for sklearn Pipelines (multi-output) using pymoo.

# Highlights
# ----------
# • GA / NSGA-II with multi-seed restarts and Pareto de-duplication.
# • Optional local L-BFGS-B polish for single-objective runs (SciPy optional).
# • Fixed inputs: lock any subset of features (e.g., exogenous `group_*`).
# • Multi-future evaluation in one run:
#    - scenario_mode="aggregate": evaluate each candidate across many futures and aggregate
#      objectives in *minimization* space via weighted mean or worst.
#    - scenario_mode="stack": treat each (future × metric) as a separate objective.
# • Bounds:
#    - bounds="data" (default): per-feature [min,max] from X_ref.
#    - bounds="unit": search in [0,1]^n and auto-unscale to data space before predict.
# • Plot helpers for diagnostics.

# Backwards compatibility
# -----------------------
# - v3.0 preserves the v2.3/v2.4 API by default: if you do nothing, behaviour is unchanged.
#   The new features are opt-in (fixed_inputs, scenario_*).

# Public API
# ----------
# optimise()                    – GA / NSGA-II wrapper (single/multi objective)
# plot_pareto()                 – quick Pareto scatter
# plot_best_features()          – horizontal bar of a policy’s top feature magnitudes
# build_scenarios_by_future()   – helper to construct scenario clamps from a DataFrame
# fixed_inputs_from_row()       – helper to clamp features by name from a row
# """

# from __future__ import annotations
# from typing import Any, List, Mapping, Optional, Sequence, Dict

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # pymoo core
# from pymoo.core.problem import Problem
# from pymoo.optimize import minimize
# from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting as _NDS
# try:
#     # pymoo ≥ 0.6
#     from pymoo.algorithms.soo.genetic_algorithm import GA
# except ModuleNotFoundError:  # pymoo 0.5
#     from pymoo.algorithms.soo.nonconvex.ga import GA  # type: ignore
# from pymoo.algorithms.moo.nsga2 import NSGA2

# # optional SciPy for single-objective polish
# try:
#     from scipy.optimize import minimize as scipy_min  # type: ignore
# except Exception:  # pragma: no cover
#     scipy_min = None

# __all__ = [
#     "optimise",
#     "plot_pareto",
#     "plot_best_features",
#     "build_scenarios_by_future",
#     "fixed_inputs_from_row",
# ]

# __version__ = "3.0"

# # --------------------------------------------------------------------------- #
# # Utilities
# # --------------------------------------------------------------------------- #

# def _canonical_goals(goals: Sequence[str]) -> List[str]:
#     """Normalize goal strings to 'min' or 'max'. Accepts common synonyms."""
#     _syn_min = {"min", "minimize", "minimise", "minimization", "minimisation"}
#     _syn_max = {"max", "maximize", "maximise", "maximization", "maximisation"}
#     out: List[str] = []
#     for g in goals:
#         gs = str(g).strip().lower()
#         if gs in _syn_min:
#             out.append("min")
#         elif gs in _syn_max:
#             out.append("max")
#         else:
#             out.append(gs)
#     return out


# def fixed_inputs_from_row(
#     row: pd.Series,
#     feature_cols: Sequence[str],
#     *,
#     prefix: str = "group_",
# ) -> Dict[str, float]:
#     """Build a feature→value dict by **name** from a row, clamping columns that
#     start with `prefix`. Useful for freezing exogenous `group_*` variables.
#     """
#     return {c: float(row[c]) for c in feature_cols if c.startswith(prefix)}


# def build_scenarios_by_future(
#     full_df: pd.DataFrame,
#     feature_cols: Sequence[str],
#     future_ids: Sequence[Any],
#     *,
#     prefix: str = "group_",
# ) -> List[Dict[str, float]]:
#     """Construct a list of scenario dicts (feature→value clamps) for each `future_id`.

#     Parameters
#     ----------
#     full_df : DataFrame containing at least 'future_id' and `feature_cols`.
#     feature_cols : columns that the model expects (order must match training).
#     future_ids : list of future IDs to include.
#     prefix : only features that start with this prefix will be clamped.

#     Returns
#     -------
#     list of dict[str,float] with one dict per future.
#     """
#     scenarios: List[Dict[str, float]] = []
#     for fid in future_ids:
#         row = full_df.loc[full_df["future_id"] == fid, feature_cols].iloc[0]
#         scenarios.append(fixed_inputs_from_row(row, feature_cols, prefix=prefix))
#     return scenarios

# # --------------------------------------------------------------------------- #
# # 1) Wrap a scikit-learn Pipeline as a pymoo Problem (vectorized evaluate)
# # --------------------------------------------------------------------------- #

# def _make_problem(
#     pipeline,
#     X_ref: pd.DataFrame,
#     train_targets: Sequence[str],
#     opt_targets: Sequence[str],
#     goals: Sequence[str],
#     *,
#     fixed_inputs: Optional[Mapping[str, float]] = None,           # global clamps
#     bounds: str = "data",
#     scenario_sets: Optional[Sequence[Mapping[str, float]]] = None, # per-future clamps
#     scenario_weights: Optional[Sequence[float]] = None,            # for aggregate mode
#     scenario_mode: str = "aggregate",                              # "aggregate" | "stack"
#     scenario_agg: str = "mean",                                    # "mean" | "worst"
# ) -> Problem:
#     """Create a vectorized pymoo Problem with optional **multi-future** evaluation.

#     Behaviour
#     ---------
#     - Decision variables = all columns of `X_ref`, **except** those clamped via
#       `fixed_inputs` (global) and/or per-scenario dicts in `scenario_sets`.
#     - Objectives are derived from `opt_targets` with `goals` converted to
#       minimization space internally.

#     Multi-future modes
#     -------------------
#     • aggregate: For each candidate, evaluate under every scenario and aggregate
#       per-objective in **minimization** space using a weighted **mean** (default)
#       or **worst** (elementwise max). The number of objectives equals
#       `len(opt_targets)`.

#     • stack: Concatenate objectives across scenarios → number of objectives is
#       `len(opt_targets) * n_scenarios` in minimization space. Useful to expose
#       trade-offs across futures explicitly.
#     """
#     if not isinstance(X_ref, pd.DataFrame):
#         raise TypeError("X_ref must be a pandas DataFrame")

#     cols = X_ref.columns.to_list()

#     # data-space per-feature bounds
#     lb_data = X_ref.min().values.astype(float)
#     ub_data = X_ref.max().values.astype(float)
#     span = (ub_data - lb_data)
#     span[span == 0.0] = 1.0  # guard against zero-span columns

#     use_unit = (str(bounds).lower() == "unit")
#     xl, xu = (np.zeros_like(lb_data), np.ones_like(lb_data)) if use_unit else (lb_data, ub_data)

#     # minimization sign:  1 for min,  -1 for max
#     goals = _canonical_goals(goals)
#     sign = np.array([1.0 if g == "min" else -1.0 for g in goals], dtype=float)
#     col_idx = [train_targets.index(t) for t in opt_targets]

#     # Global fixed inputs
#     fixed_inputs = dict(fixed_inputs or {})
#     fixed_mask_global = np.array([c in fixed_inputs for c in cols])
#     fixed_vals_global = np.array([fixed_inputs.get(c, 0.0) for c in cols], dtype=float)

#     # Scenario sets (list[dict feature->value])
#     scen_list = list(scenario_sets) if scenario_sets is not None else [None]
#     n_scen = len(scen_list)

#     scen_masks: List[Optional[np.ndarray]] = []
#     scen_vals:  List[Optional[np.ndarray]] = []
#     for s in scen_list:
#         if s is None:
#             scen_masks.append(None); scen_vals.append(None)
#         else:
#             sd = dict(s)
#             m = np.array([c in sd for c in cols])
#             v = np.array([sd.get(c, 0.0) for c in cols], dtype=float)
#             scen_masks.append(m); scen_vals.append(v)

#     # weights for aggregate
#     if scenario_mode == "aggregate":
#         if scenario_weights is None:
#             w = np.ones(n_scen, float) / n_scen
#         else:
#             w = np.asarray(scenario_weights, float)
#             if w.shape[0] != n_scen:
#                 raise ValueError("scenario_weights length must match number of scenario_sets")
#             w = w / (w.sum() if w.sum() != 0 else 1.0)
#     else:
#         w = None

#     # objectives count
#     if scenario_mode == "stack":
#         n_obj = len(opt_targets) * n_scen
#     else:
#         n_obj = len(opt_targets)

#     class _SkProblem(Problem):
#         def __init__(self):
#             super().__init__(
#                 n_var=len(cols),
#                 n_obj=n_obj,
#                 xl=xl,
#                 xu=xu,
#                 elementwise_evaluation=False,
#             )

#         def _evaluate(self, X, out, **kwargs):
#             X = np.asarray(X, float)

#             # unit-space → data-space
#             Xd = (X * span + lb_data) if use_unit else X.copy()

#             # apply global fixed inputs
#             if fixed_inputs:
#                 Xd[:, fixed_mask_global] = fixed_vals_global[fixed_mask_global]

#             # Evaluate under each scenario
#             Fs = []  # list of (n_pop, len(opt_targets)) in minimization space
#             for k in range(n_scen):
#                 Xm = Xd
#                 m = scen_masks[k]; v = scen_vals[k]
#                 if m is not None:
#                     Xm = Xd.copy()
#                     Xm[:, m] = v[m]

#                 preds = pipeline.predict(pd.DataFrame(Xm, columns=cols))
#                 preds = np.asarray(preds)
#                 if preds.ndim == 1:
#                     preds = preds[:, None]

#                 Fk_min = preds[:, col_idx] * sign  # minimization space
#                 Fs.append(Fk_min)

#             if scenario_mode == "aggregate":
#                 Fs_stack = np.stack(Fs, axis=2)   # (n_pop, n_metrics, n_scen)
#                 if scenario_agg == "mean":
#                     # weighted mean over scenarios
#                     F_min = np.tensordot(Fs_stack, w, axes=([2],[0]))
#                 elif scenario_agg == "worst":
#                     # elementwise max = worst in minimization space
#                     F_min = Fs_stack.max(axis=2)
#                 else:
#                     raise ValueError("scenario_agg must be 'mean' or 'worst'")
#                 out["F"] = F_min
#             else:  # "stack"
#                 F_min = np.concatenate(Fs, axis=1)  # (n_pop, len(metrics)*n_scen)
#                 out["F"] = F_min

#     return _SkProblem()

# # --------------------------------------------------------------------------- #
# # 2) optional local L-BFGS-B refinement (single-objective only)
# # --------------------------------------------------------------------------- #

# def _local_bfgs(x0: np.ndarray, prob: Problem) -> np.ndarray:
#     """One L-BFGS-B pass; returns x0 unchanged if SciPy absent or fails."""
#     if scipy_min is None:
#         return x0

#     def f(x):
#         d: Dict[str, Any] = {}
#         prob._evaluate(np.asarray([x]), d)  # minimization space
#         return float(np.asarray(d["F"]).reshape(-1)[0])

#     try:
#         res = scipy_min(
#             f,
#             x0,
#             method="L-BFGS-B",
#             bounds=list(zip(prob.xl, prob.xu)),
#             options={"maxiter": 200, "ftol": 1e-9},
#         )
#         return res.x if (res is not None and getattr(res, "success", False)) else x0
#     except Exception:
#         return x0

# # --------------------------------------------------------------------------- #
# # 3) public optimise()  — robust for 1..K objectives; multi-future capable
# # --------------------------------------------------------------------------- #

# def optimise(
#     pipeline,
#     X_ref: pd.DataFrame,
#     *,
#     train_targets: Sequence[str],
#     targets: Sequence[str] | str,
#     goals: Sequence[str] | str = "max",
#     n_gen: int = 100,
#     pop_size: int = 200,
#     n_restarts: int = 3,
#     fixed_inputs: Optional[Mapping[str, float]] = None,
#     bounds: str = "data",
#     # multi-future knobs (all optional)
#     scenario_sets: Optional[Sequence[Mapping[str, float]]] = None,
#     scenario_weights: Optional[Sequence[float]] = None,
#     scenario_mode: str = "aggregate",       # "aggregate" | "stack"
#     scenario_agg: str = "mean",             # used only for aggregate
#     scenario_names: Optional[Sequence[str]] = None,  # naming for stack outputs
# ):
#     """Evolutionary optimisation (GA/NSGA-II) for sklearn Pipelines.

#     Parameters
#     ----------
#     pipeline : sklearn Pipeline with `.predict(X) -> (n_samples, n_targets)`.
#     X_ref : DataFrame of reference features; defines column order and data-bounds.
#     train_targets : list of all metric names predicted by the pipeline (in order).
#     targets : metric name or list of names to optimize (subset of train_targets).
#     goals : "min"/"max" (synonyms OK) per target.
#     n_gen, pop_size, n_restarts : genetic algorithm controls and robust restarts.

#     fixed_inputs : dict[feature_name, value]
#         Globally clamp these features for *all* evaluations.

#     bounds : {"data","unit"}
#         - "data" (default): bounds from X_ref min/max per feature.
#         - "unit": search in [0,1]^n and auto-unscale to data space before predict.

#     Multi-future evaluation (optional)
#     ----------------------------------
#     scenario_sets : list of dicts (feature→value) used as **per-scenario clamps**
#         (e.g., exogenous `group_*`). Each candidate is evaluated once per scenario.
#     scenario_weights : weights for aggregate mean across scenarios.
#     scenario_mode : "aggregate" (default) or "stack".
#     scenario_agg : "mean" (weighted) or "worst" (elementwise max in min-space).
#     scenario_names : optional labels used to name stacked objectives (order must
#         match scenario_sets).

#     Returns
#     -------
#     • Single objective → `(best_X : pd.Series, best_value : float)`
#     • Multi objective → `(pareto_X : pd.DataFrame, pareto_F : pd.DataFrame)`
#       - In aggregate mode: columns(pareto_F) = `list(targets)` (original metric space).
#       - In stack mode: columns(pareto_F) = flattened labels per (scenario × target).

#     Notes
#     -----
#     - Objectives are converted to minimization internally; returned values are in
#       the **original** metric space (sign un-flipped).
#     - Decision variables exclude any clamped features (global or per-scenario).
#     """
#     # normalize inputs
#     if isinstance(targets, str):
#         targets = [targets]
#     if isinstance(goals, str):
#         goals = [goals]

#     goals = _canonical_goals(goals)

#     if len(targets) != len(goals):
#         raise ValueError(f"targets ({len(targets)}) and goals ({len(goals)}) lengths differ")
#     if any(g not in ("min", "max") for g in goals):
#         bad = [g for g in goals if g not in ("min", "max")]
#         raise ValueError(f"Unrecognized goals: {bad}. Use one of {{'min','max'}}.")
#     if n_restarts < 1:
#         raise ValueError("n_restarts must be >= 1")

#     # build problem
#     prob = _make_problem(
#         pipeline,
#         X_ref,
#         train_targets,
#         opt_targets=list(targets),
#         goals=list(goals),
#         fixed_inputs=fixed_inputs,
#         bounds=bounds,
#         scenario_sets=scenario_sets,
#         scenario_weights=scenario_weights,
#         scenario_mode=scenario_mode,
#         scenario_agg=scenario_agg,
#     )

#     # choose algorithm per dimensionality and run restarts
#     def _one_run(seed: int):
#         algo = GA(pop_size=pop_size, eliminate_duplicates=True) if len(targets) == 1 \
#                else NSGA2(pop_size=pop_size, eliminate_duplicates=True)
#         return minimize(prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)

#     results = [_one_run(seed) for seed in range(n_restarts)]

#     # ---------------- single-objective --------------------------------------
#     if len(targets) == 1:
#         # pick best by minimized F
#         def _scalar_F(res):
#             f = np.asarray(res.F).reshape(-1)
#             if f.size != 1:
#                 raise ValueError(f"Expected single-objective F with size 1, got shape {res.F.shape}")
#             return float(f[0])

#         best_ga = min(results, key=_scalar_F)

#         # optional local refinement
#         x0 = np.asarray(best_ga.X).reshape(-1)
#         try:
#             best_x_vec = _local_bfgs(x0, prob)
#         except Exception:
#             best_x_vec = x0

#         # minimization-space objective value → original metric value
#         f_minspace = np.asarray(prob.evaluate(best_x_vec[None, :])).reshape(-1)[0]
#         best_y = -f_minspace if goals[0] == "max" else f_minspace

#         best_x = pd.Series(best_x_vec, index=X_ref.columns, name="best_policy")
#         return best_x, float(best_y)

#     # ---------------- multi-objective ---------------------------------------
#     # Stack final populations across restarts
#     X_all = np.vstack([np.asarray(r.pop.get("X")) for r in results])
#     F_all = np.vstack([np.asarray(r.pop.get("F")) for r in results])  # minimization space

#     # Non-dominated sort in minimization space
#     nds = _NDS()
#     I = nds.do(F_all, only_non_dominated_front=True)
#     X_nd = X_all[I]
#     F_nd_min = F_all[I]

#     # convert back to ORIGINAL metric space (undo minimization sign)
#     sign = np.array([1.0 if g == "min" else -1.0 for g in goals], dtype=float)

#     # In stack mode, F is (n_pop, len(targets)*n_scen) in min-space; unflip per target block
#     if scenario_sets is not None and str(scenario_mode).lower() == "stack":
#         n_scen = len(scenario_sets)
#         n_t = len(targets)
#         F_sh = F_nd_min.reshape(F_nd_min.shape[0], n_t, n_scen)
#         F_un = F_sh * sign[None, :, None]
#         F_nd_orig = F_un.reshape(F_nd_min.shape[0], n_t * n_scen)
#     else:
#         F_nd_orig = F_nd_min * sign

#     # de-duplicate across restarts (tolerant to tiny float noise)
#     XF = np.concatenate([X_nd, F_nd_orig], axis=1)
#     XF_df = pd.DataFrame(np.round(XF, 10))
#     XF_df = XF_df.drop_duplicates(ignore_index=True)

#     n_x = X_ref.shape[1]
#     X_unique = XF_df.iloc[:, :n_x].to_numpy()
#     F_unique = XF_df.iloc[:, n_x:].to_numpy()

#     pareto_X = pd.DataFrame(X_unique, columns=X_ref.columns)

#     # Name objective columns
#     if scenario_sets is not None and str(scenario_mode).lower() == "stack":
#         n_scen = len(scenario_sets)
#         n_t = len(targets)
#         if scenario_names is None:
#             scenario_names = [f"scen{i}" for i in range(n_scen)]
#         if len(scenario_names) != n_scen:
#             raise ValueError("scenario_names length must match scenario_sets")
#         cols = [f"{targets[j]}@{scenario_names[k]}" for k in range(n_scen) for j in range(n_t)]
#     else:
#         cols = list(targets)

#     pareto_F = pd.DataFrame(F_unique, columns=cols)
#     return pareto_X, pareto_F

# # --------------------------------------------------------------------------- #
# # 4) plotting helpers
# # --------------------------------------------------------------------------- #

# def plot_pareto(F: pd.DataFrame | np.ndarray, *, labels: List[str] | None = None):
#     """Quick 2-D / 3-D / scatter-matrix plot of objective space."""
#     F = pd.DataFrame(F) if not isinstance(F, pd.DataFrame) else F.copy()
#     if labels is not None:
#         F.columns = labels[: F.shape[1]]

#     n = F.shape[1]
#     if n == 2:
#         plt.scatter(F.iloc[:, 0], F.iloc[:, 1], s=20)
#         plt.xlabel(F.columns[0]); plt.ylabel(F.columns[1])
#         plt.title("Pareto front (2-D)")
#     elif n == 3:
#         from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
#         ax = plt.figure().add_subplot(111, projection="3d")
#         ax.scatter(F.iloc[:, 0], F.iloc[:, 1], F.iloc[:, 2], s=20)
#         ax.set_xlabel(F.columns[0]); ax.set_ylabel(F.columns[1]); ax.set_zlabel(F.columns[2])
#         ax.set_title("Pareto front (3-D)")
#     else:
#         pd.plotting.scatter_matrix(F, figsize=(2.3 * n, 2.3 * n), diagonal="kde", s=15)
#         plt.suptitle("Pareto front (scatter-matrix)", y=1.02)

#     plt.tight_layout(); plt.show()


# def plot_best_features(series: pd.Series, *, top_n: int = 15, title: str = ""):
#     """Horizontal bar chart of the largest-magnitude features of a policy."""
#     s = series.abs().nlargest(top_n).sort_values()
#     colours = ["steelblue" if float(series[i]) >= 0 else "salmon" for i in s.index]

#     plt.figure(figsize=(6, 0.45 * top_n + 1))
#     plt.barh(s.index, s.values, color=colours)
#     plt.xlabel("Magnitude"); plt.title(title or "Top features")
#     plt.tight_layout(); plt.show()


 



# """
# narrative_profiles_optimization_utils.py  •  v2.3
# ===============================================

# Evolutionary optimisation helpers with:

# • GA / NSGA-II + multi-seed restarts.
# • Optional local L-BFGS-B polish for single-metric runs (uses SciPy if present).
# • Plot helpers.

# Public API
# ----------
# optimise()             – GA / NSGA-II wrapper
# plot_pareto()          – quick Pareto scatter
# plot_best_features()   – horizontal bar of a policy’s top feature magnitudes
# """

# from __future__ import annotations
# import numpy as np, pandas as pd, matplotlib.pyplot as plt
# from typing import Sequence, List

# # pymoo core
# from pymoo.core.problem import Problem
# from pymoo.optimize import minimize
# from pymoo.algorithms.moo.nsga2 import NSGA2
# try:
#     from pymoo.algorithms.soo.genetic_algorithm import GA          # pymoo ≥0.6
# except ModuleNotFoundError:                                         # pymoo 0.5
#     from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# # optional SciPy
# try:
#     from scipy.optimize import minimize as scipy_min
# except ModuleNotFoundError:
#     scipy_min = None   # module runs fine without SciPy – local search skipped

# __all__ = ["optimise", "plot_pareto", "plot_best_features"]

# # --------------------------------------------------------------------------- #
# # 1. wrap a scikit-learn Pipeline as a pymoo Problem
# # --------------------------------------------------------------------------- #
# def _make_problem(
#     pipeline,
#     X_ref: pd.DataFrame,
#     train_targets: Sequence[str],
#     opt_targets: Sequence[str],
#     goals: Sequence[str],
# ) -> Problem:
#     lb = X_ref.min().values.astype(float)
#     ub = X_ref.max().values.astype(float)
#     sign = np.array([1.0 if g == "min" else -1.0 for g in goals])
#     col_idx = [train_targets.index(t) for t in opt_targets]

#     class _SkProblem(Problem):
#         def __init__(self):
#             super().__init__(n_var=len(lb), n_obj=len(opt_targets),
#                              xl=lb, xu=ub, elementwise_evaluation=False)

#         def _evaluate(self, X, out, **kwargs):
#             preds = pipeline.predict(pd.DataFrame(X, columns=X_ref.columns))
#             preds = np.asarray(preds)
#             if preds.ndim == 1:
#                 preds = preds[:, None]
#             out["F"] = preds[:, col_idx] * sign

#     return _SkProblem()

# # --------------------------------------------------------------------------- #
# # 2. optional local L-BFGS-B refinement  (single-objective only)
# # --------------------------------------------------------------------------- #
# def _local_bfgs(x0: np.ndarray, prob: Problem) -> np.ndarray:
#     """One L-BFGS-B step; returns x0 unchanged if SciPy absent."""
#     if scipy_min is None:
#         return x0

#     def f(x):
#         d = {}
#         prob._evaluate(np.asarray([x]), d)
#         return d["F"][0, 0]

#     res = scipy_min(
#         f, x0, method="L-BFGS-B",
#         bounds=list(zip(prob.xl, prob.xu)),
#         options={"maxiter": 200, "ftol": 1e-9},
#     )
#     return res.x if res.success else x0

# # ---------------------------------------------------------------------------
# # 3. public optimise()  — general, robust for 1..K objectives
# # ---------------------------------------------------------------------------
# from typing import Sequence
# import numpy as np
# import pandas as pd
# from pymoo.optimize import minimize
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# def optimise(
#     pipeline,
#     X_ref: pd.DataFrame,
#     *,
#     train_targets: Sequence[str],
#     targets: Sequence[str] | str,
#     goals: Sequence[str] | str = "max",
#     n_gen: int = 100,
#     pop_size: int = 200,
#     n_restarts: int = 3,
# ):
#     """
#     Evolutionary optimisation.

#     Returns
#     -------
#     • single metric → (best_X : pd.Series, best_value : float)
#     • multi  metric → (pareto_X : pd.DataFrame, pareto_F : pd.DataFrame)

#     Notes
#     -----
#     - Goals are case-insensitive and may include synonyms e.g. "min", "minimize",
#       "minimise", "max", "maximize".
#     - For multi-objective, pareto_F values are in the ORIGINAL metric space
#       (i.e., not negated for max goals).
#     """
#     if isinstance(targets, str):
#         targets = [targets]
#     if isinstance(goals, str):
#         goals = [goals]

#     # --- canonicalize goals -------------------------------------------------
#     _syn_min = {"min", "minimize", "minimise", "minimization", "minimisation"}
#     _syn_max = {"max", "maximize", "maximise", "maximization", "maximisation"}
#     goals = [g.strip().lower() for g in goals]
#     goals = ["min" if g in _syn_min else "max" if g in _syn_max else g for g in goals]

#     if len(targets) != len(goals):
#         raise ValueError(f"targets ({len(targets)}) and goals ({len(goals)}) lengths differ")

#     if any(g not in ("min", "max") for g in goals):
#         bad = [g for g in goals if g not in ("min", "max")]
#         raise ValueError(f"Unrecognized goals: {bad}. Use one of {{'min','max'}} (case-insensitive).")

#     if n_restarts < 1:
#         raise ValueError("n_restarts must be >= 1")

#     # construct pymoo problem in minimization space (your existing helper)
#     prob = _make_problem(pipeline, X_ref, train_targets, targets, goals)

#     # --- one run (single vs multi) -----------------------------------------
#     def _one_run(seed: int):
#         if len(targets) == 1:
#             algo = GA(pop_size=pop_size, eliminate_duplicates=True)
#         else:
#             algo = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
#         return minimize(prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)

#     results = [_one_run(seed) for seed in range(n_restarts)]

#     # ---------------- single-objective -------------------------------------
#     if len(targets) == 1:
#         # pymoo returns the best (min) value in r.F; pick the best across restarts
#         def _scalar_F(res):
#             f = np.asarray(res.F).reshape(-1)
#             if f.size != 1:
#                 raise ValueError(f"Expected single-objective F with size 1, got shape {res.F.shape}")
#             return float(f[0])

#         best_ga = min(results, key=_scalar_F)

#         # optional local refinement (keep GA if refinement fails)
#         x0 = np.asarray(best_ga.X).reshape(-1)
#         try:
#             best_x_vec = _local_bfgs(x0, prob)  # expects (x_init, problem)
#         except Exception:
#             best_x_vec = x0

#         # Evaluate in minimization space then convert back to original metric
#         f_minspace = prob.evaluate(best_x_vec[None, :]).reshape(-1)[0]
#         # If the user wanted "max", the objective was negated inside the problem.
#         # Convert back to ORIGINAL metric value here.
#         best_y = -f_minspace if goals[0] == "max" else f_minspace

#         best_x = pd.Series(best_x_vec, index=X_ref.columns, name="best_policy")
#         return best_x, float(best_y)

#     # ---------------- multi-objective --------------------------------------
#     # Stack all final pops, then extract the nondominated front (rank 0)
#     X_all = np.vstack([np.asarray(r.pop.get("X")) for r in results])
#     F_all = np.vstack([np.asarray(r.pop.get("F")) for r in results])

#     # Non-dominated sort in minimization space
#     nds = NonDominatedSorting()
#     I = nds.do(F_all, only_non_dominated_front=True)
#     X_nd = X_all[I]
#     F_nd_minspace = F_all[I]

#     # Convert back to ORIGINAL metric space by undoing any internal negation
#     # For goals == "max", values in minimization space were negated -> re-negate
#     sign = np.array([(-1.0 if g == "max" else 1.0) for g in goals], dtype=float)
#     F_nd_original = F_nd_minspace * sign

#     # De-duplicate across restarts (tolerant to tiny float noise)
#     # Concatenate X and F for a stable duplicate check, then split back.
#     XF = np.concatenate([X_nd, F_nd_original], axis=1)
#     # Round to reduce floating-point jitter; adjust decimals as needed.
#     XF_df = pd.DataFrame(np.round(XF, 10))
#     XF_df = XF_df.drop_duplicates(ignore_index=True)
#     n_x = X_ref.shape[1]
#     X_unique = XF_df.iloc[:, :n_x].to_numpy()
#     F_unique = XF_df.iloc[:, n_x:].to_numpy()

#     pareto_X = pd.DataFrame(X_unique, columns=X_ref.columns)
#     pareto_F = pd.DataFrame(F_unique, columns=list(targets))
#     return pareto_X, pareto_F


# # --------------------------------------------------------------------------- #
# # 4. plotting helpers
# # --------------------------------------------------------------------------- #
# def plot_pareto(F: pd.DataFrame | np.ndarray, *, labels: List[str] | None = None):
#     """Quick 2-D / 3-D / scatter-matrix plot of objective space."""
#     F = pd.DataFrame(F) if not isinstance(F, pd.DataFrame) else F.copy()
#     if labels is not None:
#         F.columns = labels[: F.shape[1]]

#     n = F.shape[1]
#     if n == 2:
#         plt.scatter(F.iloc[:, 0], F.iloc[:, 1], s=20)
#         plt.xlabel(F.columns[0]); plt.ylabel(F.columns[1])
#         plt.title("Pareto front (2-D)")
#     elif n == 3:
#         from mpl_toolkits.mplot3d import Axes3D  # noqa
#         ax = plt.figure().add_subplot(111, projection="3d")
#         ax.scatter(F.iloc[:, 0], F.iloc[:, 1], F.iloc[:, 2], s=20)
#         ax.set_xlabel(F.columns[0]); ax.set_ylabel(F.columns[1]); ax.set_zlabel(F.columns[2])
#         ax.set_title("Pareto front (3-D)")
#     else:
#         pd.plotting.scatter_matrix(F, figsize=(2.3 * n, 2.3 * n),
#                                    diagonal="kde", s=15)
#         plt.suptitle("Pareto front (scatter-matrix)", y=1.02)

#     plt.tight_layout(); plt.show()


# def plot_best_features(series: pd.Series, *, top_n: int = 15, title: str = ""):
#     """Horizontal bar chart of the largest-magnitude features of a policy."""
#     s = series.abs().nlargest(top_n).sort_values()
#     colours = ["steelblue" if series[i] >= 0 else "salmon" for i in s.index]

#     plt.figure(figsize=(6, 0.45 * top_n + 1))
#     plt.barh(s.index, s.values, color=colours)
#     plt.xlabel("Magnitude"); plt.title(title or "Top features")
#     plt.tight_layout(); plt.show()
