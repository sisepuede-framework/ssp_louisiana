import os
import numpy as np
import pandas as pd




class TemporalDecomposition:
    

    @ staticmethod
    # Decomposition function
    def rescale(z, rall, data_all, te_all, initial_conditions_id, dir_output, time_period_ref, run):
        tregion = rall[z]
        data = data_all[data_all["region"] == tregion].copy()

        # Variables to process
        tv1_all = [c for c in data.columns
                if "co2e_" in c and not c.startswith("emission_co2e_subsector_total_")]

        # Build index (string) and sort for deterministic ordering
        data["Index"] = data["region"] + "_" + data["primary_id"].astype(str)
        data = data.sort_values(["Index", "time_period"])
        inds = data["Index"].unique()
        ref_inds = f"{tregion}_{initial_conditions_id[0]}"  # reference (region, primary_id)

        # ---------- Compute diffs and pct_diffs per Index & variable ----------
        pct_diffs_list = []
        for ind in inds:
            sub = data.loc[data["Index"] == ind, ["Index", "time_period"] + tv1_all].copy()
            step_frames = []

            for tv in tv1_all:
                pivot = sub[["Index", "time_period", tv]].copy()

                # Single-cell NAs -> mean; all-NA series -> zeros
                col = pivot[tv]
                pivot[tv] = col.fillna(col.mean(skipna=True))
                if pd.isna(pivot[tv].mean()):
                    pivot[tv] = 0.0

                # Robust zero-series check
                if float(pivot[tv].abs().sum()) == 0.0:
                    pivot[f"pct_diff_{tv}"] = 0.0
                    pivot[f"diff_{tv}"] = 0.0
                else:
                    pivot["diff"] = pivot[tv].diff().fillna(0.0)
                    prev = pivot[tv].shift(1)
                    # pct_diff = diff / prev, but 0 where prev==0 or NaN
                    pct = np.divide(
                        pivot["diff"].to_numpy(),
                        prev.to_numpy(),
                        out=np.zeros_like(pivot["diff"].to_numpy(), dtype=float),
                        where=(prev.to_numpy() != 0) & (~np.isnan(prev.to_numpy()))
                    )
                    pct[0] = 0.0  # first period has 0% change by construction
                    pivot[f"pct_diff_{tv}"] = pct
                    pivot[f"diff_{tv}"] = pivot["diff"]

                step_frames.append(pivot[["Index", "time_period", f"pct_diff_{tv}", f"diff_{tv}"]])

            # Merge all tv frames on (Index, time_period)
            merged = step_frames[0]
            for df_ in step_frames[1:]:
                merged = pd.merge(merged, df_, on=["Index", "time_period"], how="inner")
            pct_diffs_list.append(merged)

        pct_diffs = pd.concat(pct_diffs_list, ignore_index=True)
        pct_diffs = pct_diffs.sort_values(["Index", "time_period"])

        # Set MultiIndex for exact alignment by (Index, time_period)
        data = data.set_index(["Index", "time_period"])
        pct_diffs = pct_diffs.set_index(["Index", "time_period"])

        # ---------- Sector/gas mapping ----------
        te_all = te_all.copy()
        te_all["sector_gas"] = te_all.index.astype(str) + "-" + te_all["Subsector"] + "-" + te_all["Gas"]
        sector_gas_all = te_all["sector_gas"].unique()

        idx_t0 = pd.IndexSlice[:, time_period_ref]

        for sector_gas_i in sector_gas_all:
            row = te_all.loc[te_all["sector_gas"] == sector_gas_i]
            if row.empty:
                continue

            tv1 = row["Vars_list"].iloc[0]       # list of variable names for this sector-gas
            target_total = row["tvalue"].iloc[0]

            # Uncalibrated total at t0 for the *reference* Index
            try:
                uncalibrated_total = data.loc[(ref_inds, time_period_ref), tv1].sum(skipna=True)
            except KeyError:
                uncalibrated_total = np.nan

            # Deviation factor
            if pd.isna(uncalibrated_total) or uncalibrated_total == 0:
                deviation_factor = 1.0
            else:
                deviation_factor = float(target_total) / float(uncalibrated_total)

            # Scale ALL rows at t0 for the current tv1 by deviation_factor (like the R code)
            data.loc[idx_t0, tv1] = data.loc[idx_t0, tv1] * deviation_factor

            # Build each ID's full time series starting from the *single* reference init_value
            for ind in inds:
                for tv in tv1:
                    # Reference init value after deviation scaling
                    try:
                        init_value = data.loc[(ref_inds, time_period_ref), tv]
                    except KeyError:
                        # No reference value for this var; skip
                        continue

                    if pd.isna(init_value):
                        init_value = 0.0

                    # Pull this ID's pct_diffs/diffs, aligned and ordered by time
                    try:
                        pct_series = pct_diffs.loc[(ind,), f"pct_diff_{tv}"].sort_index()  # index: time_period
                        diff_series = pct_diffs.loc[(ind,), f"diff_{tv}"].sort_index()     # index: time_period
                    except KeyError:
                        # No pct/diff for this ID (var not present?); skip
                        continue

                    times = pct_series.index.to_numpy()

                    if float(init_value) == 0.0:
                        vals = (diff_series.cumsum().to_numpy()) * deviation_factor
                    else:
                        vals = float(init_value) * np.cumprod(1.0 + pct_series.to_numpy())

                    # Assign by exact (Index, time_period) keys — order-safe
                    data.loc[(ind, times), tv] = vals

        # ---------- Sector totals ----------
        data_reset = data.reset_index()  # bring Index/time_period back as columns for grouping & sums
        subsectors = te_all["Subsector"].unique()

        for s in subsectors:
            # Collect all vars listed for this subsector
            sector_vars = [v for vars_list in te_all.loc[te_all["Subsector"] == s, "Vars_list"] for v in vars_list]
            sector_vars = [v for v in sector_vars if v in data_reset.columns]  # safety
            if sector_vars:
                data_reset[f"emission_co2e_subsector_total_{s}"] = data_reset[sector_vars].sum(axis=1)
            else:
                data_reset[f"emission_co2e_subsector_total_{s}"] = 0.0

        # ---------- Write output ----------
        out_path = os.path.join(dir_output, f"{tregion}_{run}.csv")
        data_reset.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        return data_reset

    @ staticmethod
    def recompute_subsector_totals(df, te_all, prefix="emission_co2e_subsector_total_"):
        # Build subsector -> list of component vars
        mapping = {
            s: [v for vars_list in te_all.loc[te_all["Subsector"] == s, "Vars_list"] for v in vars_list]
            for s in te_all["Subsector"].unique()
        }
        # Recompute each total as row-wise sum of its components (missing -> 0)
        for s, vars_ in mapping.items():
            vars_ = [v for v in vars_ if v in df.columns]
            if vars_:
                df[f"{prefix}{s}"] = df[vars_].sum(axis=1, skipna=True)
            else:
                df[f"{prefix}{s}"] = 0.0
        return df

    
    
    # --- post-pass: enforce a single global t0 across ALL primary_ids ---
    @ staticmethod
    def enforce_global_t0_equalization(
        df: pd.DataFrame,
        time_period_ref: int = 7,
        region_col: str = "region",
        id_col: str = "primary_id",
        time_col: str = "time_period",
        te_all: pd.DataFrame | None = None,
        mapped_only: bool = False,
    ):
        # choose which variables to equalize
        if mapped_only and te_all is not None and "Vars_list" in te_all.columns:
            mapped = sorted({v for vs in te_all["Vars_list"] for v in vs})
            tv = [c for c in df.columns if c in mapped]
        else:
            tv = [c for c in df.columns if "co2e_" in c and not c.startswith("emission_co2e_subsector_total_")]

        # pick a deterministic global reference primary_id (must exist at t0)
        t0 = df[df[time_col] == time_period_ref]
        if t0.empty:
            raise ValueError(f"No rows at time_period == {time_period_ref} found.")

        ref_id = int(t0[id_col].min())
        ref_row = t0[t0[id_col] == ref_id].head(1)

        # build reference values (fallback to 0.0 if missing)
        ref_vals = {}
        for v in tv:
            if v in ref_row.columns and not ref_row.empty:
                val = ref_row[v].iloc[0]
                ref_vals[v] = 0.0 if pd.isna(val) else float(val)
            else:
                ref_vals[v] = 0.0

        # broadcast to everyone at t0
        t0_mask = df[time_col] == time_period_ref
        for v, val in ref_vals.items():
            df.loc[t0_mask, v] = val

        return df
    
    @staticmethod
    def assert_equal_t0(
        df: pd.DataFrame,
        time_period_ref: int,
        region_col="region",
        id_col="primary_id",
        time_col="time_period",
        atol=1e-9,
    ):
        tv = [c for c in df.columns if "co2e_" in c and not c.startswith("emission_co2e_subsector_total_")]

        # average duplicates (if any) at the same (region, id, time)
        keys = [region_col, id_col, time_col]
        g0 = (
            df[df[time_col] == time_period_ref]
            .groupby(keys, as_index=False)[tv]
            .mean(numeric_only=True)
        )

        for r, sub in g0.groupby(region_col):
            for v in tv:
                s = sub.set_index(id_col)[v]
                if s.notna().any():
                    ref = s.dropna().iloc[0]
                    same = s.isna() | np.isclose(s, ref, atol=atol, rtol=0)
                    if not bool(same.all()):
                        bad = s.index[~same].tolist()
                        raise AssertionError(f"Initial-year mismatch in region={r}, var={v}, ids={bad}")
        return True
    
    @ staticmethod
    def assert_equal_t0_totals(df, time_period_ref=7, prefix="emission_co2e_subsector_total_",
                           id_col="primary_id", region_col="region", time_col="time_period", atol=1e-9):
        totals = [c for c in df.columns if c.startswith(prefix)]
        g = df[df[time_col] == time_period_ref]
        for r, sub in g.groupby(region_col):
            for col in totals:
                s = sub.set_index(id_col)[col]
                if s.notna().any():
                    ref = s.dropna().iloc[0]
                    ok = s.isna() | np.isclose(s, ref, atol=atol, rtol=0)
                    if not bool(ok.all()):
                        bad = s.index[~ok].tolist()
                        raise AssertionError(f"t0 mismatch in totals: region={r}, var={col}, ids={bad}")
        return True
