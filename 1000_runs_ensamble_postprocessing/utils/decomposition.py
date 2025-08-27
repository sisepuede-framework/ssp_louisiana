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

        idx_t0_ref = (ref_inds, time_period_ref)

        for sector_gas_i in sector_gas_all:
            row = te_all.loc[te_all["sector_gas"] == sector_gas_i]
            if row.empty:
                continue

            tv1 = row["Vars_list"].iloc[0]       # list[str]
            target_total = row["tvalue"].iloc[0]

            # Uncalibrated total at t0 for the *reference* Index
            try:
                uncalibrated_total = data.loc[idx_t0_ref, tv1].sum(skipna=True)
            except KeyError:
                uncalibrated_total = np.nan

            # Deviation factor (only for the reference row at t0)
            if pd.isna(uncalibrated_total) or uncalibrated_total == 0:
                deviation_factor = 1.0
            else:
                deviation_factor = float(target_total) / float(uncalibrated_total)

            # --- scale the reference t0 ONLY ---
            if (ref_inds, time_period_ref) in data.index:
                data.loc[idx_t0_ref, tv1] = data.loc[idx_t0_ref, tv1] * deviation_factor

            # --- reconstruct each ID from t0 onwards, forcing common t0 ---
            for ind in inds:
                for tv in tv1:
                    # Everyone starts at the *reference* t0 (equal initial year)
                    try:
                        init_value = data.loc[idx_t0_ref, tv]
                    except KeyError:
                        continue
                    if pd.isna(init_value):
                        init_value = 0.0
                    init_value = float(init_value)

                    # Pull this ID's pct_diffs/diffs (already sorted by (Index,time_period))
                    try:
                        pct_series = pct_diffs.loc[(ind,), f"pct_diff_{tv}"].sort_index()
                        diff_series = pct_diffs.loc[(ind,), f"diff_{tv}"].sort_index()
                    except KeyError:
                        continue

                    # We only write t0 and AFTER (you already filtered df_in to >= t0)
                    if time_period_ref not in pct_series.index:
                        # if the ID doesn't have t0, skip it
                        continue

                    # Split to exactly t0 and strictly after t0
                    pct_after = pct_series.loc[pct_series.index > time_period_ref]
                    times_out = [time_period_ref] + list(pct_after.index)

                    if init_value == 0.0:
                        # additive path -> cumulative diffs after t0; t0 is 0
                        diffs_after = diff_series.loc[diff_series.index > time_period_ref]
                        vals_after = np.cumsum(diffs_after.to_numpy())
                        vals = np.concatenate([[0.0], vals_after])
                    else:
                        # multiplicative path -> v_t = v_{t-1} * (1 + pct_t)
                        vals_after = init_value * np.cumprod(1.0 + pct_after.to_numpy())
                        vals = np.concatenate([[init_value], vals_after])

                    # Write back (Index, time) pairs for this variable
                    data.loc[(ind, times_out), tv] = vals


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


        # ---------- Final t0 equalization for *all* co2e_ vars (mapped + unmapped) ----------
        # Gather variables we actually processed
        processed_vars = set()
        for _, row in te_all.iterrows():
            for v in row["Vars_list"]:
                processed_vars.add(v)

        # All base vars in the data
        all_co2e = [c for c in data.columns if "co2e_" in c and not c.startswith("emission_co2e_subsector_total_")]

        # Unmapped vars (never touched above) -> force common t0 level too
        unmapped = [v for v in all_co2e if v not in processed_vars]

        if unmapped:
            # reference values at t0 from the reference Index
            idx_t0_ref = (ref_inds, time_period_ref)
            try:
                ref_vals = data.loc[idx_t0_ref, unmapped].astype(float)
            except KeyError:
                ref_vals = pd.Series(index=unmapped, dtype=float)

            # broadcast to all IDs at t0 where we have a ref value
            t0_mask = pd.IndexSlice[:, time_period_ref]
            for v in unmapped:
                if v in data.columns and v in ref_vals.index and pd.notna(ref_vals[v]):
                    data.loc[t0_mask, v] = float(ref_vals[v])


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
    
    @staticmethod
    def assert_equal_t0(
        df: pd.DataFrame,
        time_period_ref: int,
        region_col: str = "region",
        id_col: str = "primary_id",
        time_col: str = "time_period",
        te_all: pd.DataFrame | None = None,
        mapped_only: bool = True,          # <— default to mapped only
        atol: float = 1e-7,                # <— a touch looser for fp noise
        rtol: float = 0.0,
    ) -> bool:
        # choose vars
        if mapped_only and te_all is not None and "Vars_list" in te_all.columns:
            tv = sorted({v for vs in te_all["Vars_list"] for v in vs})
            tv = [c for c in tv if c in df.columns]
        else:
            tv = [c for c in df.columns if "co2e_" in c and not c.startswith("emission_co2e_subsector_total_")]

        # t0 slice, average exact duplicates if any
        keys = [region_col, id_col, time_col]
        g0 = (
            df[df[time_col] == time_period_ref]
            .groupby(keys, as_index=False)[tv]
            .mean(numeric_only=True)
        )

        # for each region and var, compare to a deterministic baseline id (min id present at t0)
        for r, sub in g0.groupby(region_col):
            if sub.empty:
                continue
            baseline_id = int(sub[id_col].min())
            sub = sub.set_index(id_col)
            for v in tv:
                if v not in sub.columns:
                    continue
                s = sub[v]
                if s.notna().any():
                    if baseline_id not in s.index or pd.isna(s.loc[baseline_id]):
                        # pick first non-NaN if baseline is missing this var
                        ref = s.dropna().iloc[0]
                    else:
                        ref = s.loc[baseline_id]
                    same = s.isna() | np.isclose(s, ref, atol=atol, rtol=rtol)
                    if not bool(same.all()):
                        bad = s.index[~same].tolist()
                        raise AssertionError(f"Initial-year mismatch in region={r}, var={v}, ids={bad}")
        return True


    @staticmethod
    def assert_equal_t0_totals(
        df: pd.DataFrame,
        time_period_ref: int = 7,
        prefix: str = "emission_co2e_subsector_total_",
        id_col: str = "primary_id",
        region_col: str = "region",
        time_col: str = "time_period",
        atol: float = 1e-7,
        rtol: float = 0.0,
    ) -> bool:
        totals = [c for c in df.columns if c.startswith(prefix)]
        g = df[df[time_col] == time_period_ref]
        for r, sub in g.groupby(region_col):
            if sub.empty:
                continue
            baseline_id = int(sub[id_col].min())
            sub = sub.set_index(id_col)
            for col in totals:
                if col not in sub.columns:
                    continue
                s = sub[col]
                if s.notna().any():
                    ref = s.loc[baseline_id] if (baseline_id in s.index and pd.notna(s.loc[baseline_id])) else s.dropna().iloc[0]
                    ok = s.isna() | np.isclose(s, ref, atol=atol, rtol=rtol)
                    if not bool(ok.all()):
                        bad = s.index[~ok].tolist()
                        raise AssertionError(f"t0 mismatch in totals: region={r}, var={col}, ids={bad}")
        return True
