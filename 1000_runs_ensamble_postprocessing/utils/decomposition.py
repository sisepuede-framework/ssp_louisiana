import os
import numpy as np
import pandas as pd




class TemporalDecomposition:
    

    @ staticmethod
    def build_t0_anchor_from_file(path_csv: str,
                                time_period_ref: int,
                                baseline_id: int = 0,
                                region_col: str = "region",
                                id_col: str = "primary_id",
                                time_col: str = "time_period") -> dict[str, pd.Series]:
        """
        Returns {region: Series(co2e_* -> float)} for the baseline id at t0.
        """
        df = pd.read_csv(path_csv)
        tv = [c for c in df.columns if "co2e_" in c and not c.startswith("emission_co2e_subsector_total_")]
        mask = (df[id_col] == baseline_id) & (df[time_col] == time_period_ref)
        if not mask.any():
            raise ValueError(f"Baseline id={baseline_id} not found at t0={time_period_ref} in {path_csv}")
        anchors: dict[str, pd.Series] = {}
        for r, sub in df.loc[mask].groupby(region_col):
            s = sub.iloc[0][tv].astype(float)
            s.name = f"{r}_{baseline_id}"
            anchors[r] = s
        return anchors
    
    @staticmethod
    def rescale(z,
                rall,
                data_all,
                te_all,
                initial_conditions_id,
                dir_output,
                time_period_ref,
                run,
                global_t0_anchor_by_region: dict[str, pd.Series] | None = None,
                global_baseline_id: int = 0):

        tregion = rall[z]
        data = data_all[data_all["region"] == tregion].copy()

        # Variables to process (sorted for determinism)
        tv1_all = sorted(c for c in data.columns
                        if "co2e_" in c and not c.startswith("emission_co2e_subsector_total_"))

        # Build index and sort
        data["Index"] = data["region"] + "_" + data["primary_id"].astype(str)
        data = data.sort_values(["Index", "time_period"])
        inds = data["Index"].unique()

        # Choose intended reference label & ensure we have a usable anchor
        intended_ref_id = int(initial_conditions_id[0])
        ref_inds = f"{tregion}_{intended_ref_id}"
        inds_at_t0 = set(data.loc[data["time_period"] == time_period_ref, "Index"].unique())
        anchor_series = (global_t0_anchor_by_region or {}).get(tregion, None)

        # ---------- Compute diffs and pct_diffs per Index & variable ----------
        pct_diffs_list = []
        for ind in inds:
            sub = data.loc[data["Index"] == ind, ["Index", "time_period"] + tv1_all].copy()
            step_frames = []

            for tv in tv1_all:
                pivot = sub[["Index", "time_period", tv]].copy()
                col = pivot[tv]
                pivot[tv] = col.fillna(col.mean(skipna=True))
                if pd.isna(pivot[tv].mean()):
                    pivot[tv] = 0.0

                if float(pivot[tv].abs().sum()) == 0.0:
                    pivot[f"pct_diff_{tv}"] = 0.0
                    pivot[f"diff_{tv}"] = 0.0
                else:
                    pivot["diff"] = pivot[tv].diff().fillna(0.0)
                    prev = pivot[tv].shift(1)
                    pct = np.divide(
                        pivot["diff"].to_numpy(),
                        prev.to_numpy(),
                        out=np.zeros_like(pivot["diff"].to_numpy(), dtype=float),
                        where=(prev.to_numpy() != 0) & (~np.isnan(prev.to_numpy()))
                    )
                    pct[0] = 0.0
                    pivot[f"pct_diff_{tv}"] = pct
                    pivot[f"diff_{tv}"] = pivot["diff"]

                step_frames.append(pivot[["Index", "time_period", f"pct_diff_{tv}", f"diff_{tv}"]])

            # Use OUTER merge so a missing timestamp in one var doesn't drop t0 for all
            merged = step_frames[0]
            for df_ in step_frames[1:]:
                merged = pd.merge(merged, df_, on=["Index", "time_period"], how="outer")
            pct_diffs_list.append(merged)

        pct_diffs = pd.concat(pct_diffs_list, ignore_index=True)
        pct_diffs = pct_diffs.sort_values(["Index", "time_period"])

        # Set MultiIndex
        data = data.set_index(["Index", "time_period"])
        pct_diffs = pct_diffs.set_index(["Index", "time_period"])

        # ---------- Sector/gas mapping ----------
        te_all = te_all.copy()
        te_all["sector_gas"] = te_all.index.astype(str) + "-" + te_all["Subsector"] + "-" + te_all["Gas"]
        sector_gas_all = te_all["sector_gas"].unique()

        for sector_gas_i in sector_gas_all:
            row = te_all.loc[te_all["sector_gas"] == sector_gas_i]
            if row.empty:
                continue

            tv1 = row["Vars_list"].iloc[0]     # list[str]
            target_total = row["tvalue"].iloc[0]

            # --- Determine reference t0 vector for tv1 (in-batch ref row OR global anchor OR fallback min-id-at-t0) ---
            if (ref_inds, time_period_ref) in data.index:
                ref_vec = data.loc[(ref_inds, time_period_ref), tv1].astype(float)
            elif anchor_series is not None:
                ref_vec = pd.Series({v: float(anchor_series.get(v, 0.0)) for v in tv1}, index=tv1, dtype=float)
            else:
                # fallback: pick min id present in this batch at t0
                if not inds_at_t0:
                    continue
                cand_id = min(int(ix.split("_")[-1]) for ix in inds_at_t0)
                alt_ref = f"{tregion}_{cand_id}"
                if (alt_ref, time_period_ref) not in data.index:
                    continue
                ref_vec = data.loc[(alt_ref, time_period_ref), tv1].astype(float)

            # deviation factor so that sum(tv1) at t0 hits the target_total
            uncalibrated_total = float(np.nansum(ref_vec.values))
            deviation_factor = 1.0 if (np.isnan(uncalibrated_total) or uncalibrated_total == 0.0) \
                            else float(target_total) / uncalibrated_total

            # If the in-batch reference exists, scale that single row at t0
            if (ref_inds, time_period_ref) in data.index:
                data.loc[(ref_inds, time_period_ref), tv1] = ref_vec.values * deviation_factor

            # --- reconstruct each ID from t0 onward with common t0 (from in-batch ref OR global anchor) ---
            for ind in inds:
                for tv in tv1:
                    if (ref_inds, time_period_ref) in data.index:
                        init_value = float(data.loc[(ref_inds, time_period_ref), tv])
                    elif anchor_series is not None:
                        init_value = float(anchor_series.get(tv, 0.0)) * deviation_factor
                    else:
                        init_value = float(ref_vec.get(tv, 0.0)) * deviation_factor

                    try:
                        pct_series = pct_diffs.loc[(ind,), f"pct_diff_{tv}"].sort_index().fillna(0.0)
                        diff_series = pct_diffs.loc[(ind,), f"diff_{tv}"].sort_index().fillna(0.0)
                    except KeyError:
                        continue
                    if time_period_ref not in pct_series.index:
                        # if this ID doesn't have t0, skip writing (your pipeline filters >= t0)
                        continue

                    pct_after = pct_series.loc[pct_series.index > time_period_ref]
                    times_out = [time_period_ref] + list(pct_after.index)

                    if init_value == 0.0:
                        diffs_after = diff_series.loc[diff_series.index > time_period_ref]
                        vals_after = np.cumsum(diffs_after.to_numpy())
                        vals = np.concatenate([[0.0], vals_after])
                    else:
                        vals_after = init_value * np.cumprod(1.0 + pct_after.to_numpy())
                        vals = np.concatenate([[init_value], vals_after])

                    data.loc[(ind, times_out), tv] = vals

        # ---------- Final t0 equalization for all co2e_ vars (mapped + unmapped) ----------
        processed_vars = set(v for _, r in te_all.iterrows() for v in r["Vars_list"])
        all_co2e = [c for c in data.columns if "co2e_" in c and not c.startswith("emission_co2e_subsector_total_")]
        unmapped = [v for v in all_co2e if v not in processed_vars]

        # Reference values at t0: in-batch ref if present, else global anchor
        if (ref_inds, time_period_ref) in data.index:
            ref_vals = data.loc[(ref_inds, time_period_ref), unmapped] if unmapped else pd.Series(dtype=float)
        elif anchor_series is not None:
            ref_vals = pd.Series({v: float(anchor_series.get(v, np.nan)) for v in unmapped})
        else:
            ref_vals = pd.Series(index=unmapped, dtype=float)

        t0_mask = pd.IndexSlice[:, time_period_ref]
        for v in unmapped:
            if v in data.columns and v in ref_vals.index and pd.notna(ref_vals[v]):
                data.loc[t0_mask, v] = float(ref_vals[v])

        # ---------- Sector totals ----------
        data_reset = data.reset_index()
        subsectors = te_all["Subsector"].unique()
        for s in subsectors:
            sector_vars = [v for vars_list in te_all.loc[te_all["Subsector"] == s, "Vars_list"] for v in vars_list]
            sector_vars = [v for v in sector_vars if v in data_reset.columns]
            data_reset[f"emission_co2e_subsector_total_{s}"] = (
                data_reset[sector_vars].sum(axis=1) if sector_vars else 0.0
            )

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
