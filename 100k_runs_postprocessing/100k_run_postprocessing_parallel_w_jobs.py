#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import traceback
from io import StringIO
from typing import List, Optional, Tuple
import multiprocessing as mp
import shutil  # NEW: for cleanup
import re

# Limit threaded libs to avoid oversubscription when we parallelize by processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import yaml
import boto3

# rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter

try:
    from tqdm import tqdm
except Exception:
    # fallback if tqdm not installed
    tqdm = lambda x, **k: x

# --------------------------
# Logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID:%(process)d] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("postproc")

# --------------------------
# Small helpers
# --------------------------
def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def fetch_csv_from_s3(s3_resource, bucket_name, key):
    obj = s3_resource.Object(bucket_name, key)
    content = obj.get()['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(content))

def upload_df_to_s3(df, s3_resource, bucket, key):
    buf = StringIO()
    df.to_csv(buf, index=False)
    s3_resource.Object(bucket, key).put(Body=buf.getvalue(), ContentType="text/csv")
    logger.info(f"Uploaded to s3://{bucket}/{key}")

def sanitize_for_r(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def _is_scalar(x):
        import pandas as pd
        return not isinstance(x, (list, dict, pd.Series))
    for c in df.columns:
        s = df[c]
        if not s.map(_is_scalar).all():
            df[c] = s.astype(str)
        elif s.dtype == "object":
            df[c] = df[c].astype("string")
    return df

# --------------------------
# Core domain funcs (same as your originals, with small tweaks)
# --------------------------
def postprocess_cba(cb_raw_df: pd.DataFrame) -> pd.DataFrame:
    parts = cb_raw_df["variable"].astype(str).str.split(":", n=4, expand=True)
    parts.columns = ["name", "sector", "cb_type", "item_1", "item_2"]
    cb_data = pd.concat([cb_raw_df, parts], axis=1)
    cb_data["value"] = cb_data["value"] / 1e9
    cb_data["Year"]  = cb_data["time_period"] + 2015

    group_cols = ["cb_type", "strategy_code", "primary_id", "future_id", "Year"]
    cb_agg = (
        cb_data.groupby(group_cols, dropna=False, as_index=False)["value"]
        .sum()
        .rename(columns={"value": "Cumulative"})
    )

    agg_cb_df = (
        cb_agg.pivot_table(
            index=["primary_id", "future_id", "strategy_code", "Year"],
            columns="cb_type",
            values="Cumulative",
            aggfunc="sum"
        )
        .reset_index()
    )
    agg_cb_df.columns.name = None
    cols = ["primary_id","future_id", "strategy_code", "Year"]
    for c in ["air_pollution", "technical_cost"]:
        if c in agg_cb_df.columns:
            cols.append(c)
    return agg_cb_df[cols]

# --- Clean-up helper (keep only baseline, remove cache) ---  # NEW
def cleanup_tmp(tmp_dir: str, base_id: int, keep_tmp: bool = False) -> None:
    """Remove all louisiana_*.csv except the baseline, and delete tmp/cache."""
    if keep_tmp:
        logger.info("Keeping tmp/ contents (flag --keep-tmp is set).")
        return

    base_name = f"louisiana_{base_id}.csv"
    try:
        for fn in os.listdir(tmp_dir):
            if fn.startswith("louisiana_") and fn.endswith(".csv") and fn != base_name:
                fp = os.path.join(tmp_dir, fn)
                try:
                    os.remove(fp)
                    logger.info(f"Deleted: {fn}")
                except Exception as e:
                    logger.warning(f"Could not delete {fn}: {e}")
    except FileNotFoundError:
        pass

    cache_dir = os.path.join(tmp_dir, "cache")
    if os.path.isdir(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            logger.info("Removed tmp/cache/")
        except Exception as e:
            logger.warning(f"Could not remove tmp/cache/: {e}")

# --------------------------
# Global (per-process) state for workers
# --------------------------
PROC_STATE = {
    "S3": None,
    "BUCKET": None,
    "R_FUNC": None,
    "TMP_DIR": None,
    "TARGET_COUNTRY": None,
    "TIME_PERIOD_REF": None,
    "S3_DECOMP_PREFIX": None,
    "S3_CB_PREFIX": None,
    "BASE_ID": None,
    # cached on disk; workers will load from here
    "CACHE_DIR": None,
    "LOCAL_FILES": {},
    # NEW: config dir so workers can find cb_config_params.xlsx reliably
    "CONFIG_DIR": None,
}

def worker_init(
    profile_name: str,
    bucket_name: str,
    r_script_path: str,
    tmp_dir: str,
    cache_dir: str,
    target_country: str,
    time_period_ref: int,
    s3_decomp_prefix: str,
    s3_cb_prefix: str,
    base_primary_id: int,
    config_dir: str,  # NEW
):
    """Runs once per worker process."""
    # boto3 resource in this process
    session = boto3.Session(profile_name=profile_name)
    s3_resource = session.resource('s3')

    # R init in this process
    ro.r['source'](r_script_path)
    r_rescale = ro.globalenv['rescale']  # function from your R script

    PROC_STATE["S3"] = s3_resource
    PROC_STATE["BUCKET"] = bucket_name
    PROC_STATE["R_FUNC"] = r_rescale
    PROC_STATE["TMP_DIR"] = tmp_dir
    PROC_STATE["TARGET_COUNTRY"] = target_country
    PROC_STATE["TIME_PERIOD_REF"] = time_period_ref
    PROC_STATE["S3_DECOMP_PREFIX"] = s3_decomp_prefix
    PROC_STATE["S3_CB_PREFIX"] = s3_cb_prefix
    PROC_STATE["BASE_ID"] = base_primary_id
    PROC_STATE["CACHE_DIR"] = cache_dir
    PROC_STATE["CONFIG_DIR"] = config_dir  # NEW

    # Local cached files (written by parent)
    PROC_STATE["LOCAL_FILES"] = {
        "output_df": os.path.join(cache_dir, "output_df.pkl"),
        "input_df": os.path.join(cache_dir, "input_df.pkl"),
        "attribute_primary_df": os.path.join(cache_dir, "attribute_primary_df.pkl"),
        "attribute_strategy_df": os.path.join(cache_dir, "attribute_strategy_df.pkl"),
        "emission_targets_df": os.path.join(cache_dir, "emission_targets_df.pkl"),
    }

# --------------- Here goes the job estimation helper function ---------------
def compute_fgtv_css(
    base_case: pd.DataFrame,
    tmp_dir: str,
    primary_id_to_decompose: int
) -> None:
    """
    Compute emissions abatement and CAPEX/OPEX for IPPU fugitive gases + CCS deltas.
    Writes: fugitive_emissions_and_ccs_{primary_id}.csv to tmp_dir
    """

    # gases & sectors we attempt to process
    relevant_gases = [
        "n2o","sf6","c4f6","c4f8o","c2f6","c3f8","c6f14","c5f8","cc4f8"
    ]
    sectors = ["chemicals", "electronics", "metals"]

    # ---- Cost map (replace with your actual values if needed) ----
    # Access via COSTS[sector].get(gas, {"capex":0,"opex":0})
    COSTS = {
        "chemicals": {
            "n2o":  {"capex": 20*0.2, "opex": 20*0.8},
        },
        "electronics": {
            "sf6":  {"capex": 40*0.5, "opex": 40*0.5},
            "c4f6": {"capex": 40*0.5, "opex": 40*0.5},
            "c2f6": {"capex": 40*0.5, "opex": 40*0.5},
            "c3f8": {"capex": 40*0.5, "opex": 40*0.5},
            "c5f8": {"capex": 40*0.5, "opex": 40*0.5},
            "cc4f8":{"capex": 40*0.5, "opex": 40*0.5},
        },
        "metals": {
            "sf6":  {"capex": 20*0.7, "opex": 20*0.7},
            "c2f6": {"capex": 20*0.7, "opex": 20*0.7},
        },
    }

    # ---- Init output frame keyed to input index ----
    emissions_avoided_by_sector = pd.DataFrame(
        {
            "primary_id": base_case["primary_id"],
            "time_period": base_case["time_period"],
        },
        index=base_case.index,
    )

    # ---- Loop over gases/sectors, compute abated emissions ----
    for gas in relevant_gases:
        for sector in sectors:
            abatement_cols = [
                c for c in base_case.columns
                if c.startswith(f"ef_ippu_tonne_{gas}_per_tonne_production_{sector}")
            ]
            emissions_cols = [
                c for c in base_case.columns
                if c.startswith(f"emission_co2e_{gas}_ippu_production_{sector}")
            ]

            if not abatement_cols or not emissions_cols:
                # columns not present in this dataset → skip
                continue

            abatement_factor = base_case[abatement_cols[0]]
            emissions = base_case[emissions_cols[0]]

            # Require a nonzero baseline and at least some non-null values
            if abatement_factor.isna().all():
                continue
            try:
                baseline = abatement_factor.iloc[0]
            except Exception:
                continue
            if baseline == 0:
                continue

            # relative factor vs baseline; if factor==baseline ⇒ ratio=1 ⇒ zero abatement
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = baseline / abatement_factor.replace(0, np.nan)

            if not np.isfinite(ratio).any():
                continue

            # Emissions abated (positive means abated vs baseline)
            abated = (ratio - 1.0) * emissions

            # Fetch costs; default to 0 if undefined
            costs = COSTS.get(sector, {}).get(gas, {"capex": 0.0, "opex": 0.0})
            capex_per_unit = costs.get("capex", 0.0)
            opex_per_unit  = costs.get("opex", 0.0)

            emissions_avoided_by_sector[f"emissions_abated_{sector}_{gas}"] = abated.fillna(0.0)
            emissions_avoided_by_sector[f"capex_emissions_abated_{sector}_{gas}"] = (abated * capex_per_unit).fillna(0.0)
            emissions_avoided_by_sector[f"opex_emissions_abated_{sector}_{gas}"] = (abated * opex_per_unit).fillna(0.0)

    # ---- CCS delta vs baseline ----
    ccs_col = "emission_co2e_subsector_total_ccsq"
    if ccs_col in base_case.columns and not base_case[ccs_col].isna().all():
        try:
            baseline_ccs = base_case[ccs_col].iloc[0]
        except Exception:
            baseline_ccs = 0.0
        ccs_delta = base_case[ccs_col] - baseline_ccs
    else:
        # if no CCS column present, assume zero deltas
        ccs_delta = pd.Series(0.0, index=base_case.index)

    emissions_avoided_by_sector["ccs"] = ccs_delta.fillna(0.0)

    # time-decayed cost factors; ensure time_period is numeric
    tp = pd.to_numeric(emissions_avoided_by_sector["time_period"], errors="coerce").fillna(0)
    capex_ccs = -500 * (0.96 ** tp)
    opex_ccs  =  -50 * (0.96 ** tp)

    emissions_avoided_by_sector["capex_ccs"] = (emissions_avoided_by_sector["ccs"] * capex_ccs).fillna(0.0)
    emissions_avoided_by_sector["opex_ccs"]  = (emissions_avoided_by_sector["ccs"] * opex_ccs ).fillna(0.0)

    # if there are inf or -inf make them 0
    emissions_avoided_by_sector.replace([np.inf, -np.inf], 0, inplace=True)

    # ---- Save ----
    out_path = os.path.join(tmp_dir, f"fugitive_emissions_and_ccs_{primary_id_to_decompose}.csv")
    emissions_avoided_by_sector.to_csv(out_path, index=False)


def compute_scoe(base_case: pd.DataFrame,
                tmp_dir: str,
                primary_id_to_decompose: int) -> None:
    
    # Define fuels and sectors
    relevant_fuels = ['solid_biomass', 
                    'coal',  
                    'diesel', 
                    'electricity',
                    'gasoline', 
                    'hydrocarbon_gas_liquids',
                    'hydrogen',
                    'kerosene',
                    'natural_gas']

    sectors = ['commercial_municipal',
                'other_se',
                'residential']
    
    # init template df
    scoe_fuel_demand_by_sector = pd.DataFrame({'primary_id': base_case['primary_id'], 
                                               'time_period': base_case['time_period']}, 
                                               index=base_case.index)
    
    # Industrial cost parameters
    capex_industrial_electricity = 92666.6 * 21
    capex_industrial_other       = 92666.6 * 12
    opex_industrial_electricity  = 92666.6 * 2.5
    opex_industrial_other        = 92666.6 * 4.5
    capex_multiplier_efficiency = 5560000
    opex_multiplier_efficiency = 0

    # Loop over fuels and sectors
    for fuel in relevant_fuels:
        # find the efficiency column(s) for this fuel
        # find the demand column(s) for this fuel
        for sector in sectors:
            eff_cols = [c for c in base_case.columns
                    if c.startswith(f'efficfactor_scoe_heat_energy_{sector}_{fuel}')]
            fuel_efficiency = base_case[eff_cols[0]]
            sector_dem_cols = [c for c in base_case.columns
                    if (f'scalar_scoe_heat_energy_demand_{sector}' in c)]
            
            sector_fuel_fraction_cols = [c for c in base_case.columns
                    if (f'frac_scoe_heat_energy_{sector}_{fuel}' in c)]

            if len(sector_fuel_fraction_cols)>0 and len(sector_dem_cols)>0:
                sector_total_demand = base_case[sector_dem_cols[0]]
                sector_fuel_fraction = base_case[sector_fuel_fraction_cols[0]]

                if (sector_fuel_fraction*sector_total_demand).sum()>0 or fuel=='electricity':
                    sector_fuel_demand = sector_fuel_fraction*sector_total_demand
                    scoe_fuel_demand_by_sector[f'energy_demand_{sector}_{fuel}'] = sector_fuel_demand
                    if fuel=='electricity':
                        scoe_fuel_demand_by_sector[f'energy_demand_capex_{sector}_{fuel}'] = sector_fuel_demand*capex_industrial_electricity
                        scoe_fuel_demand_by_sector[f'energy_demand_opex_{sector}_{fuel}'] = sector_fuel_demand*opex_industrial_electricity
                    else:
                        scoe_fuel_demand_by_sector[f'energy_demand_capex_{sector}_{fuel}'] = sector_fuel_demand*capex_industrial_other
                        scoe_fuel_demand_by_sector[f'energy_demand_opex_{sector}_{fuel}'] = sector_fuel_demand*opex_industrial_other
                    sector_fuel_consumed = sector_fuel_demand/fuel_efficiency
                    sector_fuel_consumed_baseline = sector_fuel_demand/fuel_efficiency.iloc[0]
                    sector_change_in_fuel_consumed = sector_fuel_consumed_baseline-sector_fuel_consumed
                    scoe_fuel_demand_by_sector[f'efficiency_energy_saving_{sector}_{fuel}'] = sector_change_in_fuel_consumed
                    scoe_fuel_demand_by_sector[f'efficiency_capex_{sector}_{fuel}'] = sector_change_in_fuel_consumed*capex_multiplier_efficiency
                    scoe_fuel_demand_by_sector[f'efficiency_opex_{sector}_{fuel}'] = sector_change_in_fuel_consumed*opex_multiplier_efficiency
    
    #  if there are inf or -inf make them 0
    scoe_fuel_demand_by_sector.replace([np.inf, -np.inf], 0, inplace=True)
    scoe_fuel_demand_by_sector.to_csv(os.path.join(tmp_dir, f'scoe_{primary_id_to_decompose}.csv'), index=False)

def compute_inen_fuel_efficiency(
            base_case: pd.DataFrame,
            tmp_dir: str,
            primary_id_to_decompose: int) -> None:
        
    # Define fuels and sectors
    relevant_fuels = ['biomass', 
                'coal', 
                'coke', 
                'diesel', 
                'electricity',
                'furnace_gas',
                'gasoline', 
                'hydrocarbon_gas_liquids',
                'hydrogen',
                'kerosene',
                'natural_gas',
                'oil']

    sectors = ['agriculture_and_livestock',
            'cement',
            'chemicals',
            'electronics',
            'glass',
            'lime_and_carbonite',
            'metals',
            'mining',
            'other_product_manufacturing',
            'paper',
            'plastic',
            'recycled_glass',
            'recycled_metals',
            'recycled_paper',
            'recycled_plastic',
            'recycled_rubber_and_leather',
            'recycled_textiles',
            'recycled_wood',
            'rubber_and_leather',
            'textiles',
            'wood']
    
    # init template df
    ind_fuel_demand_by_sector = pd.DataFrame({'primary_id': base_case['primary_id'],
                                                'time_period': base_case['time_period']},
                                                index=base_case.index)
    
    # Industrial cost parameters
    capex_industrial_electricity = 92666.6 * 21
    capex_industrial_other       = 92666.6 * 12
    opex_industrial_electricity  = 92666.6 * 2.5
    opex_industrial_other        = 92666.6 * 4.5
    capex_multiplier_efficiency = 10000000
    opex_multiplier_efficiency = 0

    # Loop over fuels and sectors
    for fuel in relevant_fuels:
        # find the efficiency column(s) for this fuel
        eff_cols = [c for c in base_case.columns
                    if c.startswith(f'efficfactor_enfu_industrial_energy_fuel_{fuel}')]
        fuel_efficiency = base_case[eff_cols[0]]
        # find the demand column(s) for this fuel
        for sector in sectors:
            sector_dem_cols = [c for c in base_case.columns
                    if (f'energy_demand_inen_{sector}' in c)]
            
            sector_fuel_fraction_cols = [c for c in base_case.columns
                    if (f'frac_inen_energy_{sector}_{fuel}' in c)]

            if len(sector_fuel_fraction_cols)>0 and len(sector_dem_cols)>0:
                sector_total_demand = base_case[sector_dem_cols[0]]
                sector_fuel_fraction = base_case[sector_fuel_fraction_cols[0]]

                if (sector_fuel_fraction*sector_total_demand).sum()>0 or fuel=='electricity':
                    sector_fuel_demand = sector_fuel_fraction*sector_total_demand
                    ind_fuel_demand_by_sector[f'energy_demand_{sector}_{fuel}'] = sector_fuel_demand
                    if fuel=='electricity':
                        ind_fuel_demand_by_sector[f'energy_demand_capex_{sector}_{fuel}'] = sector_fuel_demand*capex_industrial_electricity
                        ind_fuel_demand_by_sector[f'energy_demand_opex_{sector}_{fuel}'] = sector_fuel_demand*opex_industrial_electricity
                    else:
                        ind_fuel_demand_by_sector[f'energy_demand_capex_{sector}_{fuel}'] = sector_fuel_demand*capex_industrial_other
                        ind_fuel_demand_by_sector[f'energy_demand_opex_{sector}_{fuel}'] = sector_fuel_demand*opex_industrial_other
                    sector_fuel_consumed = sector_fuel_demand/fuel_efficiency
                    sector_fuel_consumed_baseline = sector_fuel_demand/fuel_efficiency.iloc[0]
                    sector_change_in_fuel_consumed = sector_fuel_consumed_baseline-sector_fuel_consumed
                    ind_fuel_demand_by_sector[f'efficiency_energy_saving_{sector}_{fuel}'] = sector_change_in_fuel_consumed
                    ind_fuel_demand_by_sector[f'efficiency_capex_{sector}_{fuel}'] = sector_change_in_fuel_consumed*capex_multiplier_efficiency
                    ind_fuel_demand_by_sector[f'efficiency_opex_{sector}_{fuel}'] = sector_change_in_fuel_consumed*opex_multiplier_efficiency

    #  if there are inf or -inf make them 0
    ind_fuel_demand_by_sector.replace([np.inf, -np.inf], 0, inplace=True)

    # Save results
    ind_fuel_demand_by_sector.to_csv(os.path.join(tmp_dir, f'industrial_energy_cost_{primary_id_to_decompose}.csv'), index=False)

def compute_transportation(
        base_case: pd.DataFrame,
        tmp_dir: str,
        primary_id_to_decompose: int) -> None:
    
    dem_col = [c for c in base_case.columns 
           if 'energy_demand_enfu_subsector_total_pj_trns_fuel_electricity' in c][0]
    eff_col = [c for c in base_case.columns 
            if c.startswith('elecfuelefficiency_trns_road_light')][0]

    # 2) Compute saved volume (PJ) vs. baseline
    elec_demand   = base_case[dem_col]
    elec_eff      = base_case[eff_col]
    vol_now       = elec_demand / elec_eff
    vol_base      = elec_demand / elec_eff.iloc[0]
    saved_pj      = vol_base - vol_now

    # 3) Compute electricity cost at $880 000 per PJ saved
    capex = saved_pj * 880_000

    # 4) Build the output DataFrame
    output_trns_elec = pd.DataFrame({
        'primary_id':                             base_case['primary_id'],
        'time_period':                            base_case['time_period'],
        'electricity_volume_saved_in_PJ':         saved_pj,
        'electricity_transportation_cost_$':      capex
    }, index=base_case.index)

    relevant_fuels = [
    'diesel',
    'gasoline',
    'hydrocarbon_gas_liquids',
    'hydrogen',
    'natural_gas'
    ]

    # 4) Prepare accumulators
    total_saved_volume = pd.Series(0.0, index=base_case.index)
    demand_trns = pd.DataFrame({'time_period': base_case['time_period']},
                            index=base_case.index)

    # 5) Loop & pattern-match per fuel
    for fuel in relevant_fuels:
        # 5a) demand cols specific to this fuel
        dem_cols = [
            c for c in base_case.columns
            if f'energy_demand_enfu_subsector_total_pj_trns_fuel_{fuel}' in c
        ]
        # 5b) efficiency cols for this fuel
        eff_cols = [
            c for c in base_case.columns
            if f'fuelefficiency_trns_road_light_{fuel}' in c
        ]

        # print(f"Fuel={fuel!r}: dem_cols={dem_cols}, eff_cols={eff_cols}")
        if not dem_cols or not eff_cols:
            # print(f"  → skipping {fuel!r} (no matching columns)\n")
            continue

        fuel_demand     = base_case[dem_cols[0]]
        fuel_efficiency = base_case[eff_cols[0]]

        demand_trns[fuel] = fuel_demand

        vol_now  = fuel_demand / fuel_efficiency
        vol_base = fuel_demand / fuel_efficiency.iloc[0]
        total_saved_volume += (vol_base - vol_now)

    # 6) Build output (with identifiers)
    output_trns_non_elec = pd.DataFrame({
        'primary_id': base_case['primary_id'],
        'time_period': base_case['time_period'],
        'transportation_volume_saved_in_PJ': total_saved_volume
    }, index=base_case.index)

    # 7) Apply $880 000 per PJ
    capex_multiplier_trns = 880_000
    output_trns_non_elec['transportation_efficiency_capex'] = (
        output_trns_non_elec['transportation_volume_saved_in_PJ'] * capex_multiplier_trns
    )
    output_trns_non_elec['transportation_efficiency_opex'] = 0

    # 2) Detect the vehicle‐km column
    dist_cols = [
        c for c in base_case.columns
        if c.startswith('vehicle_distance_traveled_trns_road_light')
    ]
    if not dist_cols:
        raise KeyError("No vehicle_distance_traveled_trns_road_light_* column found")
    dist_col = dist_cols[0]
    vehicle_distance = base_case[dist_col]  # units: vkm

    # 3) Define the three fuel‐mix fraction columns
    relevant_fuels = ['diesel', 'electricity', 'gasoline']
    frac_cols = {
        fuel: f'frac_trns_fuelmix_road_light_{fuel}'
        for fuel in relevant_fuels
    }
    for col in frac_cols.values():
        if col not in base_case.columns:
            raise KeyError(f"Missing fraction column: {col}")

    # 4) Record the period-0 (baseline) shares
    frac_baseline = {
        fuel: base_case[col].iloc[0]
        for fuel, col in frac_cols.items()
    }

    # 5) Compute how many vkm have switched fuels since the baseline
    switched_from_diesel    = vehicle_distance * (frac_baseline['diesel']    - base_case[frac_cols['diesel']])
    switched_from_gasoline  = vehicle_distance * (frac_baseline['gasoline']  - base_case[frac_cols['gasoline']])
    switched_to_electricity = vehicle_distance * (base_case[frac_cols['electricity']] - frac_baseline['electricity'])

    # 6) Use your lookup‐table multipliers (in $ per vkm)
    #    * note the NEGATIVE sign for the electrification “cost” from the table
    multipliers = {
        'electricity': {'cost': 0.039, 'saving':  0.012},
        'diesel':      {'cost':  0.000, 'saving':  0.000},
        'gasoline':    {'cost':  0.000, 'saving':  0.000},
    }

    # 7) Accumulate total cost & total savings
    cost_series   = pd.Series(0.0, index=base_case.index)
    saving_series = pd.Series(0.0, index=base_case.index)

    for fuel in relevant_fuels:
        frac = base_case[frac_cols[fuel]]
        vkm  = vehicle_distance * frac
        cost_series   += vkm * multipliers[fuel]['cost']
        saving_series += vkm * multipliers[fuel]['saving']

    # 8) Build the output table
    output_fs = pd.DataFrame({
        'primary_id':                         base_case['primary_id'],
        'time_period':                        base_case['time_period'],
        'vehicle_distance_traveled_total_vkm': vehicle_distance,
        'switched_from_diesel_vkm':           switched_from_diesel,
        'switched_from_gasoline_vkm':         switched_from_gasoline,
        'switched_to_electricity_vkm':        switched_to_electricity,
        'fuel_switch_cost_$':                 cost_series,
        'fuel_switch_savings_$':              saving_series,
        'fuel_switch_net_cost_$':             cost_series - saving_series,
    }, index=base_case.index)

    # 2) Grab the heavy‐duty + public‐transit vehicle‐km
    dist_patterns = [
        r"^vehicle_distance_traveled_trns_road_heavy_.*$",
        r"^vehicle_distance_traveled_trns_public.*",
    ]
    dist_cols = [c for c in base_case.columns if any(re.match(p, c) for p in dist_patterns)]
    if not dist_cols:
        raise KeyError("No heavy‐duty/public distance columns found")
    vehicle_distance = base_case[dist_cols].sum(axis=1)

    # 3) Define your fuel‐mix fraction columns & read off baseline shares
    segments = ["freight", "regional"]
    relevant_fuels = [
        "biofuels", "diesel", "electricity",
        "gasoline", "hydrocarbon_gas_liquids",
        "hydrogen", "natural_gas"
    ]
    frac_cols = {
        fuel: [f"frac_trns_fuelmix_road_heavy_{seg}_{fuel}" for seg in segments]
        for fuel in relevant_fuels
    }
    # make sure they all exist
    for fuel, cols in frac_cols.items():
        for col in cols:
            if col not in base_case.columns:
                raise KeyError(f"Missing fraction column: {col}")

    # baseline share at t=0 (sum of freight+regional)
    frac_baseline = {
        fuel: base_case[cols].sum(axis=1).iloc[0]
        for fuel, cols in frac_cols.items()
    }

    # 4) Define your $/vkm multipliers for each fuel
    multipliers = {
        "biofuels":             {"cost": 0.00,   "saving": 0.00},
        "diesel":               {"cost": 0.00,   "saving": 0.00},
        "electricity":          {"cost": 0.042,  "saving": 0.020},
        "gasoline":             {"cost": 0.00,   "saving": 0.00},
        "hydrocarbon_gas_liquids": {"cost":0.00, "saving": 0.00},
        "hydrogen":             {"cost": 0.00,   "saving": 0.00},
        "natural_gas":          {"cost": 0.00,   "saving": 0.00},
    }

    # 5) Compute current & baseline costs/savings per fuel
    cost_now   = pd.Series(0.0, index=base_case.index)
    cost_base  = pd.Series(0.0, index=base_case.index)
    saving_now  = pd.Series(0.0, index=base_case.index)
    saving_base = pd.Series(0.0, index=base_case.index)

    for fuel in relevant_fuels:
        # current total vkm on this fuel
        cur_frac = base_case[frac_cols[fuel]].sum(axis=1)
        vkm_now  = vehicle_distance * cur_frac
        # baseline total vkm on this fuel
        vkm_base = vehicle_distance * frac_baseline[fuel]
        # accumulate
        cost_now   += vkm_now  * multipliers[fuel]["cost"]
        cost_base  += vkm_base * multipliers[fuel]["cost"]
        saving_now  += vkm_now  * multipliers[fuel]["saving"]
        saving_base += vkm_base * multipliers[fuel]["saving"]

    # 6) Difference from baseline
    cost_series   = cost_now   - cost_base
    saving_series = saving_now - saving_base
    net_series    = cost_series - saving_series

    # 7) Build output
    output_hd = pd.DataFrame({
        "primary_id":                          base_case["primary_id"],
        "region":                              base_case["region"],
        "time_period":                         base_case["time_period"],
        "vehicle_distance_traveled_total_vkm": vehicle_distance,
        "fuel_switch_cost_$":                  cost_series,
        "fuel_switch_savings_$":               saving_series,
        "fuel_switch_net_cost_$":              net_series,
    }, index=base_case.index)

    # ——————————————————————————————————————————
    # 2) Grab all rail electricity consumption (PJ) columns
    # ——————————————————————————————————————————
    rail_patterns = [r"^energy_consumption_trns_rail_.*_electricity$"]
    rail_elec_cols = [
        c for c in base_case.columns
        if any(re.match(p, c) for p in rail_patterns)
    ]
    if not rail_elec_cols:
        raise KeyError("No rail electricity consumption columns found")
    # sum across any sub‐modes (freight, passenger, etc.)
    rail_elec_PJ = base_case[rail_elec_cols].sum(axis=1)

    # ——————————————————————————————————————————
    # 3) Compute “switched to electricity” relative to baseline
    # ——————————————————————————————————————————
    baseline_elec = rail_elec_PJ.iloc[0]
    switched_to_rail_elec_PJ = rail_elec_PJ - baseline_elec

    # ——————————————————————————————————————————
    # 4) Apply cost & saving multipliers ($ per PJ)
    # ——————————————————————————————————————————
    cost_mult   = 422_400_000   # $ per PJ
    saving_mult =   2_377_710   # $ per PJ

    cost_series   = switched_to_rail_elec_PJ * cost_mult
    saving_series = switched_to_rail_elec_PJ * saving_mult
    net_series    = cost_series - saving_series  # net capex

    # ——————————————————————————————————————————
    # 5) Build the output DataFrame
    # ——————————————————————————————————————————
    output_rail = pd.DataFrame({
        "primary_id":                          base_case["primary_id"],
        "region":                              base_case["region"],
        "time_period":                         base_case["time_period"],
        "rail_elec_consumption_PJ":            rail_elec_PJ,
        "switched_to_rail_elec_PJ":            switched_to_rail_elec_PJ,
        "rail_fuel_switch_cost_$":             cost_series,
        "rail_fuel_switch_saving_$":           saving_series,
        "rail_fuel_switch_net_cost_$":         net_series,
    }, index=base_case.index)

    #  if there are inf or -inf make them 0
    output_trns_elec.replace([np.inf, -np.inf], 0, inplace=True)
    output_trns_non_elec.replace([np.inf, -np.inf], 0, inplace=True)
    output_fs.replace([np.inf, -np.inf], 0, inplace=True)
    output_hd.replace([np.inf, -np.inf], 0, inplace=True)
    output_rail.replace([np.inf, -np.inf], 0, inplace=True)

    output_trns_elec.to_csv(os.path.join(tmp_dir, f"transportation_electric_efficiency_cost_{primary_id_to_decompose}.csv"), index=False)
    output_trns_non_elec.to_csv(os.path.join(tmp_dir, f"transportation_non_electric_efficiency_cost_{primary_id_to_decompose}.csv"), index=False)
    output_fs.to_csv(os.path.join(tmp_dir, f'transportation_light_duty_fuel_switch_cost_{primary_id_to_decompose}.csv'), index=False)
    output_hd.to_csv(os.path.join(tmp_dir, f"transportation_heavy_duty_fuel_switch_cost_{primary_id_to_decompose}.csv"), index=False)
    output_rail.to_csv(os.path.join(tmp_dir, f"transportation_rail_fuel_switch_cost_{primary_id_to_decompose}.csv"), index=False)

def compute_energy_production(
        base_case: pd.DataFrame,
        tmp_dir: str,
        primary_id_to_decompose: int) -> None:
    
    ID_COLS = ["primary_id", "time_period"]
    value_cols = [c for c in base_case.columns if c not in ID_COLS]

    long = base_case.melt(
        id_vars=ID_COLS, value_vars=value_cols,
        var_name="variable", value_name="value"
    )

    # ------------ identify three groups -----------------------------
    is_capex = long["variable"].str.contains(
        r"nemomod_entc_discounted_capital_investment_", regex=True
    )
    is_opex = long["variable"].str.contains(
        r"nemomod_entc_discounted_operating_", regex=True
    )
    is_production = long["variable"].str.contains(
        r"nemomod_entc_annual_production_by_technology_", regex=True
    )

    capex_df      = long[is_capex].copy()
    opex_df       = long[is_opex].copy()
    production_df = long[is_production].copy()

    # tag rows
    capex_df["cost_type"] = "capex"
    opex_df["cost_type"]  = "opex"

    # ------------- helper to pull production type -------------------
    def extract_ptype(col):
        m = re.search(r"(?:pp|fp)_(.+)", col)   # grabs text after pp_ / fp_
        return m.group(1) if m else "unknown"

    for _df in (capex_df, opex_df, production_df):
        _df["prod_type"] = _df["variable"].apply(extract_ptype)

    # rename columns for clarity
    capex_df = capex_df.rename(columns={"value": "usd"})
    opex_df  = opex_df.rename(columns={"value": "usd"})
    production_df = production_df.rename(columns={"value": "production"})

    # ---------------- cost aggregation ------------------------------
    cost_long = pd.concat([capex_df, opex_df], ignore_index=True)

    annual_cost = (
        cost_long.groupby(["primary_id", "time_period", "prod_type", "cost_type"], as_index=False)["usd"].sum()
                .pivot(index=["primary_id", "time_period", "prod_type"],
                        columns="cost_type", values="usd")
                .fillna(0)
                .reset_index()
    )
    annual_cost["total_usd"] = annual_cost["capex"] + annual_cost["opex"]

    # ---------------- production aggregation ------------------------
    annual_prod = (
        production_df.groupby(["primary_id", "time_period", "prod_type"], as_index=False)["production"].sum()
    )

    # ---------------- merge cost + production -----------------------
    annual_pt = annual_cost.merge(
        annual_prod, on=["primary_id", "time_period", "prod_type"], how="left"
    ).fillna({"production": 0})

    annual_reg_cost = (
    cost_long.groupby(["primary_id", "time_period", "cost_type"], as_index=False)["usd"].sum()
        .pivot(index=["primary_id", "time_period"], columns="cost_type", values="usd")
        .fillna(0)
        .reset_index()
    )
    annual_reg_cost["total_usd"] = annual_reg_cost["capex"] + annual_reg_cost["opex"]

    annual_reg_prod = (
        production_df.groupby(["primary_id", "time_period"], as_index=False)["production"].sum()
    )

    annual_reg = annual_reg_cost.merge(
        annual_reg_prod, on=["primary_id", "time_period"], how="left"
    ).fillna({"production": 0})

    #  if there are inf or -inf make them 0
    annual_reg.replace([np.inf, -np.inf], 0, inplace=True)
    annual_pt.replace([np.inf, -np.inf], 0, inplace=True)

    # ---------------- save outputs ----------------------------------
    annual_pt.to_csv(os.path.join(tmp_dir, f"baseline_costs_and_production_by_prodtype_{primary_id_to_decompose}.csv"), index=False)
    annual_reg.to_csv(os.path.join(tmp_dir, f"baseline_costs_and_production_timeseries_{primary_id_to_decompose}.csv"), index=False)


def run_decomposition_worker(primary_id_to_decompose: int) -> Tuple[int, Optional[str]]:
    """Process a single primary_id. Returns (primary_id, error_msg or None)."""
    try:
        s3 = PROC_STATE["S3"]
        bucket = PROC_STATE["BUCKET"]
        r_rescale = PROC_STATE["R_FUNC"]
        tmp_dir = PROC_STATE["TMP_DIR"]
        target_country = PROC_STATE["TARGET_COUNTRY"]
        time_ref = PROC_STATE["TIME_PERIOD_REF"]
        s3_decomp_prefix = PROC_STATE["S3_DECOMP_PREFIX"]
        s3_cb_prefix = PROC_STATE["S3_CB_PREFIX"]
        base_id = PROC_STATE["BASE_ID"]

        # Load cached data (local, fast)
        output_df = pd.read_pickle(PROC_STATE["LOCAL_FILES"]["output_df"])
        input_df = pd.read_pickle(PROC_STATE["LOCAL_FILES"]["input_df"])
        attribute_primary_df = pd.read_pickle(PROC_STATE["LOCAL_FILES"]["attribute_primary_df"])
        attribute_strategy_df = pd.read_pickle(PROC_STATE["LOCAL_FILES"]["attribute_strategy_df"])
        emission_targets_df = pd.read_pickle(PROC_STATE["LOCAL_FILES"]["emission_targets_df"])

        # Filter rows for this primary_id
        data_all = output_df.loc[output_df["primary_id"] == primary_id_to_decompose].copy()
        if data_all.empty:
            return primary_id_to_decompose, f"No rows found for primary_id={primary_id_to_decompose}"

        data_all = data_all.fillna(0)
        for col in ("region", "primary_id", "time_period"):
            if col not in data_all.columns:
                return primary_id_to_decompose, f"Missing required column '{col}'"

        # restrict time
        data_all = data_all.loc[data_all["time_period"] >= time_ref].copy()
        if data_all.empty:
            return primary_id_to_decompose, f"All rows filtered out for primary_id={primary_id_to_decompose} with time_ref={time_ref}"

        # sanitize & convert to R
        data_all = sanitize_for_r(data_all)
        te_df = emission_targets_df[["Subsector","Gas","Vars","Edgar_Class", target_country]].rename(columns={target_country: "tvalue"})
        te_df = sanitize_for_r(te_df)

        with localconverter(default_converter + pandas2ri.converter):
            r_data_all = ro.conversion.py2rpy(data_all)
        with localconverter(default_converter + pandas2ri.converter):
            r_te_all   = ro.conversion.py2rpy(te_df)

        rall = data_all["region"].dropna().astype(str).unique().tolist()
        if not rall:
            return primary_id_to_decompose, "No regions found after filtering."

        # R args
        r_rall       = ro.StrVector([str(x) for x in rall])
        r_init_ids   = ro.StrVector([str(primary_id_to_decompose)])
        out_dir = tmp_dir if tmp_dir.endswith(os.sep) else tmp_dir + os.sep
        r_dir_output = ro.StrVector([out_dir])
        r_run        = ro.IntVector([int(primary_id_to_decompose)])
        r_z          = ro.IntVector([1])
        r_time_ref   = ro.IntVector([time_ref])

        # Call R
        _ = r_rescale(r_z, r_rall, r_data_all, r_te_all, r_init_ids, r_dir_output, r_time_ref, r_run)

        # Load decomposed CSV just written by the R function
        local_decomp = os.path.join(tmp_dir, f"louisiana_{primary_id_to_decompose}.csv")
        if not os.path.exists(local_decomp):
            return primary_id_to_decompose, f"Expected decomposed file not found: {local_decomp}"

        decomposed_df = pd.read_csv(local_decomp)
        # Merge inputs needed for CBA
        decomposed_df_merged = pd.merge(decomposed_df, input_df, on=["primary_id","region","time_period"], how="left")

        # Upload small summary to S3 (total emissions)
        decomposed_df['total_emissions'] = decomposed_df.filter(like="emission_co2e_subsector_total").sum(axis=1)

        # Get column names from decomposed_df
        energy_demand_cols = [col for col in decomposed_df.columns if col.startswith("energy_demand_")]
        total_value_enfu_cols = [col for col in decomposed_df.columns if col.startswith("totalvalue_enfu_fuel_consumed_inen")]

        # Get columns from input_df (not decomposed_df)
        frac_inen_energy_cols = [col for col in input_df.columns if col.startswith("frac_inen_energy_")]
        efficfactor_cols = [col for col in input_df.columns if col.startswith("efficfactor_enfu_industrial_energy_fuel")]

        # Merge efficfactor and frac_inen_energy columns from input_df into decomposed_df for upload
        df_to_upload = pd.merge(
            decomposed_df,
            input_df[["primary_id", "region", "time_period"] + efficfactor_cols + frac_inen_energy_cols],
            on=["primary_id", "region", "time_period"],
            how="left"
        )

        # Select columns to keep for upload
        cols_to_keep = (
            ["primary_id", "time_period", "total_emissions"]
            + efficfactor_cols
            + energy_demand_cols
            + frac_inen_energy_cols
            + total_value_enfu_cols
        )
        df_to_upload = df_to_upload[cols_to_keep]
        s3_key = f"{s3_decomp_prefix}emission_total_{primary_id_to_decompose}.csv"
        upload_df_to_s3(df_to_upload, s3, bucket, s3_key)

        # CBA (skip base vs base)
        if primary_id_to_decompose == base_id:
            # keep the decomposed merged as canonical base
            decomposed_df_merged.to_csv(local_decomp, index=False)
            return primary_id_to_decompose, None

        # Base file must exist locally
        base_file = os.path.join(tmp_dir, f"louisiana_{base_id}.csv")
        if not os.path.exists(base_file):
            return primary_id_to_decompose, f"Base decomposed file missing: {base_file}"

        base_decomposed_df = pd.read_csv(base_file)

        # --- Cost Benefits ---
        # lightweight import done here to isolate in worker
        from costs_benefits_ssp.cb_calculate import CostBenefits

        att_primary_copy = attribute_primary_df.copy()
        # ensure baseline first row is zeros (as per your original)
        if len(att_primary_copy.columns) >= 4:
            att_primary_copy.iloc[0] = [0, 0, 0, 0]

        # get ids from attributes
        try:
            future_id = int(att_primary_copy.loc[att_primary_copy["primary_id"] == primary_id_to_decompose, "future_id"].values[0])
            strategy_id = int(att_primary_copy.loc[att_primary_copy["primary_id"] == primary_id_to_decompose, "strategy_id"].values[0])
            strategy_code = attribute_strategy_df.loc[attribute_strategy_df["strategy_id"] == strategy_id, "strategy_code"].values[0]
        except Exception:
            return primary_id_to_decompose, f"Missing strategy/future mapping for primary_id={primary_id_to_decompose}"

        ssp_data = pd.concat([base_decomposed_df, decomposed_df_merged], ignore_index=True)
        ssp_data["primary_id"] = ssp_data["primary_id"].replace({base_id: 0})
        ssp_data = ssp_data.replace(np.nan, 0.0)
        strategy_code_base = "BASE"

        cb = CostBenefits(ssp_data, att_primary_copy, attribute_strategy_df, strategy_code_base)
        cb.ssp_data["future_id"] = 0

        # Read CB config from the actual config dir sent to workers
        cb_config_path = os.path.join(PROC_STATE["CONFIG_DIR"], "cb_config_params.xlsx")
        if not os.path.exists(cb_config_path):
            return primary_id_to_decompose, f"CB config not found: {cb_config_path}"
        cb.load_cb_parameters(cb_config_path)

        results_system = cb.compute_system_cost_for_strategy(strategy_code_tx=strategy_code)
        results_tx     = cb.compute_technical_cost_for_strategy(strategy_code_tx=strategy_code)
        results_all    = pd.concat([results_system, results_tx], ignore_index=True)
        results_all_pp = cb.cb_process_interactions(results_all)
        results_all_pp_shifted = cb.cb_shift_costs(results_all_pp)
        results_all_pp_shifted["primary_id"] = primary_id_to_decompose
        results_all_pp_shifted["future_id"]  = future_id

        agg_cb_df = postprocess_cba(results_all_pp_shifted)
        if agg_cb_df is not None and not agg_cb_df.empty:
            s3_key_cb = f"{s3_cb_prefix}cb_{primary_id_to_decompose}.csv"
            upload_df_to_s3(agg_cb_df, s3, bucket, s3_key_cb)

        # --- Jobs Data ---
        for fn in (compute_fgtv_css, compute_scoe, compute_inen_fuel_efficiency, compute_transportation, compute_energy_production):
            try:
                fn(ssp_data, tmp_dir, primary_id_to_decompose)
                logger.info(f"{fn.__name__} wrote CSV(s) for {primary_id_to_decompose}")
            except Exception as e:
                logger.warning(f"{fn.__name__} failed for {primary_id_to_decompose}: {e}")


        # Here we call the R scripts to compute jobs
        
        
        
        # Clean up temp file for non-base immediately
        if primary_id_to_decompose != base_id:
            try:
                os.remove(local_decomp)
            except Exception:
                pass

        return primary_id_to_decompose, None

    except Exception as e:
        return primary_id_to_decompose, f"{e}\n{traceback.format_exc()}"

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-id", required=True, help="Model output directory id (e.g., 42)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--profile", default=None, help="AWS profile name (overrides YAML)")
    parser.add_argument("--run-id", default=None, help="Override RUN_ID (else use value in script)")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep files in tmp/ (skip cleanup)")  # NEW
    args = parser.parse_args()

    # Paths
    SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    CONFIG_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "config")
    CW_DIR_PATH     = os.path.join(SCRIPT_DIR_PATH, "cw")
    TMP_DIR_PATH    = os.path.join(SCRIPT_DIR_PATH, "tmp")
    R_SCRIPTS_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "r_scripts")
    os.makedirs(TMP_DIR_PATH, exist_ok=True)

    # AWS
    aws_config = read_yaml(os.path.join(CONFIG_DIR_PATH, "aws_credentials_config.yaml"))
    PROFILE_NAME = args.profile or aws_config["profile_name"]
    BUCKET_NAME  = aws_config["bucket_name"]

    # RUN/Prefixes
    RUN_ID = args.run_id or "sisepuede_run_2025-10-07t13;30;14.193421"  # TODO: override as needed / pass via CLI
    RUN_DB_PREFIX = f'run_database/{RUN_ID}/'
    MODEL_OUTPUT_PREFIX = f'{RUN_DB_PREFIX}model_output/region=louisiana/model_output_{args.dir_id}/'
    MODEL_INPUT_PREFIX  = f'{RUN_DB_PREFIX}model_input/region=louisiana/model_input_{args.dir_id}/'
    TRANSFER_PREFIX     = f"transfers/{RUN_ID}/"

    # Decomposition params
    TARGET_COUNTRY = "LA"
    EMISSION_TARGETS_CSV_PATH = os.path.join(CW_DIR_PATH, 'emission_targets_LA_2021.csv')
    R_SCRIPT_PATH = os.path.join(R_SCRIPTS_DIR_PATH, 'intertemporal_function_baseline_mapping_timeref.r')
    TIME_PERIOD_REF = 7
    S3_DECOMPOSED_DIR_PREFIX = f"{RUN_DB_PREFIX}decomposed_outputs/"
    S3_CB_DIR_PREFIX         = f"{RUN_DB_PREFIX}cb_outputs/"

    # S3 session (parent) for initial big downloads only
    session = boto3.Session(profile_name=PROFILE_NAME)
    s3 = session.resource('s3')

    # Fetch data (ONCE) and cache locally for workers
    logger.info("Fetching inputs from S3 (one-time)…")
    output_df = fetch_csv_from_s3(s3, BUCKET_NAME, f'{MODEL_OUTPUT_PREFIX}data.csv')
    input_df  = fetch_csv_from_s3(s3, BUCKET_NAME, f'{MODEL_INPUT_PREFIX}data.csv')
    attribute_primary_df  = fetch_csv_from_s3(s3, BUCKET_NAME, f'{TRANSFER_PREFIX}ATTRIBUTE_PRIMARY.csv')
    attribute_strategy_df = fetch_csv_from_s3(s3, BUCKET_NAME, f'{TRANSFER_PREFIX}ATTRIBUTE_STRATEGY.csv')

    # Determine base id (strategy_id == 0)
    base_rows = attribute_primary_df[attribute_primary_df["strategy_id"] == 0]
    if base_rows.empty:
        raise ValueError("No primary_id with strategy_id == 0 in attribute_primary_df")
    PRIMARY_ID_BASE = int(base_rows.iloc[0]["primary_id"])

    emission_targets_df = pd.read_csv(EMISSION_TARGETS_CSV_PATH)

    # Save caches for workers (fast local load per process)
    CACHE_DIR = os.path.join(TMP_DIR_PATH, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    output_df.to_pickle(os.path.join(CACHE_DIR, "output_df.pkl"))
    input_df.to_pickle(os.path.join(CACHE_DIR, "input_df.pkl"))
    attribute_primary_df.to_pickle(os.path.join(CACHE_DIR, "attribute_primary_df.pkl"))
    attribute_strategy_df.to_pickle(os.path.join(CACHE_DIR, "attribute_strategy_df.pkl"))
    emission_targets_df.to_pickle(os.path.join(CACHE_DIR, "emission_targets_df.pkl"))

    # Build list of primary_ids in this model output dir
    primary_ids = output_df["primary_id"].dropna().astype(int).unique().tolist()
    primary_ids.sort()

    logger.info(f"Found {len(primary_ids)} primary_id values. Base={PRIMARY_ID_BASE}.")
    if PRIMARY_ID_BASE not in primary_ids:
        logger.warning(f"Base primary_id {PRIMARY_ID_BASE} is not present in output_df for DIR {args.dir_id}.")

    # Use spawn start method to keep rpy2 safe
    mp.set_start_method("spawn", force=True)

    # Initialize workers and run
    init_args = (
        PROFILE_NAME,
        BUCKET_NAME,
        R_SCRIPT_PATH,
        TMP_DIR_PATH,
        CACHE_DIR,
        TARGET_COUNTRY,
        TIME_PERIOD_REF,
        S3_DECOMPOSED_DIR_PREFIX,
        S3_CB_DIR_PREFIX,
        PRIMARY_ID_BASE,
        CONFIG_DIR_PATH,  # pass the config dir so workers can read cb_config_params.xlsx
    )

    logger.info(f"Starting pool with {args.workers} workers…")
    with mp.get_context("spawn").Pool(
        processes=args.workers,
        initializer=worker_init,
        initargs=init_args
    ) as pool:
        results = list(tqdm(pool.imap_unordered(run_decomposition_worker, primary_ids), total=len(primary_ids)))

    # Report
    errors = [(pid, err) for pid, err in results if err]
    if errors:
        logger.warning(f"{len(errors)} primary_id(s) failed:")
        for pid, err in errors[:10]:
            logger.warning(f"- {pid}: {err}")
        if len(errors) > 10:
            logger.warning("… (more errors not shown)")
    else:
        logger.info("All primary_id tasks completed successfully.")

    # Final sweep to ensure only baseline remains in tmp/ and cache is gone  # NEW
    cleanup_tmp(TMP_DIR_PATH, PRIMARY_ID_BASE, keep_tmp=args.keep_tmp)

if __name__ == "__main__":
    main()
