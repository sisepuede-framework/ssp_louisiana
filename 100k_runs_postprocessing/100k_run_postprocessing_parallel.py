#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time
import traceback
from io import StringIO
from typing import List, Optional, Tuple
import multiprocessing as mp

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
            df[c] = s.astype("string")
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
    "LOCAL_FILES": {}
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
    base_primary_id: int
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

    # Local cached files (written by parent)
    PROC_STATE["LOCAL_FILES"] = {
        "output_df": os.path.join(cache_dir, "output_df.pkl"),
        "input_df": os.path.join(cache_dir, "input_df.pkl"),
        "attribute_primary_df": os.path.join(cache_dir, "attribute_primary_df.pkl"),
        "attribute_strategy_df": os.path.join(cache_dir, "attribute_strategy_df.pkl"),
        "emission_targets_df": os.path.join(cache_dir, "emission_targets_df.pkl"),
    }

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
        df_to_upload = decomposed_df[["primary_id", "time_period", "total_emissions"]]
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
        # from costs_benefits_ssp.model.cb_data_model import TXTable, CostFactor, TransformationCost, StrategyInteraction  # not directly needed here

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

        # local config path is the same for all workers
        cb_config_path = os.path.join(os.path.dirname(PROC_STATE["CACHE_DIR"]), "config", "cb_config_params.xlsx")
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

        # Clean up temp file for non-base
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
        PRIMARY_ID_BASE
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

if __name__ == "__main__":
    main()
