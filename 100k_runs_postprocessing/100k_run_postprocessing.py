import os
import sys
import logging
import glob

import pandas as pd
import numpy as np
import yaml
import boto3

from io import StringIO

# rpy2: bridge Python <-> R
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter

# Cost benefits
from costs_benefits_ssp.cb_calculate import CostBenefits
from costs_benefits_ssp.model.cb_data_model import TXTable, CostFactor, TransformationCost, StrategyInteraction

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --------------------------
# Helper functions
# --------------------------

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def fetch_csv_from_s3(s3_resource, bucket_name, key):
    obj = s3_resource.Object(bucket_name, key)
    content = obj.get()['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(content))

def run_decomposition(
        EMISSION_TARGETS_CSV_PATH,
        R_SCRIPT_PATH,
        TMP_DIR_PATH,
        TIME_PERIOD_REF,
        PRIMARY_ID_TO_DECOMPOSE
):
    # Load emission targets
    emission_targets_df = pd.read_csv(EMISSION_TARGETS_CSV_PATH)
    cols_needed = ["Subsector", "Gas", "Vars", "Edgar_Class", TARGET_COUNTRY]
    missing_cols = set(cols_needed) - set(emission_targets_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in emission targets: {missing_cols}")

    emission_targets_df = emission_targets_df[cols_needed].copy()
    emission_targets_df = emission_targets_df.rename(columns={TARGET_COUNTRY: "tvalue"})
    emission_targets_df

    # Convert emission_targets_df to R data.frame
    with localconverter(default_converter + pandas2ri.converter):
        r_te_all = ro.conversion.py2rpy(emission_targets_df)

    logger.info(f"Sourcing R script: {R_SCRIPT_PATH}")
    ro.r['source'](R_SCRIPT_PATH)
    r_rescale = ro.globalenv['rescale']  # function defined in R script

    # Run decomposition steps

    # 1) Filter to the single primary_id
    data_all = output_df.loc[output_df["primary_id"] == PRIMARY_ID_TO_DECOMPOSE].copy()
    if data_all.empty:
        raise ValueError(f"No rows found for primary_id={PRIMARY_ID_TO_DECOMPOSE}")

    # 2) Basic hygiene
    data_all = data_all.fillna(0)

    # 3) Validate required columns
    for col in ("region", "primary_id", "time_period"):
        if col not in data_all.columns:
            raise ValueError(f"'{col}' column missing in df")

    # 4) Sanity: ensure exactly one primary_id
    pids = data_all["primary_id"].dropna().unique().tolist()
    if len(pids) != 1 or pids[0] != PRIMARY_ID_TO_DECOMPOSE:
        raise ValueError(f"Expected exactly one primary_id={PRIMARY_ID_TO_DECOMPOSE}, got {pids}")

    # 5) Regions for this primary_id
    rall = data_all["region"].dropna().astype(str).unique().tolist()
    if not rall:
        raise ValueError("No regions found after filtering; check 'region' data.")

    # 6) Time filter
    before = len(data_all)
    data_all = data_all.loc[data_all["time_period"] >= TIME_PERIOD_REF].copy()
    after = len(data_all)
    logger.info(f"[primary_id={PRIMARY_ID_TO_DECOMPOSE}] time_period >= {TIME_PERIOD_REF}: {before} -> {after} rows")
    if data_all.empty:
        raise ValueError(f"All rows filtered out for primary_id={PRIMARY_ID_TO_DECOMPOSE} with time_period_ref={TIME_PERIOD_REF}")

    # 7) sanitize to avoid rpy2 mixed-type issues
    def sanitize_for_r(df):
        df = df.copy()
        def _is_scalar(x):
            import pandas as pd
            return not isinstance(x, (list, dict, pd.Series))
        for c in df.columns:
            if not df[c].map(_is_scalar).all():
                df[c] = df[c].astype(str)
            elif df[c].dtype == "object":
                df[c] = df[c].astype("string")
        return df

    data_all = sanitize_for_r(data_all)
    emission_targets_df   = sanitize_for_r(emission_targets_df)  # assuming emission_targets_df already prepared earlier

    # 8) Convert to R (use localconverter, not pandas2ri.activate())

    with localconverter(default_converter + pandas2ri.converter):
        r_data_all = ro.conversion.py2rpy(data_all)
    with localconverter(default_converter + pandas2ri.converter):
        r_te_all   = ro.conversion.py2rpy(emission_targets_df)

    # 9) Prepare R args
    r_rall       = ro.StrVector([str(x) for x in rall])            # usually length 1
    r_init_ids   = ro.StrVector([str(PRIMARY_ID_TO_DECOMPOSE)])                 # exactly one id
    # add a / to TMP_DIR_PATH if missing
    if not TMP_DIR_PATH.endswith('/'):
        TMP_DIR_PATH += '/'
    r_dir_output = ro.StrVector([TMP_DIR_PATH])               # local output dir
    r_run        = ro.IntVector([int(PRIMARY_ID_TO_DECOMPOSE)])                 # use primary_id as "run" tag
    r_z          = ro.IntVector([1])
    r_time_ref   = ro.IntVector([TIME_PERIOD_REF])

    logger.info(f"Calling R::rescale(...) for primary_id={PRIMARY_ID_TO_DECOMPOSE} (regions={rall})")
    _ = r_rescale(r_z, r_rall, r_data_all, r_te_all, r_init_ids, r_dir_output, r_time_ref, r_run)
    logger.info("Finished R::rescale() for single primary_id.")

    return None

def upload_df_to_s3(df, s3_resource, bucket, key):
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    s3_resource.Object(bucket, key).put(Body=buffer.getvalue(), ContentType="text/csv")
    logger.info(f"Uploaded to s3://{bucket}/{key}")

def run_cba(primary_id_compare, 
            att_primary, 
            att_strategy, 
            BASE_DECOMPOSED_FILE_PATH, 
            compare_decomposed_df,
            CB_CONFIG_FILE_PATH, 
            PRIMARY_ID_BASE=0):

    # Skip if comparing to itself
    if primary_id_compare == PRIMARY_ID_BASE:
        logger.info(f"Skipping primary_id {primary_id_compare} as it is the baseline.")
        return

    # Set baseline row in attribute primary
    att_primary_copy = att_primary.copy()
    att_primary_copy.iloc[0] = [0, 0, 0, 0] # set first row to baseline values

    # Get the future id if the primary id we're comparing to
    future_id = att_primary.loc[att_primary["primary_id"] == primary_id_compare, "future_id"].values[0]

    # Get the strategy id and code for the primary id we're comparing to
    strategy_id = att_primary.loc[att_primary["primary_id"] == primary_id_compare, "strategy_id"].values[0]
    strategy_code = att_strategy[att_strategy["strategy_id"] == strategy_id]["strategy_code"].values[0]

    logger.info(
        f"\n--- CBA Computation ---\n"
        f"Primary ID      : {primary_id_compare}\n"
        f"Future ID       : {future_id}\n"
        f"Strategy ID     : {strategy_id}\n"
        f"Strategy Code   : {strategy_code}\n"
        f"Base Decomposed : {BASE_DECOMPOSED_FILE_PATH}\n"
        f"-----------------------\n"
    )

    
    # Check if base decomposed file exists (this is the one belonging to the base primary id)
    if not os.path.exists(BASE_DECOMPOSED_FILE_PATH):
        raise ValueError(f"File {BASE_DECOMPOSED_FILE_PATH} does not exist. Skipping.")

    # Load decomposed dfs
    base_decomposed_df = pd.read_csv(BASE_DECOMPOSED_FILE_PATH)

    # # Print shapes
    # logger.info(f"Loaded base decomposed data with shape {base_decomposed_df.shape}")
    # logger.info(f"Loaded compare decomposed data with shape {compare_decomposed_df.shape}")

    # Set up dataframe that goes into CBA
    ssp_data = pd.concat([base_decomposed_df, compare_decomposed_df]).reset_index(drop=True)
    # logger.info(f"Loaded decomposed data with shape {ssp_data.shape}")
    ssp_data["primary_id"] = ssp_data["primary_id"].replace({PRIMARY_ID_BASE: 0})
    ssp_data = ssp_data.replace(np.nan, 0.0)
    strategy_code_base = "BASE"

    # Run CBA
    cb = CostBenefits(ssp_data, att_primary_copy, att_strategy, strategy_code_base)
    cb.ssp_data["future_id"] = 0

    cb.load_cb_parameters(CB_CONFIG_FILE_PATH)

    results_system = cb.compute_system_cost_for_strategy(strategy_code_tx=strategy_code) #NOTE: change this as needed
    results_tx = cb.compute_technical_cost_for_strategy(strategy_code_tx=strategy_code) #NOTE: change this as needed
    results_all = pd.concat([results_system, results_tx], ignore_index=True)

    # logger.info(f"Computed raw CBA results with shape {results_all.shape}")

    results_all_pp = cb.cb_process_interactions(results_all)
    results_all_pp_shifted = cb.cb_shift_costs(results_all_pp)

    results_all_pp_shifted["primary_id"] = primary_id_compare
    results_all_pp_shifted["future_id"] = future_id

    return results_all_pp_shifted


def postprocess_cba(
        cb_raw_df: pd.DataFrame
):
    # --- split the 'variable' column into parts ---
    # R made 5 columns: name, sector, cb_type, item_1, item_2
    # Use n=4 so we get at most 5 pieces even if extra ':' appear later.
    parts = cb_raw_df["variable"].astype(str).str.split(":", n=4, expand=True)
    parts.columns = ["name", "sector", "cb_type", "item_1", "item_2"]

    # append parts to cb_raw_df
    cb_data = pd.concat([cb_raw_df, parts], axis=1)

    # --- scaling and year ---
    cb_data["value"] = cb_data["value"] / 1e9
    cb_data["Year"]  = cb_data["time_period"] + 2015

    # --- aggregate (sum, skipping NaNs as in na.rm=TRUE) ---
    group_cols = ["cb_type", "strategy_code", "primary_id", "future_id", "Year"]
    cb_agg = (
        cb_data
        .groupby(group_cols, dropna=False, as_index=False)["value"]
        .sum()
        .rename(columns={"value": "Cumulative"})
    )

    # --- aggregated format (dcast) ---
    agg_cb_df = (
        cb_agg
        .pivot_table(
            index=["primary_id", "future_id", "strategy_code", "Year"],
            columns="cb_type",
            values="Cumulative",
            aggfunc="sum"        # safe even if duplicates appear
            # , fill_value=0     # uncomment if you prefer 0 instead of NaN
        )
        .reset_index()
    )

    # If you prefer flat columns after pivot (remove the name from columns):
    agg_cb_df.columns.name = None

    # Keep only relevant columns
    agg_cb_df = agg_cb_df[["primary_id","future_id", "strategy_code", "Year","air_pollution", "technical_cost"]]
    
    return agg_cb_df

# --------------------------
# Setup initial paths
# --------------------------
# SCRIPT_DIR_PATH = os.getcwd()
SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "config")
CW_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "cw")
TMP_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "tmp")
R_SCRIPTS_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "r_scripts")

os.makedirs(TMP_DIR_PATH, exist_ok=True)


# --------------------------
# AWS S3 connection setup
# --------------------------
aws_config = read_yaml(os.path.join(CONFIG_DIR_PATH, "aws_credentials_config.yaml"))
PROFILE_NAME = aws_config["profile_name"]
BUCKET_NAME = aws_config["bucket_name"]

# Set your AWS profile
session = boto3.Session(profile_name=PROFILE_NAME)

# Create S3 resource
S3_RESOURCE = session.resource('s3')

# --------------------------
# Fetch data from S3
# --------------------------

# Define run ID and prefix
RUN_ID = "sisepuede_run_2025-10-07t13;30;14.193421" #NOTE: Change this depending on the run you want to postprocess
# DIR_ID = 0 # Use this for testing
DIR_ID = sys.argv[1]  #NOTE: Get DIR_ID from command line argument

RUN_DB_PREFIX = f'run_database/{RUN_ID}/'
MODEL_OUTPUT_PREFIX = f'{RUN_DB_PREFIX}model_output/region=louisiana/model_output_{DIR_ID}/'
MODEL_INPUT_PREFIX = f'{RUN_DB_PREFIX}model_input/region=louisiana/model_input_{DIR_ID}/'
TRANSFER_PREFIX = f"transfers/{RUN_ID}/"

output_df = fetch_csv_from_s3(S3_RESOURCE, BUCKET_NAME, f'{MODEL_OUTPUT_PREFIX}data.csv')
input_df = fetch_csv_from_s3(S3_RESOURCE, BUCKET_NAME, f'{MODEL_INPUT_PREFIX}data.csv')
attribute_primary_df = fetch_csv_from_s3(S3_RESOURCE, BUCKET_NAME, f'{TRANSFER_PREFIX}ATTRIBUTE_PRIMARY.csv')
attribute_strategy_df = fetch_csv_from_s3(S3_RESOURCE, BUCKET_NAME, f'{TRANSFER_PREFIX}ATTRIBUTE_STRATEGY.csv')

# Get the unique primary_ids from the output data
primary_ids = output_df["primary_id"].dropna().unique().tolist()
logger.info(f"Found {len(primary_ids)} unique primary_id values in output data.")

# Compare with input data
input_primary_ids = input_df["primary_id"].dropna().unique().tolist()
missing_in_input = set(primary_ids) - set(input_primary_ids)
if missing_in_input:
    logger.warning(f"Warning: {len(missing_in_input)} primary_id values in output not found in input data: {missing_in_input}")
else:
    logger.info("All primary_id values in output data are present in input data.")

# Check that this primary ids are present in attribute_primary_df
attribute_primary_ids = attribute_primary_df["primary_id"].dropna().unique().tolist()
missing_in_attribute = set(primary_ids) - set(attribute_primary_ids)
if missing_in_attribute:
    logger.warning(f"Warning: {len(missing_in_attribute)} primary_id values in output not found in attribute_primary_df: {missing_in_attribute}")
else:
    logger.info("All primary_id values in output data are present in attribute_primary_df.")

# --------------------------
# Decomposition
# --------------------------

# Set up paths and parameters
TARGET_COUNTRY = "LA" #NOTE: Change this if you're working with a different region
EMISSION_TARGETS_CSV_PATH = os.path.join(CW_DIR_PATH, 'emission_targets_LA_2021.csv') #NOTE: Change this if you're working with a different file
R_SCRIPT_PATH = os.path.join(R_SCRIPTS_DIR_PATH, 'intertemporal_function_baseline_mapping_timeref.r')
TIME_PERIOD_REF = 7 #NOTE: Change this if you're using a different reference year
S3_DECOMPOSED_DIR_PREFIX = f"{RUN_DB_PREFIX}decomposed_outputs/" #NOTE: Change this if you're using a different S3 prefix

# Run decomposition for each primary_id
for primary_id_to_decompose in primary_ids:

    run_decomposition(
            EMISSION_TARGETS_CSV_PATH,
            R_SCRIPT_PATH,
            TMP_DIR_PATH,
            TIME_PERIOD_REF,
            primary_id_to_decompose
    )

    # Fetch PRIMARY_ID_BASE from attribute_primary_df where strategy_id == 0
    base_rows = attribute_primary_df[attribute_primary_df["strategy_id"] == 0]
    if base_rows.empty:
        raise ValueError("No primary_id found in attribute_primary_df with strategy_id == 0")
    
    PRIMARY_ID_BASE = base_rows.iloc[0]["primary_id"]
    logger.info(f"Using PRIMARY_ID_BASE = {PRIMARY_ID_BASE} for cost-benefit analysis.")

    # Load the decomposed data
    decomposed_df = pd.read_csv(os.path.join(TMP_DIR_PATH, f"louisiana_{primary_id_to_decompose}.csv"))
    logger.info(f"Decomposed data for primary_id={primary_id_to_decompose} has {len(decomposed_df)} rows and {len(decomposed_df.columns)} columns.")
    
    # We need to merge inputs since we need them for cost-benefit calculations
    decomposed_df_merged = pd.merge(decomposed_df, input_df, on=["primary_id", "region", "time_period"], how="left")

    # if primary_id_to_decompose is the base, save the merged file as well for future reference
    if primary_id_to_decompose == PRIMARY_ID_BASE:
        decomposed_df_merged.to_csv(os.path.join(TMP_DIR_PATH, f"louisiana_{primary_id_to_decompose}.csv"), index=False)

    # Calculate total emissions for upload
    decomposed_df['total_emissions'] = decomposed_df[[col for col in decomposed_df.columns if col.startswith("emission_co2e_subsector_total")]].sum(axis=1)
    df_to_upload = decomposed_df[["primary_id", "time_period", "total_emissions"]]

    # Upload decomposed data to S3
    s3_key = f"{S3_DECOMPOSED_DIR_PREFIX}emission_total_{primary_id_to_decompose}.csv"
    upload_df_to_s3(df_to_upload, S3_RESOURCE, BUCKET_NAME, s3_key)


    # --------------------------
    # Cost-benefit analysis
    # --------------------------

    BASE_DECOMPOSED_FILE_PATH = os.path.join(TMP_DIR_PATH, f"louisiana_{PRIMARY_ID_BASE}.csv")
    CB_CONFIG_FILE_PATH = os.path.join(CONFIG_DIR_PATH, "cb_config_params.xlsx")

     # Check if file exists
    if not os.path.exists(BASE_DECOMPOSED_FILE_PATH):
        raise ValueError(f"File {BASE_DECOMPOSED_FILE_PATH} does not exist, please make sure to have one.")

    cb_raw_df = run_cba(
        primary_id_compare=primary_id_to_decompose,
        att_primary=attribute_primary_df,
        att_strategy=attribute_strategy_df,
        BASE_DECOMPOSED_FILE_PATH=BASE_DECOMPOSED_FILE_PATH,
        compare_decomposed_df=decomposed_df_merged,
        CB_CONFIG_FILE_PATH=CB_CONFIG_FILE_PATH,
        PRIMARY_ID_BASE=PRIMARY_ID_BASE
    )

    # check if it's not empty
    if cb_raw_df is None or cb_raw_df.empty:
        logger.warning(f"No CBA results for primary_id {primary_id_to_decompose}. Skipping upload.")
        continue

    # postprocess and upload to S3
    agg_cb_df = postprocess_cba(cb_raw_df)

    # check if it's not empty
    if agg_cb_df is None or agg_cb_df.empty:
        logger.warning(f"No postprocessed CBA results for primary_id {primary_id_to_decompose}. Skipping upload.")
        continue

    S3_CB_DIR_PREFIX = f"{RUN_DB_PREFIX}cb_outputs/"
    s3_key = f"{S3_CB_DIR_PREFIX}cb_{primary_id_to_decompose}.csv"
    upload_df_to_s3(agg_cb_df, S3_RESOURCE, BUCKET_NAME, s3_key)

    # --------------------------
    # Eliminate local file for current primary_id
    # --------------------------
    fname = f"louisiana_{primary_id_to_decompose}.csv"
    file_path = os.path.join(TMP_DIR_PATH, fname)
    if os.path.exists(file_path) and primary_id_to_decompose != PRIMARY_ID_BASE:
        try:
            os.remove(file_path)
            logger.info(f"Deleted: {fname}")
        except Exception as e:
            logger.warning(f"Could not delete {fname}: {e}")