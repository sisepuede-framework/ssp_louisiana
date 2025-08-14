# NOTE: This script is used for a standard CB analysis with a fixed future_id and different strategy_ids.

## Load packages
from costs_benefits_ssp.cb_calculate import CostBenefits
import numpy as np
import pandas as pd 
import sys
import os 

from costs_benefits_ssp.model.cb_data_model import TXTable,CostFactor,TransformationCost,StrategyInteraction

import polars as pl

##---- Define Directories ----##
SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
DATA_DIR_PATH = os.path.join(PARENT_DIR_PATH, "data")
ENSEMBLE_DIR_PATH = os.path.join(DATA_DIR_PATH, "ensemble_data")
build_path = lambda PATH  : os.path.abspath(os.path.join(*PATH))
POST_PROCESSED_DIR_PATH = build_path([PARENT_DIR_PATH,"output"])
CB_DEFAULT_DEFINITION_PATH = build_path([SCRIPT_DIR_PATH, "cb_cost_factors"])
OUTPUT_CB_PATH = build_path([SCRIPT_DIR_PATH, "cb_results"])
data_id = "2025-08-13T02;26;01.977344"
OUTPUT_LOUSIANA_CB_PATH = build_path([OUTPUT_CB_PATH, data_id])
RUN_RAW_DATA_DIR_PATH = os.path.join(ENSEMBLE_DIR_PATH, f"sisepuede_summary_results_run_sisepuede_run_{data_id}")


# Make sure output directory exists
os.makedirs(OUTPUT_CB_PATH, exist_ok=True)
os.makedirs(OUTPUT_LOUSIANA_CB_PATH, exist_ok=True)

## Load the data
ssp_data = pd.read_csv(os.path.join(RUN_RAW_DATA_DIR_PATH, f"sisepuede_results_IDE_{data_id}.csv"))
att_primary = pd.read_csv(os.path.join(RUN_RAW_DATA_DIR_PATH, "ATTRIBUTE_PRIMARY.csv"))
att_strategy = pd.read_csv(os.path.join(RUN_RAW_DATA_DIR_PATH, "ATTRIBUTE_STRATEGY.csv"))

## Define base strategy
strategy_code_base = "BASE"

## Instantiate an object of the CostBenefits class
cb = CostBenefits(ssp_data, att_primary, att_strategy, strategy_code_base)

# Once the excel file has been updated, we can reload it to update the cost factors database
cb.load_cb_parameters(os.path.join(CB_DEFAULT_DEFINITION_PATH, "cb_config_params.xlsx"))

# Compute System Costs
results_system = cb.compute_system_cost_for_all_strategies(verbose=False)

# Compute Technical Costs
results_tx = cb.compute_technical_cost_for_all_strategies(verbose=False)

# Combine results
results_all = pd.concat([results_system, results_tx], ignore_index = True)

#-------------POST PROCESS SIMULATION RESULTS---------------
# Post process interactions among strategies that affect the same variables
results_all_pp = cb.cb_process_interactions(results_all)

# SHIFT any stray costs incurred from 2015 to 2025 to 2025 and 2035
results_all_pp_shifted = cb.cb_shift_costs(results_all_pp)

results_all_pp_shifted.to_csv(os.path.join(RUN_RAW_DATA_DIR_PATH, f"cba_la.csv"), index = False)
