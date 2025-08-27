#NOTE: Use this script when having an experiment with different future_ids and the same strategy_id.

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
build_path = lambda PATH  : os.path.abspath(os.path.join(*PATH))
CB_DEFAULT_DEFINITION_PATH = build_path([SCRIPT_DIR_PATH, "cb_cost_factors"])
OUTPUT_CB_PATH = build_path([SCRIPT_DIR_PATH, "cb_results"])
data_id = "2025-08-17t23;06;14.607675"
OUTPUT_LOUSIANA_CB_PATH = build_path([OUTPUT_CB_PATH, data_id])
RUN_DIR_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR_PATH), 
    "data", 
    "ensemble_data", 
    f"sisepuede_summary_results_run_sisepuede_run_{data_id}"
)

# Make sure output directory exists
os.makedirs(OUTPUT_CB_PATH, exist_ok=True)
os.makedirs(OUTPUT_LOUSIANA_CB_PATH, exist_ok=True)

## Load the data
#ssp_data = pd.read_csv(os.path.join(SSP_RESULTS_PATH, "louisiana.csv"))
att_primary = pd.read_csv(os.path.join(RUN_DIR_PATH, "ATTRIBUTE_PRIMARY.csv"))
att_strategy = pd.read_csv(os.path.join(RUN_DIR_PATH, "ATTRIBUTE_STRATEGY.csv"))

## Subset ssp data
"""
We assume that primary_id 354354, which corresponds to future_id 0, is the baseline execution.

We will modify ATTRIBUTE_PRIMARY.csv, which has the following format:
    primary_id  design_id  strategy_id  future_id
0         354354          4         6004          0

To the format:

   primary_id  design_id  strategy_id  future_id
0           0          0            0          0

"""
future_id_compare = 0
att_primary.iloc[0] = [0,0,0,0]

future_id_cli_arg = int(sys.argv[1])
#future_id_cli_arg = int(1)
strategy_id = 6004
future_id_compare = att_primary.query(f"strategy_id=={strategy_id} and future_id=={future_id_cli_arg}").primary_id.values[0]

future_id_base = 394394

primary_ids = [future_id_base, future_id_compare]


q = (
    pl.scan_csv(os.path.join(RUN_DIR_PATH, f"sisepuede_results_IDE_{data_id}.csv"), ignore_errors=True)
    .filter(
        pl.col('primary_id').is_in(primary_ids)
    )
)

ssp_data = q.collect().to_pandas()
primary_id_base = 394394

ssp_data["primary_id"] = ssp_data["primary_id"].replace({ 394394 : 0})
ssp_data = ssp_data.replace(np.nan, 0.0)

## Define base strategy
strategy_code_base = "BASE"

## Instantiate an object of the CostBenefits class
cb = CostBenefits(ssp_data, att_primary, att_strategy, strategy_code_base)

cb.ssp_data["future_id"] = 0

# Once the excel file has been updated, we can reload it in order to update the cost factors database
cb.load_cb_parameters(os.path.join(CB_DEFAULT_DEFINITION_PATH, "cb_config_params.xlsx"))

# Compute System Costs
results_system = cb.compute_system_cost_for_strategy(strategy_code_tx='PFLO:ALL_LA_ACTIONS')

# Compute Technical Costs
results_tx = cb.compute_technical_cost_for_strategy(strategy_code_tx="PFLO:ALL_LA_ACTIONS")

# Combine results
results_all = pd.concat([results_system, results_tx], ignore_index = True)

#-------------POST PROCESS SIMULATION RESULTS---------------
# Post process interactions among strategies that affect the same variables
results_all_pp = cb.cb_process_interactions(results_all)

# SHIFT any stray costs incurred from 2015 to 2025 to 2025 and 2035
results_all_pp_shifted = cb.cb_shift_costs(results_all_pp)

results_all_pp_shifted["primary_id"] = future_id_compare
results_all_pp_shifted["future_id"] = future_id_cli_arg

results_all_pp_shifted.to_csv(os.path.join(OUTPUT_LOUSIANA_CB_PATH, f"cba_la_{future_id_cli_arg}.csv"), index = False)

