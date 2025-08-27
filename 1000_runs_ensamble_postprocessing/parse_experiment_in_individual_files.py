# First step of decomposition pipeline
# This script parses a large CSV file containing simulation data and splits it into individual files based on unique primary IDs.
# Each individual file is saved in a directory named after the run ID.
# It ensures that the necessary directories exist and handles the creation of output files.
# after this, you can run the decomposition notebook

import pandas as pd
import os

# Set root directory and file name
SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "data")
ENSEMBLE_DIR_PATH = os.path.join(DATA_DIR_PATH, "ensemble_data")


run_id = "2025-08-27T11;02;10.572056"
RUN_ENSEMBLE_DIR_PATH = os.path.join(ENSEMBLE_DIR_PATH, f"sisepuede_summary_results_run_sisepuede_run_{run_id}")
file_name = "WIDE_INPUTS_OUTPUTS.csv"
full_path = os.path.join(RUN_ENSEMBLE_DIR_PATH, file_name)

# Read the big CSV
full_sim = pd.read_csv(full_path)

# Get unique primary_ids
all_ids = full_sim["primary_id"].unique()

# Ensure output folder exists
OUTPUT_DIR_PATH = os.path.join(DATA_DIR_PATH, "parsed_runs")
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

# Create a directory with the same name as the run_id
OUTPUT_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, run_id)
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
print(f"Output directory created: {OUTPUT_DIR_PATH}")

# Split and save
for i, pid in enumerate(all_ids, 1):   # start numbering at 1
    pivot = full_sim[full_sim["primary_id"] == pid]
    out_path = os.path.join(OUTPUT_DIR_PATH, f"{i}.csv")
    pivot.to_csv(out_path, index=False)
    # print(f"Saved: {out_path}")
    
print(f"Saved {len(all_ids)} files in: {OUTPUT_DIR_PATH}")
