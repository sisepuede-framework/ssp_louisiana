import pandas as pd
import os

# Set root directory and file name
SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "data")
ENSEMBLE_DIR_PATH = os.path.join(DATA_DIR_PATH, "ensemble_raw_data")

# Ensure the data directory exists
os.makedirs(DATA_DIR_PATH, exist_ok=True)
os.makedirs(ENSEMBLE_DIR_PATH, exist_ok=True)

file_name = "de38bb46-7d00-4cf5-8844-f6cc20695024.csv"
full_path = os.path.join(ENSEMBLE_DIR_PATH, file_name)

# Read the big CSV
full_sim = pd.read_csv(full_path)

# Get unique primary_ids
all_ids = full_sim["primary_id"].unique()

# Ensure output folder exists
OUTPUT_DIR_PATH = os.path.join(DATA_DIR_PATH, "parsed_runs")
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

# Create a directory with the same name as the file
file_base_name = os.path.splitext(file_name)[0]
OUTPUT_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, file_base_name)
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

# Split and save
for i, pid in enumerate(all_ids, 1):   # start numbering at 1
    pivot = full_sim[full_sim["primary_id"] == pid]
    out_path = os.path.join(OUTPUT_DIR_PATH, f"{i}.csv")
    pivot.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
