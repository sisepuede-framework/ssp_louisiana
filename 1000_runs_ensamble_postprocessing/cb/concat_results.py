import os
import pandas as pd

# --- folder path ---
##---- Define Directories ----##
SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
build_path = lambda PATH  : os.path.abspath(os.path.join(*PATH))
CB_DEFAULT_DEFINITION_PATH = build_path([SCRIPT_DIR_PATH, "cb_cost_factors"])
OUTPUT_CB_PATH = build_path([SCRIPT_DIR_PATH, "cb_results"])
data_id = "2025-08-28t15;29;22.344855"
RUN_OUTPUT_CB_PATH = build_path([OUTPUT_CB_PATH, data_id])
OUTPUT_LOUSIANA_CB_PATH = build_path([OUTPUT_CB_PATH, data_id])
RUN_DIR_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR_PATH), 
    "data", 
    "ensemble_data", 
    f"sisepuede_summary_results_run_sisepuede_run_{data_id}"
)

# --- find all CSV files in the folder ---
csv_files = [f for f in os.listdir(RUN_OUTPUT_CB_PATH) if f.lower().endswith(".csv")]
print(f"Found {len(csv_files)} CSV files in {RUN_OUTPUT_CB_PATH}")

# --- read and concatenate ---
df_list = []
for file in csv_files:
    file_path = os.path.join(RUN_OUTPUT_CB_PATH, file)
    df = pd.read_csv(file_path)
    df_list.append(df)
    print(f"Loaded {file} with shape {df.shape}")

# Combine into one DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

# Optional: save to a single CSV
output_file = os.path.join(RUN_DIR_PATH, f"combined_cb_results_{data_id}.csv")
print(f"Shape of combined DataFrame: {combined_df.shape}")
print(f"Null values in the combined DataFrame: {combined_df.isnull().sum()}")
combined_df.to_csv(output_file, index=False)
print(f"Combined {len(csv_files)} CSV files into: {output_file}")
