import os
import pandas as pd

# --- folder path ---
data_id = "de38bb46-7d00-4cf5-8844-f6cc20695024"
SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
CB_RESULTS_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, "cb_results")
CB_LA_RESULTS_DIR_PATH = os.path.join(CB_RESULTS_DIR_PATH, data_id)
DECOMPOSITION_OUTPUT_DIR_PATH = os.path.join(PARENT_DIR_PATH, "output")

# --- find all CSV files in the folder ---
csv_files = [f for f in os.listdir(CB_LA_RESULTS_DIR_PATH) if f.lower().endswith(".csv")]

# --- read and concatenate ---
df_list = []
for file in csv_files:
    file_path = os.path.join(CB_LA_RESULTS_DIR_PATH, file)
    df = pd.read_csv(file_path)
    df_list.append(df)
    print(f"Loaded {file} with shape {df.shape}")

# Combine into one DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

# Optional: save to a single CSV
output_file = os.path.join(DECOMPOSITION_OUTPUT_DIR_PATH, f"combined_cb_results_{data_id}.csv")
print(f"Shape of combined DataFrame: {combined_df.shape}")
print(f"Null values in the combined DataFrame: {combined_df.isnull().sum()}")
combined_df.to_csv(output_file, index=False)
print(f"Combined {len(csv_files)} CSV files into: {output_file}")
