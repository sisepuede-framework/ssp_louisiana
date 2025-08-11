import os
import pandas as pd

# --- paths ---
SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_PATH  = os.path.join(SCRIPT_DIR_PATH, "output")
data_id = "de38bb46-7d00-4cf5-8844-f6cc20695024"



in_file = os.path.join(OUTPUT_DIR_PATH, f"combined_cb_results_{data_id}.csv")
out_file = os.path.join(OUTPUT_DIR_PATH, f"wide_cb_data_lhc{data_id}.csv")

# --- load ---
cb_data = pd.read_csv(in_file)

# --- split the 'variable' column into parts ---
# R made 5 columns: name, sector, cb_type, item_1, item_2
# Use n=4 so we get at most 5 pieces even if extra ':' appear later.
parts = cb_data["variable"].astype(str).str.split(":", n=4, expand=True)
parts.columns = ["name", "sector", "cb_type", "item_1", "item_2"]

cb_data = pd.concat([cb_data, parts], axis=1)

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

# --- wide format (dcast) ---
wide_cb = (
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
wide_cb.columns.name = None

# --- save ---
wide_cb.to_csv(out_file, index=False)
print(f"Saved: {out_file}")
