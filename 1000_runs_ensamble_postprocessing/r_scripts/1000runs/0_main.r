#################################################
# Post processing process
#################################################

# load packages
library(data.table)
library(parallel)


rm(list=ls())


run <- "1000_runs_ensamble_postprocessing/ssp_output/sisepuede_run_2025-11-16T10;43;25.118223_PARETO5.11/"
file.name <- "sisepuede_runs_20251119_5.11_future_1_pareto_with_baselines.csv"

source('1000_runs_ensamble_postprocessing/r_scripts/1000runs/parse_experiment_in_individual_files.r')

source('1000_runs_ensamble_postprocessing/r_scripts/run_script_baseline_run_new_1000ensamble.r')