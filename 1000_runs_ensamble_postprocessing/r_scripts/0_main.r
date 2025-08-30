#################################################
# Post processing process
#################################################

# load packages
library(data.table)
library(parallel)


rm(list=ls())


run <- "1000_runs_ensamble_postprocessing/ssp_output/sisepuede_run_2025-08-28t15;29;22.344855/"
file.name <- "9a7fe49a-fef0-4d39-bcd1-d677f91da13d.csv"

source('1000_runs_ensamble_postprocessing/r_scripts/parse_experiment_in_individual_files.r')

source('1000_runs_ensamble_postprocessing/r_scripts/run_script_baseline_run_new_1000ensamble.r')