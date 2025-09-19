#################################################
# Post processing process
#################################################

# load packages
library(data.table)
library(parallel)


rm(list=ls())


run <- "1000_runs_ensamble_postprocessing/ssp_output/sisepuede_run_2025-09-18t09;19;22.726476/"
file.name <- "8fa83d4a-a9a7-481b-a0c2-fa6edae506bd.csv"

source('1000_runs_ensamble_postprocessing/r_scripts/1000runs/parse_experiment_in_individual_files.r')

source('1000_runs_ensamble_postprocessing/r_scripts/run_script_baseline_run_new_1000ensamble.r')