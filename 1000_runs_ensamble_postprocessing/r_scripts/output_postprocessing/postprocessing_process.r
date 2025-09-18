#################################################
# Post processing process
#################################################

# load packages
library(data.table)
library(reshape2)
library(mFilter)
library(ggplot2)

rm(list=ls())

#ouputfile
dir.output  <- "1000_runs_ensamble_postprocessing/ssp_output/sisepuede_summary_results_run_sisepuede_run_2025-09-17T01;49;45.687668/"
output.file <- "WIDE_INPUTS_OUTPUTS.csv"

region <- "louisiana" 
iso_code3 <- "LA"

year_ref <- 2021


source('1000_runs_ensamble_postprocessing/r_scripts/output_postprocessing/scr/run_script_baseline_run_new.r')

source('1000_runs_ensamble_postprocessing/r_scripts/output_postprocessing/scr/data_prep_new_mapping_louisiana.r')

#source('1000_runs_ensamble_postprocessing/r_scripts/output_postprocessing/scr/data_prep_drivers.r')
