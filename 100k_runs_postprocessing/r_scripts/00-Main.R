# =============================================================================
# Script Name:     03-Analysis.R
# Purpose:         Load data files
# Author:          Nikkolas Monceaux
# Date Created:    2025-10-01
# Last Modified:   2025-10-01 by Nikkolas Monceaux
#
# Inputs:          input
# Outputs:         output
#
# Dependencies:    
# Notes:           
# =============================================================================
# ----- Package load -----
library(here)
library(readr)
library(dplyr)
library(tidyr)
library(readxl)
library(writexl)
library(openxlsx)
#library(ggplot2)
#library(CESgraphics)

PRIMARY_ID_TO_DECOMPOSE = 43800444

primary_id   <- PRIMARY_ID_TO_DECOMPOSE

source("100k_runs_postprocessing/r_scripts/helper/99_functions.R")
source("100k_runs_postprocessing/r_scripts/helper/input.R")
source("100k_runs_postprocessing/r_scripts/helper/Econ_Impact.R")
source("100k_runs_postprocessing/r_scripts/helper/output.R")
source("100k_runs_postprocessing/r_scripts/helper/ces_theme.R")


source("100k_runs_postprocessing/r_scripts/01-Data_Clean.R")
source("100k_runs_postprocessing/r_scripts/02-Analysis.R")
source("100k_runs_postprocessing/r_scripts/LSU-leap-post-processing.R")
#source("scripts/03-01-Output.R")
#source("scripts/03-02-Output.R")