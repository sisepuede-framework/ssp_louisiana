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

PRIMARY_ID_TO_DECOMPOSE = 43800438

primary_id   <- PRIMARY_ID_TO_DECOMPOSE

source("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/R/99_functions.R")
source("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/R/input.R")
source("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/R/Econ_Impact.R")
source("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/R/output.R")
source("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/R/ces_theme.R")


source("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/scripts/01-Data_Clean.R")
source("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/scripts/02-Analysis.R")
source("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/scripts/LSU leap post processing.R")
#source("scripts/03-01-Output.R")
#source("scripts/03-02-Output.R")