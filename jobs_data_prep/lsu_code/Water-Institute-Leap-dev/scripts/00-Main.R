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
#library(ggplot2)
#library(CESgraphics)

source("R/99_functions.R")
source("R/input.R")
source("R/Econ_Impact.R")
source("R/output.R")
source("R/ces_theme.R")


source("scripts/01-Data_Clean.R")
source("scripts/02-Analysis.R")
source("scripts/LSU leap post processing")
#source("scripts/03-01-Output.R")
#source("scripts/03-02-Output.R")