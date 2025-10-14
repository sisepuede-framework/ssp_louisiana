# =============================================================================
# Script Name:     03-Analysis.R
# Purpose:         Load data files
# Author:          Nikkolas Monceaux
# Date Created:    2025-09-02
# Last Modified:   2025-09-05 by Nikkolas Monceaux
#
# Inputs:          input
# Outputs:         output
#
# Dependencies:    
# Notes:           
# =============================================================================



# ----- Parameters defined -----
parameters <- c(utility_wacc = .07, utility_plant_life = 40)

# ----- Data Load ----- 
shocks <- list()
shocks$power    <- readRDS(paste0("100k_runs_postprocessing/tmp/power_shocks_", primary_id, ".rds"))
shocks$industry <- readRDS(paste0("100k_runs_postprocessing/tmp/ind_shocks_", primary_id, ".rds"))
shocks$ccs      <- readRDS(paste0("100k_runs_postprocessing/tmp/ccs_shocks_", primary_id, ".rds"))
load("100k_runs_postprocessing/r_scripts/lsu_files/clean_data.rda")
  
# ----- Economic Impacts -----
primaries <- unique(shocks$power$primary_id[shocks$power$primary_id != 0]) #primary_id represents scenarios, 0 is baseline

baseline <- list()
baseline$power_shocks <- shocks$power  %>%
  filter(primary_id == 0)
baseline$power_impact <- leim_calc(baseline$power_shocks, shock="total_shock", time="time_period") %>%
  collapse_totals_time() 

baseline$industry_shocks <- shocks$industry  %>%
  filter(primary_id == 0)
baseline$industry_impact <- leim_calc(baseline$industry_shocks, shock="total_shock", time="time_period", la_rps="la_rps") %>%
  collapse_totals_time() 

baseline$ccs_shocks <- shocks$ccs  %>%
  filter(primary_id == 0)
#Impact is 0. No effect in baseline.

power_impact <- list()
industry_impact <- list()
ccs_impact <- list()
power_shock <- list()
for(id in primaries) {
  
  if (id %in% shocks$power$primary_id) {
  power_impact[[as.character(id)]] <- shocks$power %>%
    filter(primary_id == id) %>%
    leim_calc(shock="total_shock", time="time_period") %>%
    collapse_totals_time()
  }
  
  if (id %in% shocks$industry$primary_id) {
  industry_impact[[as.character(id)]] <- shocks$industry %>%
    filter(primary_id == id) %>%
    leim_calc(shock="total_shock", time="time_period", la_rps="la_rps") %>%
    collapse_totals_time()
  }
  
  if (id %in% shocks$ccs$primary_id) {
    ccs_impact[[as.character(id)]] <- shocks$ccs %>%
      filter(primary_id == id) %>%
      leim_calc(shock="ccs_shock", time="time_period") %>%
      collapse_totals_time()
  }
}

power_diff <- list()
industry_diff <- list()
ccs_diff <- list()
for(id in primaries) {
  #id <-primaries[1]
  power_diff[[as.character(id)]] <- as.data.frame(power_impact[[as.character(id)]] - baseline$power_impact[]) %>%
    mutate(time=baseline$power_impact$time[])
  
  industry_diff[[as.character(id)]] <- as.data.frame(industry_impact[[as.character(id)]] - baseline$industry_impact[]) %>%
    mutate(time=baseline$industry_impact$time[])
  
  if (id %in% names(ccs_impact)) {
    ccs_diff[[as.character(id)]] <- as.data.frame(ccs_impact[[as.character(id)]]) %>%
      mutate(time=ccs_impact[[as.character(id)]][["time"]])
  }
}

# ----- Data Save -----
saveRDS(baseline,         paste0("100k_runs_postprocessing/tmp/cleanbaseline_", primary_id, ".rds"))
saveRDS(power_impact,     paste0("100k_runs_postprocessing/tmp/cleanpower_impact_", primary_id, ".rds"))
saveRDS(industry_impact,  paste0("100k_runs_postprocessing/tmp/cleanindustry_impact_", primary_id, ".rds"))
saveRDS(ccs_impact,       paste0("100k_runs_postprocessing/tmp/cleanccs_impact_", primary_id, ".rds"))
saveRDS(power_diff,       paste0("100k_runs_postprocessing/tmp/cleanpower_diff_", primary_id, ".rds"))
saveRDS(industry_diff,    paste0("100k_runs_postprocessing/tmp/cleanindustry_diff_", primary_id, ".rds"))
saveRDS(ccs_diff,         paste0("100k_runs_postprocessing/tmp/cleanccs_diff_", primary_id, ".rds"))
saveRDS(primaries,        paste0("100k_runs_postprocessing/tmp/primaries_", primary_id, ".rds"))