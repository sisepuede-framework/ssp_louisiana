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
shocks$power <- readRDS("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/intermediate/power_shocks.rds")
shocks$industry <- readRDS("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/intermediate/ind_shocks.rds")
shocks$ccs <- readRDS("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/intermediate/ccs_shocks.rds")
load("jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/raw/clean_data.rda")
  
# ----- Economic Impacts -----
primaries <- unique(shocks$power$primary_id[shocks$power$primary_id != 36800368]) #primary_id represents scenarios, 0 is baseline

baseline <- list()
baseline$power_shocks <- shocks$power  %>%
  filter(primary_id == 36800368)
baseline$power_impact <- leim_calc(baseline$power_shocks, shock="total_shock", time="time_period") %>%
  collapse_totals_time() 

baseline$industry_shocks <- shocks$industry  %>%
  filter(primary_id == 36800368)
baseline$industry_impact <- leim_calc(baseline$industry_shocks, shock="total_shock", time="time_period", la_rps="la_rps") %>%
  collapse_totals_time() 

baseline$ccs_shocks <- shocks$ccs  %>%
  filter(primary_id == 36800368)
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
saveRDS(baseline, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/cleanbaseline.rds")
saveRDS(power_impact, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/cleanpower_impact.rds")
saveRDS(industry_impact, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/cleanindustry_impact.rds")
saveRDS(ccs_impact, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/cleanccs_impact.rds")
saveRDS(power_diff, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/cleanpower_diff.rds")
saveRDS(industry_diff, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/cleanindustry_diff.rds")
saveRDS(ccs_diff, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/cleanccs_diff.rds")
saveRDS(primaries, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/intermediate/primaries.rds")