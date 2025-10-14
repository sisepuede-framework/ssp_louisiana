# =============================================================================
# Script Name:     02-Data_Clean.R
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
raw <- list()
parameters <- c(utility_wacc = .07, utility_plant_life = 40)

# ----- Data Load ----- 
raw$sector_alloc <- read_csv("100k_runs_postprocessing/r_scripts/lsu_files/sector_allocation_prep.csv")
raw$energy       <- read_csv(paste0("100k_runs_postprocessing/tmp/baseline_costs_and_production_by_prodtype_", primary_id, ".csv"))
raw$industrial   <- read_csv(paste0("100k_runs_postprocessing/tmp/industrial_energy_cost_", primary_id, ".csv"))
raw$trans_elec   <- read_csv(paste0("100k_runs_postprocessing/tmp/transportation_electric_efficiency_cost_", primary_id, ".csv")) %>%
  mutate(
    electricity_volume_saved_in_MWh	= electricity_volume_saved_in_PJ * 277777.78,
    electricity_volume_saved_value = electricity_volume_saved_in_MWh * 111
  )
raw$trans_heavy  <- read_csv(paste0("100k_runs_postprocessing/tmp/transportation_heavy_duty_fuel_switch_cost_", primary_id, ".csv")) %>%
  rename(fuel_switch_net_cost_heavy = `fuel_switch_net_cost_$`)
raw$trans_non_elec <- read_csv(paste0("100k_runs_postprocessing/tmp/transportation_non_electric_efficiency_cost_", primary_id, ".csv")) %>%
  mutate(
    electricity_volume_saved_in_MWh = transportation_volume_saved_in_PJ * 277777.78,
    electricity_volume_saved_value_non = electricity_volume_saved_in_MWh * 111,
    transportation_efficiency_non = transportation_efficiency_capex	+ transportation_efficiency_opex
  )
raw$trans_rail <- read_csv(paste0("100k_runs_postprocessing/tmp/transportation_rail_fuel_switch_cost_", primary_id, ".csv")) %>%
  mutate(
    electricity_volume_saved_in_MWh = -(rail_elec_consumption_PJ + switched_to_rail_elec_PJ) * 277777.78,
    electricity_volume_saved_value_rail  = electricity_volume_saved_in_MWh * 111
  ) %>%
  rename(rail_fuel_switch_net_cost = `rail_fuel_switch_net_cost_$`)
raw$trans_light <- read_csv(paste0("100k_runs_postprocessing/tmp/transportation_light_duty_fuel_switch_cost_", primary_id, ".csv")) %>%
  rename(fuel_switch_net_cost_light = `fuel_switch_net_cost_$`)
raw$ccs <- read_csv(paste0("100k_runs_postprocessing/tmp/fugitive_emissions_and_ccs_", primary_id, ".csv")) %>%
  select(primary_id, time_period, capex_ccs, opex_ccs) %>%
  mutate(inv_ccs = (capex_ccs + opex_ccs))

# ----- Data Clean -----
# ---- This pipeline prepares the electric price shock to go into the energy IO shock ----
cons_shocks <- raw$energy %>%  #electricity cost shocks
  group_by(time_period, primary_id) %>%
  summarise(
    capex = sum(capex, na.rm = TRUE),
    opex = sum(opex, na.rm = TRUE),
    total_usd = sum(total_usd, na.rm = TRUE),
    production = sum(production, na.rm = TRUE),
    .groups = "drop"  # optional: ungroup the result
  ) %>%
  group_by(primary_id) %>%
  group_modify(~ revenue_requirement(.x, time_period, capex, opex, wacc = parameters["utility_wacc"], life = parameters["utility_plant_life"])) %>%
  select(time_period, primary_id, production, revenue_requirement) 

# ---- This pipeline preps the IO shocks for energy ----
energy_shocks <- raw$energy %>% #just energy shocks and electricity cost shocks
  pivot_wider(
    id_cols = c("primary_id", "time_period"),
    names_from = prod_type,
    values_from = c(capex, opex),
    names_glue = "{prod_type}_{.value}"
  ) %>%
  rowwise() %>%
  mutate(
    other_capex = sum(c_across(ends_with("capex") & !starts_with("wind") & !starts_with("solar")), na.rm=TRUE),
    opex = sum(c_across(ends_with("opex")), na.rm=TRUE)
  ) %>%
  ungroup() %>%
  left_join(cons_shocks, by=c("primary_id", "time_period")) %>%
  select(primary_id, time_period, production, solar_capex, wind_capex, other_capex, opex, revenue_requirement) %>%
  group_by(primary_id) %>%
  mutate(
    solar_capex_smooth = annual_capex_alloc(solar_capex, ((time_period-3)*12), (time_period*12))[-c(1, 2), ],
    wind_capex_smooth = annual_capex_alloc(wind_capex, ((time_period-3)*12), (time_period*12))[-c(1, 2), ],
    other_capex_smooth = annual_capex_alloc(other_capex, ((time_period-3)*12), (time_period*12))[-c(1, 2), ],
    production_kwh = production * 278000000,
    cost_per_kwh = revenue_requirement * 1000000 / production_kwh
  ) %>%
  ungroup()

shocks <- energy_shocks%>%
  crossing(raw$sector_alloc) %>%
  mutate(
    solar_capex_shock = solar_capex_smooth * solar_power_capex_alloc * 1000000,
    wind_capex_shock = wind_capex_smooth * wind_power_capex_alloc * 1000000,
    other_capex_shock = other_capex_smooth * other_power_capex_alloc * 1000000,
    power_opex_shock = opex * power_opex_alloc * 1000000,
    labor_income_shock = revenue_requirement * - labor_income_alloc * 1000000,
    total_shock = solar_capex_shock + wind_capex_shock + other_capex_shock + power_opex_shock + labor_income_shock
  ) %>%
  select(primary_id, time_period, RIMSID_64, RIMSID_376, solar_capex_shock, wind_capex_shock, other_capex_shock, power_opex_shock, labor_income_shock, total_shock) %>%
  filter(total_shock != 0)

# ---- This pipeline preps the IO shocks for industrial ----
ind_shocks <- raw$industrial %>%
  rowwise() %>%
  mutate(
    eff_capex = sum(c_across(starts_with("efficiency_capex")), na.rm=TRUE) + sum(c_across(starts_with("energy_demand_capex"))),
    eff_opex = sum(c_across(starts_with("efficiency_opex")), na.rm=TRUE) + sum(c_across(starts_with("energy_demand_opex"))),
    demand_coal = sum(c_across(starts_with("energy_demand") & ends_with("coal") & -contains("capex") & -contains("opex")) , na.rm=TRUE),
    demand_coke = sum(c_across(starts_with("energy_demand") & ends_with("coal") & -contains("capex") & -contains("coke")) , na.rm=TRUE),
    demand_diesel	= sum(c_across(starts_with("energy_demand") & ends_with("diesel") & -contains("capex") & -contains("opex")) , na.rm=TRUE),
    demand_electricity = sum(c_across(starts_with("energy_demand") & ends_with("electricity") & -contains("capex") & -contains("opex")) , na.rm=TRUE),
    demand_furnace_gas	= sum(c_across(starts_with("energy_demand") & ends_with("furnace_gas") & -contains("capex") & -contains("opex")) , na.rm=TRUE),
    demand_gasoline	= sum(c_across(starts_with("energy_demand") & ends_with("gasoline") & -contains("capex") & -contains("opex")) , na.rm=TRUE),
    demand_hydrogen = sum(c_across(starts_with("energy_demand") & ends_with("hydrogen") & -contains("capex") & -contains("opex")) , na.rm=TRUE),
    demand_natural_gas	= sum(c_across(starts_with("energy_demand") & ends_with("natural_gas") & -contains("capex") & -contains("opex")) , na.rm=TRUE),
    demand_oil =	sum(c_across(starts_with("energy_demand") & ends_with("oil") & -contains("capex") & -contains("opex")) , na.rm=TRUE)
  ) %>%
  select(primary_id, time_period, eff_capex, eff_opex, demand_coal, demand_coke, demand_diesel, demand_electricity, demand_furnace_gas, demand_gasoline, demand_hydrogen, demand_natural_gas, demand_oil) %>%
  crossing(raw$sector_alloc) %>%
  mutate(
    indus_eff_capex_shock = industrial_efficiency_capex * eff_capex	,
    indus_eff_opex_shock = industrial_efficiency_opex * eff_opex	,
    indus_eff_demand_coal_shock = industrial_efficiency_demand_coal * demand_coal	,
    indus_eff_demand_coke_shock = industrial_efficiency_demand_coke	* demand_coke,
    indus_eff_demand_diesel_shock = industrial_efficiency_demand_diesel	* demand_diesel,
    indus_eff_demand_elec_shock = industrial_efficiency_demand_electricity * demand_electricity,
    indus_eff_demand_furnace_shock = industrial_efficiency_demand_furnace_gas	* demand_furnace_gas,
    indus_eff_demand_gasoline_shock = industrial_efficiency_demand_gasoline	* demand_gasoline,
    indus_eff_demand_hydrogen_shock = industrial_efficiency_demand_hydrogen	* demand_hydrogen,
    indus_eff_demand_gas_shock = industrial_efficiency_demand_natural_gas	* demand_natural_gas,
    indus_eff_demand_oil_shock = industrial_efficiency_demand_oil	* demand_oil,
    total_shock = indus_eff_capex_shock + indus_eff_opex_shock,
    la_rps = .47
  ) %>% #DROPPED OTHERS. NOW THE total_shock is just capex+opex
  filter(total_shock != 0) %>%
  select(primary_id, time_period, la_rps, RIMSID_64, RIMSID_376,indus_eff_capex_shock, 
         indus_eff_opex_shock, indus_eff_demand_coal_shock, indus_eff_demand_coke_shock, 
         indus_eff_demand_diesel_shock, indus_eff_demand_elec_shock, indus_eff_demand_furnace_shock, 
         indus_eff_demand_gasoline_shock, indus_eff_demand_hydrogen_shock, indus_eff_demand_gas_shock,
         indus_eff_demand_oil_shock, total_shock)

# ---- This pipeline preps the IO shocks for transportation ----
trans_shocks <- left_join(raw$trans_elec, raw$trans_heavy, by=c("primary_id", "time_period")) %>% 
  left_join(raw$trans_non_elec, by=c("primary_id", "time_period")) %>%
  left_join(raw$trans_rail, by=c("primary_id", "time_period")) %>%
  left_join(raw$trans_light, by=c("primary_id", "time_period")) %>%
  crossing(raw$sector_alloc) %>%
  mutate(
    trans_elec_shock = transportation_electric * `electricity_transportation_cost_$`	,
    trans_heavy_shock = transportation_heavy_duty	* -fuel_switch_net_cost_heavy,
    trans_non_elec_shock = transportation_non_electric	* transportation_efficiency_non,
    trans_rail_shock = transportation_rail * rail_fuel_switch_net_cost	,
    trans_light_shock = transportation_light_duty	* fuel_switch_net_cost_light,
    elec_consumption_shock = elec_consumption * (electricity_volume_saved_value + electricity_volume_saved_value_non + electricity_volume_saved_value_rail),
    total_shock = trans_elec_shock + trans_heavy_shock + trans_non_elec_shock + trans_rail_shock + trans_light_shock + elec_consumption_shock 
  ) %>%
  filter(total_shock != 0) %>%
  select(primary_id, time_period, RIMSID_64, RIMSID_376, trans_elec_shock, trans_heavy_shock, trans_non_elec_shock, trans_rail_shock, trans_light_shock, elec_consumption_shock, total_shock) 
  
  
# ---- This pipeline preps the IO shocks for CCS ----
ccs_shocks <- raw$ccs %>%
  crossing(raw$sector_alloc) %>%
  mutate(ccs_shock = ccs * inv_ccs * 1000000) %>%
  select(primary_id, time_period, RIMSID_64, RIMSID_376,ccs_shock) %>%
  filter(ccs_shock != 0)

# ----- Data Save -----
saveRDS(energy_shocks,paste0("100k_runs_postprocessing/tmp/energy_data_", primary_id, ".rds"))
saveRDS(shocks, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/intermediate/power_shocks.rds")
saveRDS(ind_shocks, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/intermediate/ind_shocks.rds")
saveRDS(ccs_shocks, "jobs_data_prep/lsu_code/Water-Institute-Leap-dev/data/intermediate/ccs_shocks.rds") 