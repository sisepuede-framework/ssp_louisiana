#################################################
# Post processing process
#################################################

# load packages
library(data.table)
library(RColorBrewer)
library(ggplot2)
library(scales)

rm(list=ls())



df <- fread('1000_runs_ensamble_postprocessing/ssp_output/sisepuede_summary_results_run_sisepuede_run_2025-09-17T01;49;45.687668/WIDE_INPUTS_OUTPUTS.csv')
#df <- fread('1000_runs_ensamble_postprocessing/ssp_output/sisepuede_summary_results_run_sisepuede_run_2025-09-17T01;49;45.687668/louisiana.csv')

att <- fread('1000_runs_ensamble_postprocessing/ssp_output/sisepuede_summary_results_run_sisepuede_run_2025-09-17T01;49;45.687668/ATTRIBUTE_PRIMARY.csv')
stt <- fread('1000_runs_ensamble_postprocessing/ssp_output/sisepuede_summary_results_run_sisepuede_run_2025-09-17T01;49;45.687668/ATTRIBUTE_STRATEGY.csv')


df <- merge(df, att, by = "primary_id", all.x = TRUE)

table(df$strategy_id)



 
forest_ch4 <- c("emission_co2e_ch4_frst_methane_mangroves","emission_co2e_ch4_frst_methane_primary","emission_co2e_ch4_frst_methane_secondary")

forest_c02 <- c('emission_co2e_co2_frst_harvested_wood_products','emission_co2e_co2_frst_sequestration_mangroves','emission_co2e_co2_frst_sequestration_primary',
                'emission_co2e_co2_frst_sequestration_secondary','emission_co2e_co2_frst_forest_fires')

df_long <- melt(df, 
                id.vars = c("primary_id",'strategy_id', "time_period"), 
                measure.vars = forest_c02)

ggplot(df_long, aes(x = time_period, y = value, fill = variable)) +
  geom_area(position = "stack") +
  facet_wrap(~ strategy_id, scales = "fixed") +
  scale_fill_viridis_d(option = "turbo") +
  scale_y_continuous(labels = label_number()) +
  scale_x_continuous(labels = label_number()) +
  labs(title = "",
       x = "Time Period",
       y = "",
       fill = "Variable") +
  theme_dark()


ggplot(subset(df_long, variable=='yield_agrc_other_annual_tonne'), aes(x = time_period, y = value, fill = variable)) +
  geom_area(position = "stack") +
  facet_wrap(~ strategy_id, scales = "fixed") +
  scale_fill_viridis_d(option = "turbo") +
  scale_y_continuous(labels = label_number()) +
  scale_x_continuous(labels = label_number()) +
  labs(title = "Vehicle Distance Traveled (Public Transport) Over Time",
       x = "Time Period",
       y = "Distance Traveled",
       fill = "Variable") +
  theme_dark()


pop_lvst_ <- grep("^pop_lvst_", colnames(df), value = TRUE)


df_long <- melt(df, 
                id.vars = c("primary_id", "strategy", "time_period"), 
                measure.vars = pop_lvst_)

ggplot(df_long, aes(x = time_period, y = value, fill = variable)) +
  geom_area(position = "stack") +
  facet_wrap(~ strategy, scales = "fixed") +
  scale_fill_viridis_d(option = "turbo") +
  scale_y_continuous(labels = label_number()) +
  scale_x_continuous(labels = label_number()) +
  labs(title = "",
       x = "Time Period",
       y = "",
       fill = "Variable") +
  theme_dark()

ggplot(subset(df_long, variable=='pop_lvst_buffalo'), aes(x = time_period, y = value, fill = variable)) +
  geom_area(position = "stack") +
  facet_wrap(~ strategy, scales = "fixed") +
  scale_fill_viridis_d(option = "turbo") +
  scale_y_continuous(labels = label_number()) +
  scale_x_continuous(labels = label_number()) +
  labs(title = "Vehicle Distance Traveled (Public Transport) Over Time",
       x = "Time Period",
       y = "Distance Traveled",
       fill = "Variable") +
  theme_dark()











vehicle_distance_traveled_trns_public <- grep("^vehicle_distance_traveled_trns_public", colnames(df), value = TRUE)

df_long <- melt(df, 
                id.vars = c("primary_id", "strategy", "time_period"), 
                measure.vars = vehicle_distance_traveled_trns_public)


ggplot(df_long, aes(x = time_period, y = value, fill = variable)) +
  geom_area(position = "stack") +
  facet_wrap(~ strategy, scales = "fixed") +
  scale_fill_viridis_d(option = "turbo") +
  scale_y_continuous(labels = label_number()) +
  scale_x_continuous(labels = label_number()) +
  labs(title = "Vehicle Distance Traveled (Public Transport) Over Time",
       x = "Time Period",
       y = "Distance Traveled",
       fill = "Variable") +
  theme_dark()


vehicle_distance_traveled_trns_road_light <- grep("^vehicle_distance_traveled_trns_road_light", colnames(df), value = TRUE)

df_long <- melt(df, 
                id.vars = c("primary_id", "strategy", "time_period"), 
                measure.vars = vehicle_distance_traveled_trns_road_light)


ggplot(df_long, aes(x = time_period, y = value, fill = variable)) +
  geom_area(position = "stack") +
  facet_wrap(~ strategy, scales = "fixed") +
  scale_fill_viridis_d(option = "turbo") +
  scale_y_continuous(labels = label_number()) +
  scale_x_continuous(labels = label_number()) +
  labs(title = "",
       x = "Time Period",
       y = "Distance Traveled",
       fill = "Variable") +
  theme_dark()




vehicle_distance_traveled_trns_road_light