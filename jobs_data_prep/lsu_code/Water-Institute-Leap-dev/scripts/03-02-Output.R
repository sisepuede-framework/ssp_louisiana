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

library(readr)
library(dplyr)
library(tidyr)
library(readxl)
library(ggplot2)
library(CESgraphics)
source("R/99_functions.R")


raw <- list()
inter <- list()
# ----- Data Load ----- 
raw$energy <- read_csv("data/raw/baseline_costs_and_production_by_prodtype.csv") %>%
  group_by(primary_id, time_period) %>%
  mutate(
    production_gwh = production * 278,
    percent_capex = capex / sum(capex),
    percent_opex = opex / sum(opex),
    percent_prod = production_gwh / sum(production_gwh)
    ) 
energy_data <- readRDS("data/intermediate/energy_data.rds") %>%
  select(primary_id, time_period, cost_per_kwh) %>%
  tidyr::pivot_wider(id_cols = time_period, names_from=primary_id, values_from=cost_per_kwh ) %>%
  rename(cost_per_kwh_0 = `0`, cost_per_kwh_71071 = `71071`)
primaries <- readRDS("data/intermediate/primaries.rds")

# ----- Data Prep -----
inter$annual <- raw$energy %>%
  group_by(primary_id, time_period) %>%
  summarise(
    capex = sum(capex, na.rm = TRUE),
    opex = sum(opex, na.rm = TRUE),
    production_gwh = sum(production_gwh, na.rm = TRUE)
            ) %>%
  ungroup() %>%
  pivot_wider(
    id_cols = time_period ,
    names_from = primary_id,
    values_from = c(capex, opex, production_gwh)
  )

# ----- Graphing -----
for (id in primaries) { 
  graph <- ggplot(inter$annual, mapping=aes(x=time_period)) +
    geom_line(aes(y=capex_0, color="Baseline" )) +
    geom_line(aes(y=!!sym(paste0("capex_", id)), color="Scenario")) + 
    labs(x = "Year", y = "CAPEX (Million $)", title="Total") + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
    scale_y_continuous(labels=scales::label_number(scale=1, big.mark = ","), breaks=scales::pretty_breaks(n=8)) +
    scale_color_manual(values=c("Baseline" = get_ces_color("purple"), "Scenario" = get_ces_color("gold"))) + 
    ces_theme() 
  graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                    logo_path = "sources/logo/CESLogo.png")
  save_ces_plot(graph, paste0("output/Econ_Impact/Power/Total_CAPEX/", id, ".png"))
  
  graph <- ggplot(inter$annual, mapping=aes(x=time_period)) +
    geom_line(aes(y=opex_0, color="Baseline" )) +
    geom_line(aes(y=!!sym(paste0("opex_", id)), color="Scenario")) + 
    labs(x = "Year", y = "OPEX (Million $)", title="Total") + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
    scale_y_continuous(labels=scales::label_number(scale=1, big.mark = ","), breaks=scales::pretty_breaks(n=8)) +
    scale_color_manual(values=c("Baseline" = get_ces_color("purple"), "Scenario" = get_ces_color("gold"))) + 
    ces_theme() 
  graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                    logo_path = "sources/logo/CESLogo.png")
  save_ces_plot(graph, paste0("output/Econ_Impact/Power/Total_OPEX/", id, ".png"))
  
  graph <- ggplot(inter$annual, mapping=aes(x=time_period)) +
    geom_line(aes(y=production_gwh_0, color="Baseline" )) +
    geom_line(aes(y=!!sym(paste0("production_gwh_", id)), color="Scenario")) + 
    labs(x = "Year", y = "Production (GWh)", title="Total") + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
    scale_y_continuous(labels=scales::label_number(scale=1, big.mark = ","), breaks=scales::pretty_breaks(n=8)) +
    scale_color_manual(values=c("Baseline" = get_ces_color("purple"), "Scenario" = get_ces_color("gold"))) + 
    ces_theme() 
  graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                    logo_path = "sources/logo/CESLogo.png")
  save_ces_plot(graph, paste0("output/Econ_Impact/Power/Total_Production/", id, ".png"))
}

looping <- unique(raw$energy$prod_type)
for (loop_var in looping) {
  inter$annual <- raw$energy %>%
    filter(prod_type == loop_var) %>%
    group_by(primary_id, time_period) %>%
    pivot_wider(
      id_cols = time_period ,
      names_from = primary_id,
      values_from = c(capex, opex, production_gwh, percent_prod)
    ) 
  for (id in primaries) { 
    graph <- ggplot(inter$annual, mapping=aes(x=time_period)) +
      geom_line(aes(y=capex_0, color="Baseline" )) +
      geom_line(aes(y=!!sym(paste0("capex_", id)), color="Scenario"), alpha=.7) + 
      labs(x = "Year", y = "CAPEX (Million $)", title=paste(loop_var, id)) + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
      scale_y_continuous(labels=scales::label_number(scale=1, big.mark = ","), breaks=scales::pretty_breaks(n=8)) +
      scale_color_manual(values=c("Baseline" = get_ces_color("purple"), "Scenario" = get_ces_color("gold"))) + 
      ces_theme() 
    graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                      logo_path = "sources/logo/CESLogo.png")
    save_ces_plot(graph, paste0("output/Econ_Impact/Power/CAPEX/", loop_var, ".png"))
    
    graph <- ggplot(inter$annual, mapping=aes(x=time_period)) +
      geom_line(aes(y=opex_0, color="Baseline" )) +
      geom_line(aes(y=!!sym(paste0("opex_", id)), color="Scenario"), alpha=.7) + 
      labs(x = "Year", y = "CAPEX (Million $)", title=paste(loop_var, id)) + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
      scale_y_continuous(labels=scales::label_number(scale=1, big.mark = ","), breaks=scales::pretty_breaks(n=8)) +
      scale_color_manual(values=c("Baseline" = get_ces_color("purple"), "Scenario" = get_ces_color("gold"))) + 
      ces_theme() 
    graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                      logo_path = "sources/logo/CESLogo.png")
    save_ces_plot(graph, paste0("output/Econ_Impact/Power/OPEX/", loop_var, ".png"))
    
    graph <- ggplot(inter$annual, mapping=aes(x=time_period)) +
      geom_line(aes(y=production_gwh_0, color="Baseline" )) +
      geom_line(aes(y=!!sym(paste0("production_gwh_", id)), color="Scenario"), alpha=.7) + 
      labs(x = "Year", y = "Production (GWh)", title=paste(loop_var, id)) + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
      scale_y_continuous(labels=scales::label_number(scale=1, big.mark = ","), breaks=scales::pretty_breaks(n=8)) +
      scale_color_manual(values=c("Baseline" = get_ces_color("purple"), "Scenario" = get_ces_color("gold"))) + 
      ces_theme() 
    graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                      logo_path = "sources/logo/CESLogo.png")
    save_ces_plot(graph, paste0("output/Econ_Impact/Power/Prod/", loop_var, ".png"))
    
    graph <- ggplot(inter$annual, mapping=aes(x=time_period)) +
      geom_line(aes(y=percent_prod_0, color="Baseline" , )) +
      geom_line(aes(y=!!sym(paste0("percent_prod_", id)), color="Scenario"), alpha=.7) + 
      labs(x = "Year", y = "Percent of Electricity", title=paste(loop_var, id)) + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
      scale_y_continuous(labels=scales::percent, breaks=scales::pretty_breaks(n=8)) +
      scale_color_manual(values=c("Baseline" = get_ces_color("purple"), "Scenario" = get_ces_color("gold"))) + 
      ces_theme() 
    graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                      logo_path = "sources/logo/CESLogo.png")
    save_ces_plot(graph, paste0("output/Econ_Impact/Power/Percent_Prod/", loop_var, ".png"))  
  }
}

# ----- Plot electric rates -----
for (id in primaries) { 
  graph <- ggplot(energy_data, mapping=aes(x=time_period)) +
    geom_line(aes(y=cost_per_kwh_0, color="Baseline" )) +
    geom_line(aes(y=!!sym(paste0("cost_per_kwh_", id)), color="Scenario"), alpha=.7) + 
    labs(x = "Year", y = "$/KWh", title=paste("Electricity Cost ", id)) + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
    scale_y_continuous(labels=scales::label_number(scale=1, big.mark = ","), breaks=scales::pretty_breaks(n=6)) +
    scale_color_manual(values=c("Baseline" = get_ces_color("purple"), "Scenario" = get_ces_color("gold"))) + 
    ces_theme() 
  graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                    logo_path = "sources/logo/CESLogo.png")
  save_ces_plot(graph, paste0("output/Econ_Impact/Power/cost_per_kwh/", id, ".png"))
}