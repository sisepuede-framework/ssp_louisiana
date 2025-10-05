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


# ----- Data Load -----
baseline <- readRDS("data/clean/baseline.rds")
power_impact <- readRDS("data/clean/power_impact.rds")
industry_impact <- readRDS("data/clean/industry_impact.rds")
ccs_impact <- readRDS("data/clean/ccs_impact.rds")
power_diff <- readRDS("data/clean/power_diff.rds")
industry_diff <- readRDS("data/clean/industry_diff.rds")
ccs_diff <- readRDS("data/clean/ccs_diff.rds")
primaries <- readRDS("data/intermediate/primaries.rds")


# ----- Baseline Plots -----
vars <- list("la_employment_total", "la_value_total", "la_earnings_total", "la_employment_direct") #Vars to graph
maps <- list(
  label = c(la_employment_total= "LA Jobs Supported", la_value_total= "LA GDP Added (millions)", la_earnings_total= "LA Earnings (millions)", la_employment_direct="LA Direct Jobs Supported"), 
  scales = c(la_employment_total= 1, la_value_total= 1e-6, la_earnings_total=1e-6, la_employment_direct=1), 
  datasets = list(power_impact=power_impact, industry_impact=industry_impact, ccs_impact=ccs_impact, power_diff=power_diff, industry_diff=industry_diff, ccs_diff=ccs_diff),
  graph_names = c(power_impact = "Power", industry_impact= "Industry", ccs_impact= "CCS", power_diff= "diff_Power", industry_diff= "diff_Industry", ccs_diff= "diff_CCS"),
  titles = c(power_impact = "Power Impact", industry_impact= "Industry Impact", ccs_impact= "CCS Impact", power_diff= "Power Impact vs Baseline", industry_diff= "Industrial Impact vs Baseline", ccs_diff= "CCS Impact vs Baseline")
)


for (var in vars) {
  #Plots graphs for baseline scenario
  graph <- ggplot(baseline$power_impact, mapping=aes(x = time ) ) +
    geom_line(aes(y= !!sym(var)),color=get_ces_color("purple")) +
    labs(x = "Year", y = maps$label[var]) + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
    scale_y_continuous(labels=scales::label_number(scale=maps$scales[var], big.mark = ","), breaks=scales::pretty_breaks(n=6)) +
    ces_theme() 
  
  graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                    logo_path = "sources/logo/CESLogo.png")
  
  save_ces_plot(graph, paste0("output/Econ_Impact/baseline/LA_power_", var, "baseline.png"))
  
  graph <- ggplot(baseline$industry_impact, mapping=aes(x = time ) ) +
    geom_line(aes(y= !!sym(var)),color=get_ces_color("purple")) +
    labs(x = "Year", y = maps$label[var], title="Baseline") + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
    scale_y_continuous(labels=scales::label_number(scale=maps$scales[var], big.mark = ","), breaks=scales::pretty_breaks(n=6)) +
    ces_theme() 
  
  graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                    logo_path = "sources/logo/CESLogo.png")
  
  save_ces_plot(graph, paste0("output/Econ_Impact/baseline/LA_industry_", var, "baseline.png"))
}

# ----- Plot Loop -----

for (id in primaries) {
  for (data_used in names(maps$datasets)) {
    for (var in vars) {
      this_data <- maps$datasets[[data_used]][[as.character(id)]]
      #Plots graphs for scenario id
      if (!is.null(this_data)) {
      graph <- ggplot(this_data, mapping=aes(x = time ) )+
        geom_line(aes(y= !!sym(var)),color=get_ces_color("purple")) +
        labs(x = "Year", y = maps$label[var], title=paste(maps$titles[data_used], as.character(id))) + scale_x_continuous(limits=c(0,36), expand = c(0, 0)) +
        scale_y_continuous(labels=scales::label_number(scale=maps$scales[var], big.mark = ","), breaks=scales::pretty_breaks(n=6)) +
        ces_theme() 
      
      graph <-  add_ces_source_and_logo(graph, source_text = "Source: CES Analysis",
                                        logo_path = "sources/logo/CESLogo.png")
      save_ces_plot(graph, paste0("output/Econ_Impact/scenarios/", maps$graph_names[[data_used]], "/", var, as.character(id),".png"))
      }
    }
  }
}

