library(writexl)
library(openxlsx)

collapse_totals <- function(df) {
  summary_vec <- c(
    us_output_direct        = sum(df$us_output_direct, na.rm = TRUE),
    us_output_indirect      = sum(df$us_output_indirect, na.rm = TRUE),
    us_output_induced       = sum(df$us_output_induced, na.rm = TRUE),
    us_output_total         = sum(df$us_output_total, na.rm = TRUE),

    us_value_direct         = sum(df$us_value_direct, na.rm = TRUE),
    us_value_indirect       = sum(df$us_value_indirect, na.rm = TRUE),
    us_value_induced        = sum(df$us_value_induced, na.rm = TRUE),
    us_value_total          = sum(df$us_value_total, na.rm = TRUE),

    us_earnings_direct      = sum(df$us_earnings_direct, na.rm = TRUE),
    us_earnings_indirect    = sum(df$us_earnings_indirect, na.rm = TRUE),
    us_earnings_induced     = sum(df$us_earnings_induced, na.rm = TRUE),
    us_earnings_total       = sum(df$us_earnings_total, na.rm = TRUE),

    us_employment_direct    = sum(df$us_employment_direct, na.rm = TRUE),
    us_employment_indirect  = sum(df$us_employment_indirect, na.rm = TRUE),
    us_employment_induced   = sum(df$us_employment_induced, na.rm = TRUE),
    us_employment_total     = sum(df$us_employment_total, na.rm = TRUE),

    la_output_direct        = sum(df$la_output_direct, na.rm = TRUE),
    la_output_indirect      = sum(df$la_output_indirect, na.rm = TRUE),
    la_output_induced       = sum(df$la_output_induced, na.rm = TRUE),
    la_output_total         = sum(df$la_output_total, na.rm = TRUE),

    la_value_direct         = sum(df$la_value_direct, na.rm = TRUE),
    la_value_indirect       = sum(df$la_value_indirect, na.rm = TRUE),
    la_value_induced        = sum(df$la_value_induced, na.rm = TRUE),
    la_value_total          = sum(df$la_value_total, na.rm = TRUE),

    la_earnings_direct      = sum(df$la_earnings_direct, na.rm = TRUE),
    la_earnings_indirect    = sum(df$la_earnings_indirect, na.rm = TRUE),
    la_earnings_induced     = sum(df$la_earnings_induced, na.rm = TRUE),
    la_earnings_total       = sum(df$la_earnings_total, na.rm = TRUE),

    la_employment_direct    = sum(df$la_employment_direct, na.rm = TRUE),
    la_employment_indirect  = sum(df$la_employment_indirect, na.rm = TRUE),
    la_employment_induced   = sum(df$la_employment_induced, na.rm = TRUE),
    la_employment_total     = sum(df$la_employment_total, na.rm = TRUE)
  )
  return(summary_vec)
}

collapse_totals_time <- function(df) {
  summary <- df %>%
    group_by(time) %>%
    summarize(
    us_output_direct        = sum(us_output_direct, na.rm = TRUE),
    us_output_indirect      = sum(us_output_indirect, na.rm = TRUE),
    us_output_induced       = sum(us_output_induced, na.rm = TRUE),
    us_output_total         = sum(us_output_total, na.rm = TRUE),

    us_value_direct         = sum(us_value_direct, na.rm = TRUE),
    us_value_indirect       = sum(us_value_indirect, na.rm = TRUE),
    us_value_induced        = sum(us_value_induced, na.rm = TRUE),
    us_value_total          = sum(us_value_total, na.rm = TRUE),

    us_earnings_direct      = sum(us_earnings_direct, na.rm = TRUE),
    us_earnings_indirect    = sum(us_earnings_indirect, na.rm = TRUE),
    us_earnings_induced     = sum(us_earnings_induced, na.rm = TRUE),
    us_earnings_total       = sum(us_earnings_total, na.rm = TRUE),

    us_employment_direct    = sum(us_employment_direct, na.rm = TRUE),
    us_employment_indirect  = sum(us_employment_indirect, na.rm = TRUE),
    us_employment_induced   = sum(us_employment_induced, na.rm = TRUE),
    us_employment_total     = sum(us_employment_total, na.rm = TRUE),

    la_output_direct        = sum(la_output_direct, na.rm = TRUE),
    la_output_indirect      = sum(la_output_indirect, na.rm = TRUE),
    la_output_induced       = sum(la_output_induced, na.rm = TRUE),
    la_output_total         = sum(la_output_total, na.rm = TRUE),

    la_value_direct         = sum(la_value_direct, na.rm = TRUE),
    la_value_indirect       = sum(la_value_indirect, na.rm = TRUE),
    la_value_induced        = sum(la_value_induced, na.rm = TRUE),
    la_value_total          = sum(la_value_total, na.rm = TRUE),

    la_earnings_direct      = sum(la_earnings_direct, na.rm = TRUE),
    la_earnings_indirect    = sum(la_earnings_indirect, na.rm = TRUE),
    la_earnings_induced     = sum(la_earnings_induced, na.rm = TRUE),
    la_earnings_total       = sum(la_earnings_total, na.rm = TRUE),

    la_employment_direct    = sum(la_employment_direct, na.rm = TRUE),
    la_employment_indirect  = sum(la_employment_indirect, na.rm = TRUE),
    la_employment_induced   = sum(la_employment_induced, na.rm = TRUE),
    la_employment_total     = sum(la_employment_total, na.rm = TRUE)
  ) %>%
    ungroup()
  return(summary)
}

parish_collapse <- function(df) {
  summary_df <- df %>%
    group_by(parish,fips) %>%
    summarise(
      parish_output_direct     = sum(parish_output_direct, na.rm = TRUE),
      parish_output_indirect   = sum(parish_output_indirect, na.rm = TRUE),
      parish_output_induced    = sum(parish_output_induced, na.rm = TRUE),
      parish_output_total      = sum(parish_output_total, na.rm = TRUE),

      parish_value_direct      = sum(parish_value_direct, na.rm = TRUE),
      parish_value_indirect    = sum(parish_value_indirect, na.rm = TRUE),
      parish_value_induced     = sum(parish_value_induced, na.rm = TRUE),
      parish_value_total       = sum(parish_value_total, na.rm = TRUE),

      parish_earnings_direct   = sum(parish_earnings_direct, na.rm = TRUE),
      parish_earnings_indirect = sum(parish_earnings_indirect, na.rm = TRUE),
      parish_earnings_induced  = sum(parish_earnings_induced, na.rm = TRUE),
      parish_earnings_total    = sum(parish_earnings_total, na.rm = TRUE),

      parish_employment_direct   = sum(parish_employment_direct, na.rm = TRUE),
      parish_employment_indirect = sum(parish_employment_indirect, na.rm = TRUE),
      parish_employment_induced  = sum(parish_employment_induced, na.rm = TRUE),
      parish_employment_total    = sum(parish_employment_total, na.rm = TRUE),
    ) %>%
  ungroup()
  return(summary_df)
}

taxes_collapse <- function(df) {
  summary_vec <- c(
    us_direct_tax        = sum(df$us_direct_tax, na.rm = TRUE),
    us_indirect_tax      = sum(df$us_indirect_tax, na.rm = TRUE),
    us_induced_tax       = sum(df$us_induced_tax, na.rm = TRUE),
    us_total_tax         = sum(df$us_total_tax, na.rm = TRUE),

    la_direct_sales_tax         = sum(df$la_direct_sales_tax, na.rm = TRUE),
    la_indirect_sales_tax       = sum(df$la_indirect_sales_tax, na.rm = TRUE),
    la_induced_sales_tax        = sum(df$la_induced_sales_tax, na.rm = TRUE),
    la_total_sales_tax          = sum(df$la_total_sales_tax, na.rm = TRUE),

    la_direct_income_tax      = sum(df$la_direct_income_tax, na.rm = TRUE),
    la_indirect_income_tax    = sum(df$la_indirect_income_tax, na.rm = TRUE),
    la_induced_income_tax     = sum(df$la_induced_income_tax, na.rm = TRUE),
    la_total_income_tax       = sum(df$la_total_income_tax, na.rm = TRUE),

    la_direct_ci_tax    = sum(df$la_direct_ci_tax, na.rm = TRUE),
    la_indirect_ci_tax  = sum(df$la_indirect_ci_tax, na.rm = TRUE),
    la_induced_ci_tax   = sum(df$la_induced_ci_tax, na.rm = TRUE),
    la_total_ci_tax     = sum(df$la_total_ci_tax, na.rm = TRUE),

    la_direct_other_tax        = sum(df$la_direct_other_tax, na.rm = TRUE),
    la_indirect_other_tax      = sum(df$la_indirect_other_tax, na.rm = TRUE),
    la_induced_other_tax       = sum(df$la_induced_other_tax, na.rm = TRUE),
    la_total_other_tax         = sum(df$la_total_other_tax, na.rm = TRUE),

    la_direct_total_taxes         = sum(df$la_direct_total_taxes, na.rm = TRUE),
    la_indirect_total_taxes       = sum(df$la_indirect_total_taxes, na.rm = TRUE),
    la_induced_total_taxes        = sum(df$la_induced_total_taxes, na.rm = TRUE),
    la_total_total_taxes          = sum(df$la_total_total_taxes, na.rm = TRUE)
  )
  return(summary_vec)
}

parish_tax_collapse <- function(df) {
  summary_df <- df %>%
    group_by(parish,fips) %>%
    summarise(
      parish_direct_sales_tax     = sum(parish_direct_sales_tax, na.rm = TRUE),
      parish_indirect_sales_tax   = sum(parish_indirect_sales_tax, na.rm = TRUE),
      parish_induced_sales_tax    = sum(parish_induced_sales_tax, na.rm = TRUE),
      parish_total_sales_tax      = sum(parish_total_sales_tax, na.rm = TRUE),

      parish_direct_property_tax  = sum(parish_direct_property_tax, na.rm = TRUE),
      parish_indirect_property_tax = sum(parish_indirect_property_tax, na.rm = TRUE),
      parish_induced_property_tax = sum(parish_induced_property_tax, na.rm = TRUE),
      parish_total_property_tax    = sum(parish_total_property_tax, na.rm = TRUE),

      parish_direct_other_tax   = sum(parish_direct_other_tax, na.rm = TRUE),
      parish_indirect_other_tax = sum(parish_indirect_other_tax, na.rm = TRUE),
      parish_induced_other_tax  = sum(parish_induced_other_tax, na.rm = TRUE),
      parish_total_other_tax    = sum(parish_total_other_tax, na.rm = TRUE),

      parish_direct_total_taxes   = sum(parish_direct_total_taxes, na.rm = TRUE),
      parish_indirect_total_taxes = sum(parish_indirect_total_taxes, na.rm = TRUE),
      parish_induced_total_taxes  = sum(parish_induced_total_taxes, na.rm = TRUE),
      parish_total_total_taxes    = sum(parish_total_total_taxes, na.rm = TRUE),
    ) %>%
    ungroup()
  return(summary_df)
}

full_run_old <- function(path, ex="CON", df, ...) {
  main_df <- leim_calc(df, ...)
  main_df <- taxes_calc(main_df)
  main_summary <- collapse_totals(main_df)
  main_tax_summary <- taxes_collapse(main_df)

  main_summary_df <- data.frame(Name = names(main_summary), Value=as.vector(main_summary)) %>%
    rename(Impact = Name)
  main_tax_summary_df <- data.frame(Name = names(main_tax_summary), Value=as.vector(main_tax_summary)) %>%
    rename(Tax = Name)

  parish_df <- parish_calc(df, ...)
  parish_df <- parish_taxes_calc(parish_df)
  parish_summary <- parish_collapse(parish_df)
  parish_tax_summary <- parish_tax_collapse(parish_df)

  wb <- loadWorkbook(path)
  writeData(wb, paste0(ex," Main"), main_summary_df)
  writeData(wb, paste0(ex, " Tax"), main_tax_summary_df)
  writeData(wb, paste0(ex, " Parish"), parish_summary)
  writeData(wb, paste0(ex, " Parish Tax"), parish_tax_summary)
  saveWorkbook(wb, path, overwrite=TRUE)
}

full_run <- function(ex="CON", df, ...) {
  output = list()
  output$main_df <- leim_calc(df, ...)
  output$main_df <- taxes_calc(output$main_df)
  main_summary <- collapse_totals(output$main_df)
  main_tax_summary <- taxes_collapse(output$main_df)

  output$main_summary_df <- data.frame(Name = names(main_summary), Value=as.vector(main_summary)) %>%
    rename(Impact = Name)
  output$main_tax_summary_df <- data.frame(Name = names(main_tax_summary), Value=as.vector(main_tax_summary)) %>%
    rename(Tax = Name)

  output$parish_df <- parish_calc(df, ...)
  output$parish_df <- parish_taxes_calc(output$parish_df)
  output$parish_summary <- parish_collapse(output$parish_df)
  output$parish_tax_summary <- parish_tax_collapse(output$parish_df)

  openxlsx::write.xlsx(output, paste0("output/throughput/", ex, "_leim.xlsx"), overwrite=TRUE)

  return(output)
}

adjust_run <- function(ex="CON", df, ...) {
  #This is for a run with the Parish spending adjustment included.
  multiroot_initializer <- adjust_initialize(df, ...)
  adjusted_rows <- adjust_rootsolve(multiroot_initializer$input_df, multiroot_initializer$adjust_rows, multiroot_initializer$start)
  shock_df <- bind_rows(import_f(df, ...), adjusted_rows)
  full_run(ex, shock_df)
}



