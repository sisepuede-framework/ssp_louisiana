library(dplyr)

merge_multipliers <- function(df, multipliers_64, multipliers_376) {
  if (!"RIMSID_64" %in% colnames(df)) df$RIMSID_64 <- NA
  if (!"RIMSID_376" %in% colnames(df)) df$RIMSID_376 <- NA
  # Drop rows with non-missing RIMSID_376, so RIMSID_376 takes precedence
  df_376 <- df %>% filter(!is.na(RIMSID_376) & RIMSID_376 != 0) %>% select(-RIMSID_64)
  df_64  <- df %>% filter((is.na(RIMSID_376) | RIMSID_376 == 0) & !is.na(RIMSID_64) & RIMSID_64 != 0)

  # Join each subset with the correct multipliers
  if (nrow(df_376) > 0) {
    df_376 <- left_join(df_376, multipliers_376, by = "RIMSID_376")
  }
  if (nrow(df_64) > 0) {
    df_64 <- left_join(df_64, multipliers_64, by = "RIMSID_64")
  }

  # Combine back together
  df <- bind_rows(df_376, df_64)
  return(df)
}

us_economic_impact <- function(df) {
  df <- mutate(df,
               us_rps = dplyr::coalesce(us_rps, USRPC),
               la_rps = dplyr::coalesce(la_rps, LARPC),
               # US-level
               us_purchase = shock * us_rps,

               us_output_direct   = us_purchase,
               us_output_indirect = (us_purchase / US_Direct_Output) * US_Indirect_Output,
               us_output_induced  = (us_purchase / US_Direct_Output) * US_Induced_Output,
               us_output_total    = (us_purchase / US_Direct_Output) * US_Total_Output,

               us_value_direct   = (us_purchase / US_Direct_Output) * US_Direct_VA,
               us_value_indirect = (us_purchase / US_Direct_Output) * US_Indirect_VA,
               us_value_induced  = (us_purchase / US_Direct_Output) * US_Induced_VA,
               us_value_total    = (us_purchase / US_Direct_Output) * US_Total_VA,

               us_earnings_direct   = (us_purchase / US_Direct_Output) * US_Direct_Earnings,
               us_earnings_indirect = (us_purchase / US_Direct_Output) * US_Indirect_Earnings,
               us_earnings_induced  = (us_purchase / US_Direct_Output) * US_Induced_Earnings,
               us_earnings_total    = (us_purchase / US_Direct_Output) * US_Total_Earnings,

               us_employment_direct   = (us_purchase / US_Direct_Output) * US_Direct_Employment,
               us_employment_indirect = (us_purchase / US_Direct_Output) * US_Indirect_Employment,
               us_employment_induced  = (us_purchase / US_Direct_Output) * US_Induced_Employment,
               us_employment_total    = (us_purchase / US_Direct_Output) * US_Total_Employment
  )
}

la_economic_impact <- function(df) {
  df <- mutate(df,
               # Louisiana-level
               la_purchase = shock * la_rps,
               la_output_direct   = la_purchase,
               la_output_indirect = (la_purchase / LA_Direct_Output) * LA_Indirect_Output,
               la_output_induced  = (la_purchase / LA_Direct_Output) * LA_Induced_Output,
               la_output_total    = (la_purchase / LA_Direct_Output) * LA_Total_Output,

               la_value_direct   = (la_purchase / LA_Direct_Output) * LA_Direct_VA,
               la_value_indirect = (la_purchase / LA_Direct_Output) * LA_Indirect_VA,
               la_value_induced  = (la_purchase / LA_Direct_Output) * LA_Induced_VA,
               la_value_total    = (la_purchase / LA_Direct_Output) * LA_Total_VA,

               la_earnings_direct   = (la_purchase / LA_Direct_Output) * LA_Direct_Earnings,
               la_earnings_indirect = (la_purchase / LA_Direct_Output) * LA_Indirect_Earnings,
               la_earnings_induced  = (la_purchase / LA_Direct_Output) * LA_Induced_Earnings,
               la_earnings_total    = (la_purchase / LA_Direct_Output) * LA_Total_Earnings,

               la_employment_direct   = (la_purchase / LA_Direct_Output) * LA_Direct_Employment,
               la_employment_indirect = (la_purchase / LA_Direct_Output) * LA_Indirect_Employment,
               la_employment_induced  = (la_purchase / LA_Direct_Output) * LA_Induced_Employment,
               la_employment_total    = (la_purchase / LA_Direct_Output) * LA_Total_Employment
  )
  return(df)
}

leim_calc <- function(df, ...) {
  std_df <- import_f(df, ...)
  merged_df <- merge_multipliers(std_df, leim_64, leim_376)
  merged_df <- us_economic_impact(merged_df)
  output_df <- la_economic_impact(merged_df)
  return(output_df)
}

