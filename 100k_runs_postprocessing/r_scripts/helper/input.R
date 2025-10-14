library(dplyr)

import_f <- function(df,
                     RIMSID_64="RIMSID_64",
                     RIMSID_376 = "RIMSID_376",
                     project_lat="project_lat",
                     project_long="project_long",
                     shock="shock",
                     us_rps="us_rps",
                     la_rps="la_rps",
                     project_fips="project_fips",
                     gov_adjust = "gov_adjust",
                     time = "time") {
  # Make a mapping of new names to user-supplied (or default) names
  mapping <- c(
    RIMSID_64 = RIMSID_64,
    RIMSID_376 = RIMSID_376,
    project_lat = project_lat,
    project_long = project_long,
    shock = shock,
    us_rps= us_rps,
    la_rps= la_rps,
    project_fips = project_fips,
    gov_adjust = gov_adjust,
    time =  time
  )
  # Only rename if the column is present
  std_df <- df %>%
    rename(any_of(mapping))
  std_df$shock_id <- seq_len(nrow(std_df))
  if (!"us_rps" %in% names(df)) std_df$us_rps <- NA
  if (!"la_rps" %in% names(df)) std_df$la_rps <- NA
  return(std_df)
}
