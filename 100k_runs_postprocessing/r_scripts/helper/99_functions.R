#----import_csv_folder()----
#import csv - takes folderpath to read all csv into tibbles
import_csv_folder <- function(folder_path, env = globalenv()) {
  
  csv_files <- list.files(
    path = folder_path,
    pattern = "\\.csv$",
    full.names = TRUE
  )
  
  file_names <- tools::file_path_sans_ext(basename(csv_files))
  
  walk2(
    .x = csv_files,
    .y = file_names,
    .f = function(file, name) {
      cat(blue(paste0("Loading ", name, "...\n")))
      data <- read_csv(file, show_col_types = FALSE) |> 
        janitor::clean_names()
      assign(x = name, value = data, envir = env)
      cat(green(paste0("Done loading ", name, "\n")))
    }
  )
}
#----collapse_sum()----
# collapse, summing variables
collapse_sum <- function(data, ..., .groups = "drop") {
  vars <- rlang::ensyms(...)
  
  data %>%
    summarise(
      !!!purrr::set_names(vars, vars) %>%
        purrr::map(~ expr(sum(!!.x, na.rm = TRUE))),
      .groups = .groups
    )
}
#----prepare_leim_data()----
#combine leim datasets and make necessary changes to rpc
prepare_leim_data <- function(df_64, df_376) {
  clean_leim <- function(df, slice_n, rims_type_val, rimsid_col, desc_col) {
    cols_to_convert <- df %>%
      summarise(across(everything(), ~ any(grepl("[%$]", .[1])))) %>%
      select(where(identity)) %>%
      names()
    
    df %>%
      slice(1:slice_n) %>%
      rename(rimsid = !!rimsid_col, rimsid_desc = !!desc_col) %>%
      mutate(rims_type = rims_type_val) %>%
      mutate(across(where(is.character), ~ str_replace_all(., "-", "0"))) %>%
      mutate(across(
        all_of(
          intersect(
            c(cols_to_convert, grep("employment", names(.), value = TRUE)),
            names(select(., where(is.character)))
          )
        ),
        readr::parse_number
      ))
  }
  
  leim_64_cleaned <- clean_leim(df_64, nrow(df_64), 64, "rimsid_64", "industry_unique_identifier")
  leim_376_cleaned <- clean_leim(df_376, nrow(df_376), 376, "rimsid_376_1", "rimsid_376_3")
  
  bind_rows(leim_64_cleaned, leim_376_cleaned) %>%
    mutate(sector_id = paste0(rims_type, "_", rimsid)) %>%
    select(-contains("rimsid"), -rims_type) %>%
    select(-starts_with("baton_rouge", ignore.case = TRUE)) %>%
    
    # Set only *_direct_* variables for selected sector_ids to 1,000,000
    mutate(across(
      matches("(_direct_value_added|_direct_gross_output|_direct_earnings)$"),
      ~ if_else(sector_id %in% c("64_64", "376_376"), 1e6, .)
    )) %>%
    
    pivot_longer(
      cols = matches("^(louisiana|u_s)_(?!.*purchasing_share).*", perl = TRUE),
      names_to = c("geography", "impact_type"),
      names_pattern = "^(louisiana|u_s)_(.*)$",
      values_to = "multiplier"
    ) %>%
    pivot_wider(
      names_from = geography,
      values_from = multiplier,
      names_glue = "{geography}_multiplier",
      values_fn = first
    ) %>%
    mutate(
      louisiana_purchasing_share = louisiana_purchasing_share / 100,
      u_s_purchasing_share = u_s_purchasing_share / 100,
      louisiana_purchasing_share = if_else(
        sector_id %in% c("376_24", "376_374"), #change RPC for certain sectors
        louisiana_purchasing_share / 2,
        louisiana_purchasing_share
      )
    )
}

#----summarise_impacts()----
#collapse data, summarizing impacts for employment, earnings, and value added
summarise_impacts <- function(data, ..., .groups = "drop") {
  prefixes <- enquos(...) %>% map_chr(as_name)
  
  # Build summarise expressions
  summarise_exprs <- list()
  
  for (prefix in prefixes) {
    # Build variable names
    la_col <- sym(paste0(prefix, "_louisiana_impact"))
    us_col <- sym(paste0(prefix, "_us_impact"))
    
    la_name <- paste0(prefix, "_louisiana_impact")
    us_name <- paste0(prefix, "_us_impact")
    
    # Build dynamic expressions using :=
    summarise_exprs[[la_name]] <- expr(
      if (unique(impact_metric) == "Employment") {
        mean(!!la_col, na.rm = TRUE)
      } else {
        sum(!!la_col, na.rm = TRUE)
      }
    )
    
    summarise_exprs[[us_name]] <- expr(
      if (unique(impact_metric) == "Employment") {
        mean(!!us_col, na.rm = TRUE)
      } else {
        sum(!!us_col, na.rm = TRUE)
      }
    )
  }
  
  data %>%
    summarise(!!!summarise_exprs, .groups = .groups)
}
#----revenue_requirement()----
# calculate a revenue requirement for utilities to shock as labor income
revenue_requirement <- function(df, time_period, capex, opex, wacc, life) {
  # Standardize column names for ease of reference
  df <- df %>%
    rename(
      time_period = {{time_period}},
      capex = {{capex}},
      opex = {{opex}}
    ) %>%
    arrange(time_period)
  
  # Create full depreciation schedule: list of depreciation values per year
  time_vector <- df$time_period
  n_years <- length(time_vector)
  depreciation_vec <- numeric(n_years)
  
  for (i in seq_along(time_vector)) {
    capex_year <- time_vector[i]
    capex_value <- df$capex[i]
    
    # Allocate depreciation to future years (starting next year)
    for (j in 1:life) {
      dep_year <- capex_year + j
      idx <- which(time_vector == dep_year)
      if (length(idx) == 1) {
        depreciation_vec[idx] <- depreciation_vec[idx] + capex_value / life
      }
    }
  }
  
  df <- df %>%
    mutate(
      depreciation = depreciation_vec,
      cum_capex = cumsum(capex),
      cum_depreciation = cumsum(depreciation),
      rate_base = cum_capex - cum_depreciation,
      return = rate_base * wacc,
      revenue_requirement = return + opex
    )
  
  return(df)
}

# Allocates CAPEX using overnight cost allocation
cum_allocation <- function(start, end) {
  #This function calculates the allocation matrix using the start and end dates
  #rows represent dates (monthly). Columns represents separate input items
  alpha <- 4.082
  beta <- 3.25
  cum_matrix <- outer(
    seq(from = min(start)+1, to = max(end), by = 1) ,
    1:length(start),
    function(t,j) {
      frac <- ifelse(t > start[j], (t-start[j]) / (end[j]-start[j]), 0)
      frac <- ifelse(t > end[j], 1, frac)
      (1-cos(frac * pi / 2)^beta)^alpha
    }
  )
  return(cum_matrix)
}

allocation <- function(start, end) {
  matrix <- rbind(0, diff(cum_allocation(start, end)))
  return(matrix)
}

capex_allocated <- function(capex, start, end) {
  output <- allocation(start, end) %*% capex
  return(output)
}
  
annual_capex_alloc <- function(capex, start, end) {
  vector <- capex_allocated(capex, start, end)
  years <- rep(
    seq_len(ceiling(length(vector)/12)), 
    each=12
    )
  annual <- rowsum(vector, years)
  return(annual)
}
