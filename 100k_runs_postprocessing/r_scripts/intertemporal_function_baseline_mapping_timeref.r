rescale <-function(z, rall, data_all, te_all, initial_conditions_id, dir.output, time_period_ref, run)
{
  # z <- 1
  tregion <- rall[z]

  # subset data for the region
  data <- subset(data_all, region == tregion)

  # list all emission variables (exclude subsector totals)
  tv1_all <- subset(colnames(data), grepl("co2e_", colnames(data)) == TRUE)
  tv1_all <- subset(tv1_all, grepl("emission_co2e_subsector_total_", tv1_all) == FALSE)

  # build Index (region_primary_id)
  data$Index <- paste0(data$region, "_", data$primary_id)
  inds <- unique(data$Index)
  ref_inds <- paste0(tregion, "_", initial_conditions_id)

  # -----------------------------------------
  # 1) Compute pct_diffs and diffs for all vars
  # -----------------------------------------
  pct_diffs <- list()
  for (i in 1:length(inds)) {
    step1 <- list()
    for (j in 1:length(tv1_all)) {
      pivot <- data[data$Index == inds[i], c("Index", "time_period", tv1_all[j])]

      # remove single-cell NA with variable mean; full-NA variable => set to 0
      pivot[, tv1_all[j]] <- ifelse(is.na(pivot[, tv1_all[j]]) == TRUE,
                                    mean(pivot[, tv1_all[j]], na.rm = TRUE),
                                    pivot[, tv1_all[j]])
      if (is.na(mean(pivot[, tv1_all[j]])) == TRUE) {
        pivot[, tv1_all[j]] <- 0
      } else {
        pivot[, tv1_all[j]] <- pivot[, tv1_all[j]]
      }

      # pct diff and absolute diff
      if (mean(unique(pivot[, tv1_all[j]])) == 0) {
        pivot[, paste0("pct_diff_", tv1_all[j])] <- 0
      } else {
        pivot$diff <- c(diff(pivot[, tv1_all[j]]), 0)
        pivot[, paste0("pct_diff_", tv1_all[j])] <-
          c(0, pivot$diff[1:(nrow(pivot)-1)] / pivot[, tv1_all[j]][1:(nrow(pivot)-1)])
        pivot[, paste0("pct_diff_", tv1_all[j])] <-
          ifelse(is.na(pivot[, paste0("pct_diff_", tv1_all[j])]) == TRUE, 0,
                 pivot[, paste0("pct_diff_", tv1_all[j])])
        pivot[, paste0("pct_diff_", tv1_all[j])] <-
          ifelse(pivot[, paste0("pct_diff_", tv1_all[j])] == Inf, 0,
                 pivot[, paste0("pct_diff_", tv1_all[j])])
      }
      pivot[, paste0("diff_", tv1_all[j])] <- c(0, diff(pivot[, tv1_all[j]]))
      pivot <- pivot[, c("Index", "time_period",
                         paste0("pct_diff_", tv1_all[j]),
                         paste0("diff_", tv1_all[j]))]
      step1 <- append(step1, list(pivot))
    }
    step1 <- Reduce(function(...) merge(..., all = TRUE), step1)
    pct_diffs <- append(pct_diffs, list(step1))
  }
  pct_diffs <- do.call("rbind", pct_diffs)
  pct_diffs <- pct_diffs[order(pct_diffs$Index, pct_diffs$time_period), ]

  # -----------------------------------------
  # 2) Prepare targets: sector-gas mapping
  # -----------------------------------------
  te_all$sector_gas <- paste(row.names(te_all), te_all$Subsector, te_all$Gas, sep = "-")
  sector_gas_all <- unique(te_all$sector_gas)

  # helper: all emission columns present in data (non-subsector totals)
  base_emis_cols <- grep("^emission_co2e_", colnames(data), value = TRUE)
  base_emis_cols <- base_emis_cols[!grepl("^emission_co2e_subsector_total_", base_emis_cols)]

  # -----------------------------------------
  # 3) Initial-year enforcement per sector-gas
  # -----------------------------------------
  for (w in 1:length(sector_gas_all)) {
    sector_gas_i <- sector_gas_all[w]
    tv1_raw <- unlist(strsplit(subset(te_all, sector_gas == sector_gas_i)$Vars, ":"))

    # ensure variables exist in data
    tv1 <- intersect(tv1_raw, colnames(data))
    if (length(tv1) == 0) next

    target_total <- subset(te_all, sector_gas == sector_gas_i)[, "tvalue"]
    # uncalibrated total only for the reference index at the reference year
    uncalibrated_total <- sum(
      data[data$time_period == time_period_ref & data$Index == ref_inds, tv1],
      na.rm = TRUE
    )

    # robust initial-year handling
    if (isTRUE(all.equal(as.numeric(target_total), 0, tolerance = 1e-12))) {
      # zero target => zero out initial year for ref index
      data[data$time_period == time_period_ref & data$Index == ref_inds, tv1] <- 0
    } else if (isTRUE(all.equal(uncalibrated_total, 0, tolerance = 1e-12))) {
      # uncalibrated zero but target > 0 => seed equal shares at initial year for ref index
      eq_share <- as.numeric(target_total) / length(tv1)
      data[data$time_period == time_period_ref & data$Index == ref_inds, tv1] <- eq_share
    } else {
      # normal proportional scaling at initial year for all indices (keeps your original behavior)
      deviation_factor <- as.numeric(target_total) / uncalibrated_total
      data[data$time_period == time_period_ref, tv1] <-
        data[data$time_period == time_period_ref, tv1] * deviation_factor
    }

    # strict equality check for this sector-gas on the ref index at initial year
    stopifnot(
      isTRUE(
        all.equal(
          round(sum(data[data$time_period == time_period_ref & data$Index == ref_inds, tv1], na.rm = TRUE), 6),
          round(as.numeric(target_total), 6),
          tolerance = 1e-6
        )
      )
    )

    # -----------------------------------------
    # 4) Propagate over time using pct_diffs / diffs
    # -----------------------------------------
    for (i in 1:length(inds)) {
      for (j in 1:length(tv1)) {
        init_value <- data[data$Index == paste0(rall[z], "_", initial_conditions_id) &
                             data$time_period == time_period_ref, tv1[j]]
        if (is.na(init_value)) init_value <- 0

        if (init_value == 0) {
          # if initial value is zero, accumulate diffs
          data[data$Index == inds[i], tv1[j]] <-
            init_value + cumsum(pct_diffs[pct_diffs$Index == inds[i], paste0("diff_", tv1[j])])
        } else {
          # otherwise, compound pct changes
          time_change <- cumprod(1 + pct_diffs[pct_diffs$Index == inds[i], paste0("pct_diff_", tv1[j])])
          data[data$Index == inds[i], tv1[j]] <- init_value * time_change
        }
      }
    }
  }

  # -----------------------------------------
  # 5) Optional: final global initial-year sanity check (ref index)
  # -----------------------------------------
  init_mask <- (data$time_period == time_period_ref & data$Index == ref_inds)
  global_init_sum_data   <- sum(data[init_mask, base_emis_cols], na.rm = TRUE)
  global_init_sum_target <- sum(te_all$tvalue, na.rm = TRUE)

  if (!isTRUE(all.equal(global_init_sum_data, global_init_sum_target, tolerance = 1e-6))) {
    warning(sprintf(
      "Initial-year GLOBAL emissions mismatch for %s: data = %.6f, target = %.6f",
      ref_inds, global_init_sum_data, global_init_sum_target
    ))
    # If strict equality is required across *all* emissions at initial year, uncomment:
    # if (global_init_sum_data > 0) {
    #   data[init_mask, base_emis_cols] <- data[init_mask, base_emis_cols] *
    #     (global_init_sum_target / global_init_sum_data)
    # }
  }

  # -----------------------------------------
  # 6) Estimate subsector totals
  # -----------------------------------------
  subsectors <- unique(te_all$Subsector)
  for (a in 1:length(subsectors)) {
    subsector_vars <- unlist(lapply(subset(te_all, Subsector == subsectors[a])$Vars, function(x) { strsplit(x, ":") }))
    subsector_vars <- intersect(subsector_vars, colnames(data))
    if (length(subsector_vars) > 1)  {
      data[, paste0("emission_co2e_subsector_total_", subsectors[a])] <- rowSums(data[, subsector_vars, drop = FALSE], na.rm = TRUE)
    } else if (length(subsector_vars) == 1) {
      data[, paste0("emission_co2e_subsector_total_", subsectors[a])] <- data[, subsector_vars]
    } else {
      data[, paste0("emission_co2e_subsector_total_", subsectors[a])] <- 0
    }
  }

  # -----------------------------------------
  # 7) Write output
  # -----------------------------------------
  data$Index <- NULL
  write.csv(data, paste0(dir.output, tregion, "_", run, ".csv"), row.names = FALSE)
  rm(data)
  print(paste0(dir.output, tregion, "_", run, ".csv"))
}
