#load emissions targets 
te_all<-read.csv('1000_runs_ensamble_postprocessing/cw/emission_targets_louisiana.csv')
target_country <- "LA"
te_all<-te_all[,c("Subsector","Gas","Vars","Edgar_Class",target_country)]
te_all[,"tvalue"] <- te_all[,target_country]
te_all[,target_country] <- NULL
target_vars <- unlist(strsplit(te_all$Vars,":"))

output.folder <- '1000_runs_ensamble_postprocessing/ensemble_data/parsed_runs/'
files_names <- list.files(output.folder,".csv")

length(files_names)

# removes <- c("140.csv","142.csv","144.csv","166.csv","212.csv",
#              "224.csv","291.csv",
#              "345.csv","413.csv",
#              "487.csv","533.csv",
#              "571.csv","594.csv",
#              "881.csv","882.csv",
#              "89.csv")
# files_names <- subset(files_names,!(files_names%in%removes))
# run
# files_names[run]


# (Optional) Pre-load the function so forked workers inherit it.
source('1000_runs_ensamble_postprocessing/r_scripts/intertemporal_function_baseline_mapping_timeref.r')

# Worker function for a single run
process_one <- function(run) {
  tryCatch({
    # Ensure DT uses 1 thread per worker
    data.table::setDTthreads(1)

    # Read input
    output.file <- files_names[run]
    dt <- data.table::fread(file.path(output.folder, output.file))
    df <- as.data.frame(dt)  # keep this if `rescale()` expects data.frame

    # Params
    rall <- unique(df$region)
    dir.output <- '1000_runs_ensamble_postprocessing/ensemble_data/meta_decomposed/'
    initial_conditions_id <- unique(df$primary_id)
    time_period_ref <- 7

    # Filter by reference time
    df <- subset(df, time_period >= time_period_ref)

    # Safety: if not inherited (rare), source again
    if (!exists("rescale", mode = "function")) {
      source('1000_runs_ensamble_postprocessing/r_scripts/intertemporal_function_baseline_mapping_timeref.r')
    }

    # Call target function
    z <- 1
    rescale(z, rall, df, te_all, initial_conditions_id, dir.output, time_period_ref, run)

    list(run = run, ok = TRUE)
  }, error = function(e) list(run = run, ok = FALSE, err = conditionMessage(e)))
}

# Parallel over all files
ncores <- max(1, detectCores() - 1)
results <- mclapply(seq_along(files_names), process_one,
                    mc.cores = ncores, mc.preschedule = FALSE)

#now collect all decomposed runs and the experimental files 
# One thread per worker to avoid oversubscription
data.table::setDTthreads(1)

outputall.folder <- "1000_runs_ensamble_postprocessing/ensemble_data/meta_decomposed/"
files_names <- list.files(outputall.folder, pattern = "\\.csv$", full.names = TRUE)

ncores <- max(1, detectCores() - 1)

# Read files in parallel, then bind
parts <- mclapply(files_names, function(f) data.table::fread(f),
                  mc.cores = ncores, mc.preschedule = FALSE)
data_complete <- data.table::rbindlist(parts, use.names = TRUE, fill = TRUE)

run <- "1000_runs_ensamble_postprocessing/ssp_output/sisepuede_run_2025-08-28t15;29;22.344855/"
file.name <- "9a7fe49a-fef0-4d39-bcd1-d677f91da13d.csv"

att <- fread(paste0(run, "ATTRIBUTE_PRIMARY.csv"))
head(att)

dim(data_complete)

data_complete <- merge(data_complete,att,by="primary_id")
dim(data_complete)

atts <- read.csv(paste0(run,"ATTRIBUTE_STRATEGY.csv"))

dim(data_complete)
data_complete <- merge(data_complete,atts[c("strategy_id","strategy")],by="strategy_id")
dim(data_complete)


fwrite(data_complete, paste0("1000_runs_ensamble_postprocessing/ensemble_data/Tableau/data/", file.name),row.names=FALSE)

