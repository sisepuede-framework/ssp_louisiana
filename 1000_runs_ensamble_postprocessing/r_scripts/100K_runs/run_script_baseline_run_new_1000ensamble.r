library(data.table)

#regions 

#load emissions targets 
te_all<-read.csv('1000_runs_ensamble_postprocessing/cw/emission_targets_louisiana_250903.csv')

target_country <- "LA"
te_all<-te_all[,c("Subsector","Gas","Vars","Edgar_Class",target_country)]
te_all[,"tvalue"] <- te_all[,target_country]
te_all[,target_country] <- NULL
#target_vars <- unlist(strsplit(te_all$Vars,":"))

#output.folder <- paste0(root,"LHC_sample/2025_02_11/meta/")
output.folder <- '1000_runs_ensamble_postprocessing/ensemble_data/meta/'
dir.output <- '1000_runs_ensamble_postprocessing/ensemble_data/meta_decomposed/'

files_names <- list.files(output.folder,".csv")
# removes <- c(404928,404929,404930,404931,404932,404933,404934,404935,404936,404937,404938,404939,404940,404941,404942,404943,404944,404945,404946,404947,404948)
# removes <- paste0(as.character(removes),".csv")
# files_names <- subset(files_names,!(files_names%in%removes))

#for (run in 1:length(files_names))
chunks <- split(1:length(files_names), cut(1:length(files_names), 10, labels = FALSE))

for (run in chunks[[10]])
{
#run <- 30
output.file <- files_names[run]
#output.file <- "403327.csv"
data_all <- read.csv(paste0(output.folder,output.file))
data_all[is.na(data_all)] <- 0
rall <- unique(data_all$region)

#set params of intertemporal decomposition function

initial_conditions_id <- unique(data_all$primary_id)
time_period_ref <- 7

dim(data_all)
data_all <- subset(data_all,time_period>=time_period_ref)
dim(data_all)

source('1000_runs_ensamble_postprocessing/r_scripts/1000runs/intertemporal_function_baseline_mapping_timeref.r')
z<-1
rescale(z,rall,data_all,te_all,initial_conditions_id,dir.output,time_period_ref,run)    
}


#now collect all decomposed runs and the experimental files 
setDTthreads(0)
files <- list.files(dir.output, pattern = "\\.csv$", full.names = TRUE)
dt <- rbindlist(lapply(files, fread), use.names = TRUE, fill = TRUE)
dim(dt)

# weird_runs <- c(403425,403434,403454,403554,402522,403810,403873,404177,404330,404375,404410,404485,404502,
#                 404804,404965,404988,405073,405225,405369,402744,402758,403085,402469,403128,403214,403261,403386,
#                 405108,403318,405108,403447)
# dt <- subset(dt,!(primary_id%in%weird_runs))
# dim(dt)




# merge with attributes
run <-'1000_runs_ensamble_postprocessing/ssp_output/sisepuede_run_2025-09-18t09;19;22.726476/'
att <- fread(paste0(run, 'ATTRIBUTE_PRIMARY.csv'))
dt <- merge(dt, att, by="primary_id", all.x=TRUE)

table(att$strategy_id, exclude = NULL)

out_path <- '1000_runs_ensamble_postprocessing/ensemble_data/out/sisepuede_results_IDE_2025-09-18t09;19;22.726476.csv'
fwrite(dt, out_path)

#upload to S3
aws s3 cp "/Users/fabianfuentes/git/ssp_louisiana/1000_runs_ensamble_postprocessing/ensemble_data/out/sisepuede_results_IDE_2025-09-18t09;19;22.726476.csv" 's3://sisepuede-data/transfers/sisepuede_run_2025-09-18t09;19;22.726476/'