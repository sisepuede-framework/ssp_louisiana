
#################################################
# Post processing process
#################################################
rm(list=ls())

# load packages
library(data.table)
library(parallel)
library(aws.s3)



aws <- "/usr/local/bin/aws"  
cmd <- paste(
  shQuote(aws),
  "s3 ls",
  shQuote("s3://sisepuede-data/run_database/sisepuede_run_2025-10-03t16;42;41.795254/model_output/region=louisiana/")
)

# folders for each instance
folders <- system(cmd, intern = TRUE)
cat(paste(folders, collapse = "\n"))
numbers <- sub(".*model_output_(\\d+)/.*", "\\1", folders)


# define paths
bucket <- "sisepuede-data"

FILEIDPOSIX  <- "2025-10-03t16;42;41.795254"
INSTANCEID   <- numbers[2]  # change this to run different instances
REGION       <- "louisiana"      
TABLETYPE    <- "model_output"

object <- 
  paste0("run_database/sisepuede_run_",
                 FILEIDPOSIX,"/",
                 TABLETYPE,"/","region=",
                 REGION,"/model_output_",
                 INSTANCEID,"/data.csv"
                 )

# download file 
save_object(
  object = object,
  bucket = bucket,
  file = "data.csv"  # local output name
)

df <- fread('data.csv')

table(df$primary_id)












#regions 

#load emissions targets 
te_all<-read.csv('1000_runs_ensamble_postprocessing/cw/emission_targets_LA_2021.csv')

target_country <- "LA"
te_all<-te_all[,c("Subsector","Gas","Vars","Edgar_Class",target_country)]
te_all[,"tvalue"] <- te_all[,target_country]
te_all[,target_country] <- NULL
#target_vars <- unlist(strsplit(te_all$Vars,":"))

#output.folder <- paste0(root,"LHC_sample/2025_02_11/meta/")

dir.output <- '1000_runs_ensamble_postprocessing/ensemble_data/meta_decomposed/'

primary_ids <-unique(df$primary_id)
primary_ids

for (run in primary_ids)
{

data_all <- subset(df, primary_id == run)
data_all <- as.data.frame(data_all)
data_all[is.na(data_all)] <- 0
rall <- unique(data_all$region)

#set params of intertemporal decomposition function

initial_conditions_id <- unique(data_all$primary_id)
time_period_ref <- 7

dim(data_all)
data_all <- subset(data_all,time_period>=time_period_ref)
dim(data_all)

source('1000_runs_ensamble_postprocessing/r_scripts/100K_runs/intertemporal_function_baseline_mapping_timeref.r')
z<-1
rescale(z,rall,data_all,te_all,initial_conditions_id,dir.output,time_period_ref,run)    

}


#now collect all decomposed runs and the experimental files 
setDTthreads(0)
files <- list.files(dir.output, pattern = "\\.csv$", full.names = TRUE)
dt <- rbindlist(lapply(files, fread), use.names = TRUE, fill = TRUE)
dim(dt)


# merge with attributes
run <-'1000_runs_ensamble_postprocessing/ssp_output/sisepuede_run_2025-10-03t16;42;41.795254/'
att <- fread(paste0(run, 'ATTRIBUTE_PRIMARY.csv'))
dt <- merge(dt, att, by="primary_id", all.x=TRUE)

table(att$strategy_id, exclude = NULL)

out_path <- '1000_runs_ensamble_postprocessing/ensemble_data/out/sisepuede_run_2025-10-03t16;42;41.795254.csv'
fwrite(dt, out_path)

#upload to S3

# aws s3 cp "/Users/fabianfuentes/git/ssp_louisiana/1000_runs_ensamble_postprocessing/ensemble_data/out/sisepuede_results_IDE_2025-09-18t09;19;22.726476.csv" 's3://sisepuede-data/transfers/sisepuede_run_2025-09-18t09;19;22.726476/'



