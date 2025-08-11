library(data.table)
library(data.table)
#regions 

#Set root directory
# in mac 
root <- "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/"

#load emissions targets 
te_all<-read.csv(paste0(root,"/code/emission_targets_louisiana.csv"))
target_country <- "LA"
te_all<-te_all[,c("Subsector","Gas","Vars","Edgar_Class",target_country)]
te_all[,"tvalue"] <- te_all[,target_country]
te_all[,target_country] <- NULL
target_vars <- unlist(strsplit(te_all$Vars,":"))

output.folder <- paste0(root,"LHC_sample/2025_02_11/meta/")
files_names <- list.files(output.folder,".csv")
removes <- c("140.csv","142.csv","144.csv","166.csv","212.csv",
             "224.csv","291.csv",
             "345.csv","413.csv",
             "487.csv","533.csv",
             "571.csv","594.csv",
             "881.csv","882.csv",
             "89.csv")
files_names <- subset(files_names,!(files_names%in%removes))
run
files_names[run]

for (run in 1:length(files_names))
{
output.file <- files_names[run]
data_all <- read.csv(paste0(output.folder,output.file))
rall <- unique(data_all$region)

#set params of intertemporal decomposition function
dir.output <- paste0(root,"LHC_sample/2025_02_11/meta_decomposed/")
initial_conditions_id <- unique(data_all$primary_id)
time_period_ref <- 7

dim(data_all)
data_all <- subset(data_all,time_period>=time_period_ref)
dim(data_all)

source(paste0(root,"/code/","intertemporal_function_baseline_mapping_timeref.r"))
z<-1
rescale(z,rall,data_all,te_all,initial_conditions_id,dir.output,time_period_ref,run)    
}


#now collect all decomposed runs and the experimental files 
outputall.folder <- paste0(root,"LHC_sample/2025_02_11/meta_decomposed/")
files_names <- list.files(outputall.folder,".csv")
data_complete <- list()
for (i in 1:length(files_names))
#for (i in 1:10)
{
 pivot <- read.csv(paste0(outputall.folder,files_names[i]))
 data_complete <- append(data_complete,list(pivot))
}
data_complete <- do.call("rbind",data_complete)
write.csv(data_complete, paste0(root,"LHC_sample/2025_06_22/sisepuede_results_sisepuede_run_2025-01-14T17;04;06.975301_IDE_WIDE_INPUTS_OUTPUTS.csv"),row.names=FALSE)
