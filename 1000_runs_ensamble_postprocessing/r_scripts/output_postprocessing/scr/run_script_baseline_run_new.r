################################################################################
# This script runs the intertemporal decomposition for the baseline run
################################################################################

te_all <- fread(paste0("1000_runs_ensamble_postprocessing/r_scripts/output_postprocessing/data/emission_targets_",region,"_",year_ref,".csv")) 
te_all <- data.frame(te_all)
#te_all <- subset(te_all,Subsector%in%c( "lvst","lsmm","agrc","ippu","waso","trww","frst","lndu","soil"))
target_country <- iso_code3
te_all<-te_all[,c("Subsector","Gas","Vars","Edgar_Class",target_country)]

colnames(te_all)

te_all[,"tvalue"] <- te_all[,target_country]

te_all[,target_country] <- NULL
colnames(te_all)
target_vars <- unlist(strsplit(te_all$Vars,":"))

# data from SiSePuede
data_all<-read.csv(paste0(dir.output,output.file))

rall <- unique(data_all$region)

#set params of rescaling function
initial_conditions_id <- "_0"
time_period_ref <- year_ref-2015

dim(data_all)
data_all <- subset(data_all,time_period>=time_period_ref)
dim(data_all)

data_all$emission_co2e_co2_agrc_soil_carbon_organic_soils <- 0

#revise which sector-gas ids are zero at baseline 
te_all$simulation <- 0
for (i in 1:nrow(te_all))
 {
    #i<- 4
    vars <- unlist(strsplit(te_all$Vars[i],":"))
    if (length(vars)>1) {
    te_all$simulation[i] <- as.numeric(rowSums(data_all[data_all$primary_id==gsub("_","",initial_conditions_id) &  data_all$time_period==time_period_ref,vars]))
    } else {
     te_all$simulation[i] <- as.numeric(data_all[data_all$primary_id==gsub("_","",initial_conditions_id) &  data_all$time_period==time_period_ref,vars])   
    }
    print(paste0("complete ", i ))
}

te_all$simulation <- ifelse(te_all$simulation==0 & te_all$tvalue>0,0,1)
correct<- aggregate(list(factor_correction=te_all$simulation),list(Edgar_Class=te_all$Edgar_Class),mean)
te_all <- merge(te_all,correct,by="Edgar_Class")
te_all$tvalue <- te_all$tvalue/te_all$factor_correction
te_all$simulation<-NULL 
te_all$factor_correction<-NULL
te_all$Edgar_Class<-NULL

#now run

source("1000_runs_ensamble_postprocessing/r_scripts/output_postprocessing/scr/intertemporal_decomposition.r")
z<-1
rescale(z,rall,data_all,te_all,initial_conditions_id,dir.output,time_period_ref)

print('Finish:run_script_baseline_run_new_asp process')

