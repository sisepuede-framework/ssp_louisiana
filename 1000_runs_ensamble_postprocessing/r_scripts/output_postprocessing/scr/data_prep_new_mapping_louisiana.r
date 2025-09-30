# This script prepares the data for emissions mapping in Uganda
file.name <- paste0(region,".csv")
iso_code3 <- iso_code3
Country <- region

mapping <- read.csv(paste0("1000_runs_ensamble_postprocessing/r_scripts/output_postprocessing/data/sisepuede_edgar_active_crosswalk.csv"))

# modification of AG - Livestock:N2O subsector matching
# mapping$Vars[3] <- "emission_co2e_n2o_lsmm_direct_anaerobic_digester:emission_co2e_n2o_lsmm_direct_anaerobic_lagoon:emission_co2e_n2o_lsmm_direct_composting:emission_co2e_n2o_lsmm_direct_daily_spread:emission_co2e_n2o_lsmm_direct_deep_bedding:emission_co2e_n2o_lsmm_direct_dry_lot:emission_co2e_n2o_lsmm_direct_incineration:emission_co2e_n2o_lsmm_direct_liquid_slurry:emission_co2e_n2o_lsmm_direct_paddock_pasture_range:emission_co2e_n2o_lsmm_direct_poultry_manure:emission_co2e_n2o_lsmm_direct_storage_solid:emission_co2e_n2o_lsmm_indirect_anaerobic_digester:emission_co2e_n2o_lsmm_indirect_anaerobic_lagoon:emission_co2e_n2o_lsmm_indirect_composting:emission_co2e_n2o_lsmm_indirect_daily_spread:emission_co2e_n2o_lsmm_indirect_deep_bedding:emission_co2e_n2o_lsmm_indirect_dry_lot:emission_co2e_n2o_lsmm_indirect_incineration:emission_co2e_n2o_lsmm_indirect_liquid_slurry:emission_co2e_n2o_lsmm_indirect_paddock_pasture_range:emission_co2e_n2o_lsmm_indirect_poultry_manure:emission_co2e_n2o_lsmm_indirect_storage_solid"

# add edgar
edgar <- read.csv('1000_runs_ensamble_postprocessing/cw/emission_targets_louisiana_all_years_1990_2021_new_model.csv')
edgar <- data.frame(edgar)
edgar <- subset(edgar,Code==iso_code3)
edgar$Edgar_Class<- paste(edgar$CSC.Subsector,edgar$Gas,sep=":")

#load data  
data <- read.csv(paste0(dir.output,file.name)) 
data <- subset(data,region==Country)

#order data
setorder(data, primary_id, time_period, region)

#emission vars only 
id_vars <-c('region','time_period',"primary_id")
vars <- subset(colnames(data),!(colnames(data)%in%id_vars))
target_vars <- subset(vars,grepl("co2e_",vars)==TRUE)
total_vars <- subset(target_vars,grepl("emission_co2e_subsector_total",target_vars)==TRUE)
target_vars <- subset(target_vars,!(target_vars%in%total_vars))

#load inventory mapping table 
mapping$ids <- paste(row.names(mapping),mapping$Subsector,mapping$Gas,sep="_")
#now create those new columns in the simulation data set 
for  (i in 1:nrow(mapping))
{
#i<- 63
tvars <- mapping$Vars[i]
tvars <- unlist(strsplit(tvars,":"))
tvars <- subset(tvars,tvars%in%colnames(data))
if (length(tvars)>1) {
 data [,mapping$ids[i]] <- rowSums(data[,tvars])
} else if (length(tvars)==1 ) 
{ 
  data [,mapping$ids[i]] <- data[,tvars]
} else {
  data [,mapping$ids[i]] <- 0
} 
}
#now we just keep the new variables and the time period which we will reduce to above 2022
data_new <- data [,c(id_vars,mapping$ids)]
dim(data_new)

#convert from wide to long 
data_new <- data.table(data_new)
data_new <- reshape2::melt(data_new, id.vars = id_vars,
                   measure.vars = mapping$ids,
             )
data_new <- data.frame(data_new)
data_new$ids <- as.character(data_new$variable)

#merge with mapping 
 mapping$Vars <- NULL
 colnames(mapping) <- gsub("Edgar_Sector","CSC.Sector",colnames(mapping))
 colnames(mapping) <- gsub("Edgar_Subsector","CSC.Subsector",colnames(mapping)) 
 data_new <- merge(data_new,mapping,by="ids")

#now aggregare at inventory level 
data_new <- aggregate(list(value=data_new$value),by=list(primary_id=data_new$primary_id,
                                             time_period=data_new$time_period,
                                             Edgar_Class=data_new$Edgar_Class,
                                             CSC.Sector=data_new$CSC.Sector,
                                             CSC.Subsector=data_new$CSC.Subsector,
                                             Subsector=data_new$Subsector),sum)


data_new$Year <- data_new$time_period + 2015
data_new$Gas <- do.call("rbind",strsplit(data_new$Edgar_Class,":"))[,2]


#merge additional files  
att <- read.csv(paste0(dir.output,"ATTRIBUTE_PRIMARY.csv"))
head(att)

dim(data_new)

data_new <- merge(data_new,att,by="primary_id")
dim(data_new)

atts <- read.csv(paste0(dir.output,"ATTRIBUTE_STRATEGY.csv"))

#merge 
dim(data_new)
data_new <- merge(data_new,atts[c("strategy_id","strategy")],by="strategy_id")
dim(data_new)

#melt edgar data 
id_varsEd <- c("Code","Subsector","CSC.Sector","CSC.Subsector","Gas","Edgar_Class")
measure.vars_Ed <- subset(colnames(edgar),grepl("X",colnames(edgar))==TRUE)
edgar <- data.table(edgar)
edgar <- melt(edgar, id.vars = id_varsEd, measure.vars =measure.vars_Ed)
edgar <- data.frame(edgar)
edgar$Year <- as.numeric(gsub("X","",edgar$variable))

#make sure both data frames have the same columns 
#edgar 
edgar$variable <- NULL
edgar$strategy_id <- NA
edgar$primary_id <- NA 
edgar$design_id <- NA 
edgar$future_id <- NA 
edgar$Contry <- Country
edgar$strategy <- "Historical" 
edgar$source <- "EDGAR"
edgar <- subset(edgar,Year<=max(edgar$Year))

#data_new 
data_new$time_period <- NULL 
data_new$Code <- iso_code3 
data_new$Contry <- Country
data_new$source <- "SISEPUEDE"
data_new <- subset(data_new,Year>=max(edgar$Year))


#rbind both 
data_new <- rbind(data_new,edgar)
data_new <- data_new[order(data_new$strategy_id,data_new$CSC.Subsector,data_new$Gas,data_new$Year),]


#write file
dir.tableau <- '1000_runs_ensamble_postprocessing/r_scripts/output_postprocessing/Tableau/'
file.name <- paste0("emissions_", region, "_", year_ref, "_", output.file)

write.csv(data_new,paste0(dir.tableau,file.name),row.names=FALSE)

print('Finish:data_prep_new_mapping process')


