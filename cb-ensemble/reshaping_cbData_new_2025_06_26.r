dir.data <- "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/simulations raw/lousiana_cb_strategias/"
files_table <- data.frame(file_name=list.dirs(path = dir.data, full.names = FALSE, recursive = FALSE))
split_names <- do.call(rbind, strsplit(as.character(files_table$file_name), "_"))
files_table$primary_id <- as.numeric(split_names[,2])

target_cb_file <- "cost_benefit_results_"

cb_data_all <- list()
for (i in 1:nrow(files_table))
{ 
#read cb data 
# i <- 60
 cb_data <-read.csv(paste0(dir.data,"cb_",files_table$primary_id[i],"/", target_cb_file,files_table$primary_id[i],".csv"))
 cb_data$primary_id <- files_table$primary_id[i]
 cb_data$X <- NULL
 #merge both  
 cb_data_all <- append(cb_data_all,list(cb_data))
}
cb_data_all <- do.call("rbind",cb_data_all)
cb_chars <- data.frame(do.call(rbind, strsplit(as.character(cb_data_all$variable), ":")))
colnames(cb_chars) <- c("name","sector","cb_type","item_1","item_2")
dim(cb_data_all)
dim(cb_chars)
cb_data_all <- cbind(cb_data_all,cb_chars)
cb_data_all$value <- cb_data_all$value/1e9 #all values in billions
#add Year 
cb_data_all$Year <- cb_data_all$time_period+2015

#add ids 
#primary id
dir.ids <- "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/simulations raw/data_for_LSU/"
ap <-  read.csv(paste0(dir.ids,"ATTRIBUTE_PRIMARY.csv"))
ap <- subset(ap,primary_id%in%unique(cb_data_all$primary_id))

#strategy_id 
as <-  read.csv(paste0(dir.ids,"ATTRIBUTE_STRATEGY.csv"))
as <- subset(as,strategy_id%in%unique(ap$strategy_id))
as <- unique(as[,c("strategy_code", "strategy_id")])

dim(cb_data_all)
cb_data_all <-merge(cb_data_all,as,by="strategy_code")
dim(cb_data_all)
cb_data_all$ids <- paste(cb_data_all$variable,cb_data_all$strategy_id,sep=":")

#create aggregation table for tornado 
cb <- cb_data_all 
cb <- aggregate(list(Cumulative = cb$value),list(strategy_id=cb$strategy_id, strategy_code = cb$strategy_code, sector = cb$sector,cb_type = cb$cb_type, Year = cb$Year),sum, na.rm=TRUE)
#change from long to wide format  
library(reshape2)
wide_cb <- dcast(cb, strategy_id +strategy_code + sector + cb_type ~ Year, value.var = "Cumulative")
write.csv(wide_cb,paste0(dir.ids,"wide_cb_data_2025_06_26.csv"),row.names=FALSE)
















######
# +++++++++++++++
#####

#read all folders 
 dir.data <- "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/simulations raw/lousiana_cb_strategias/"
 files_table <- data.frame(file_name=list.dirs(path = dir.data, full.names = FALSE, recursive = FALSE))
 split_names <- do.call(rbind, strsplit(as.character(files_table$file_name), "_"))
 files_table$primary_id <- as.numeric(split_names[,2])

#now lets read inside each folder 
# target_cb_file <- "economy_wide_cost_benefit_results_"

target_cb_file <- "cost_benefit_results_"

#we need to read the time_series one., or not maybe 

cb_data_all <- list()
for (i in 1:nrow(files_table))
{ 
#read cb data 
# i <- 60
 cb_data <-read.csv(paste0(dir.data,"cb_",files_table$primary_id[i],"/", target_cb_file,files_table$primary_id[i],".csv"))
 cb_data$primary_id <- files_table$primary_id[i]
 #merge both  
 cb_data_all <- append(cb_data_all,list(cb_data))
}
cb_data_all <- do.call("rbind",cb_data_all)
cb_chars <- data.frame(do.call(rbind, strsplit(as.character(cb_data_all$variable), ":")))
colnames(cb_chars) <- c("name","sector","cb_type","item_1","item_2")






cb_chars$cb_type_new <- cb_chars$cb_type
cb_chars$cb_type_new <- ifelse(grepl("technical_cost",cb_chars$cb_type)==TRUE & grepl("capex",cb_chars$item_2)==TRUE, "technical_cost_capex",cb_chars$cb_type_new)
cb_chars$cb_type_new <- ifelse(grepl("technical_cost",cb_chars$cb_type)==TRUE & grepl("opex",cb_chars$item_2)==TRUE, "technical_cost_opex",cb_chars$cb_type_new)
cb_chars$cb_type <- cb_chars$cb_type_new
cb_chars$cb_type_new <- NULL 
cb_chars$cb_type <- ifelse(cb_chars$cb_type=="technical_cost","technical_cost_capex",cb_chars$cb_type)
dim(cb_data_all)
dim(cb_chars)
cb_data_all <- cbind(cb_data_all,cb_chars)

#aggregate results 
#by sector
 cb_data1 <- aggregate(list(value=cb_data_all$value),list(primary_id=cb_data_all$primary_id,sector=cb_data_all$sector,time_period=cb_data_all$time_period),sum)
 cb_data1$value <- cb_data1$value/1e6
 cb_data1 <- reshape2::dcast(cb_data1, formula = primary_id + time_period ~ sector , value.var = "value")
 colnames(cb_data1) <- c(c("primary_id","time_period"),paste0(subset(colnames(cb_data1),!(colnames(cb_data1)%in%c("primary_id","time_period"))),"_cb"))
 cb_data1[is.na(cb_data1)] <- 0

#by item 
 cb_data2 <- aggregate(list(value=cb_data_all$value),list(primary_id=cb_data_all$primary_id,cb_type=cb_data_all$cb_type,time_period=cb_data_all$time_period),sum)
 cb_data2$value <- cb_data2$value/1e6
 cb_data2 <- reshape2::dcast(cb_data2, formula = primary_id + time_period ~ cb_type, value.var = "value")
 cb_data2[is.na(cb_data2)] <- 0
#merge both  
 dim(cb_data1)
 dim(cb_data2)
 cb_data_all <- merge(cb_data1,cb_data2)
 dim(cb_data_all)

cb_data_all$Direct_Benefit <- rowSums(cb_data_all[,c("technical_savings","fuel_cost" ,"crop_value","lvst_value","ippu_value")]) 
cb_data_all$Indirect_Benefit <- rowSums(cb_data_all[,c( "air_pollution","congestion","ecosystem_services","env_pollution","land_pollution","road_safety","water_pollution","consumer_savings")]) 

  #write this file 
dir.out  <- r'(C:\Users\edmun\OneDrive\Edmundo-ITESM\3.Proyectos\59. Lousiana Project\LA_CaseStudy\Tableau\)'
write.csv(cb_data_all,paste0(dir.out,"cb_data.csv"),row.names=FALSE)


#process read all files together  
#identify which are capex vs opex, etc, 
#rbind them all,
#then agreggate and transform 




cb_data_all <- list()
for (i in 1:nrow(files_table))
{ 
#read cb data 
# i <- 60
 cb_data <-read.csv(paste0(dir.data,"cb_",files_table$primary_id[i],"\\", target_cb_file,files_table$primary_id[i],".csv"))
 cb_chars <- data.frame(do.call(rbind, strsplit(as.character(cb_data$variable), ":")))
 colnames(cb_chars) <- c("name","sector","cb_type","item_1","item_2")
 cb_data <- cbind(cb_data,cb_chars)
#aggregate results in two tables  
#by sector
 cb_data1 <- aggregate(list(value=cb_data$value),list(sector=cb_data$sector),sum)
 cb_data1$value <- cb_data1$value/1e6
 cb_data1 <- reshape2::dcast(cb_data1, formula = 1 ~ sector, value.var = "value")
 cb_data1$primary_id <- files_table$primary_id[i]
#by item 
 cb_data2 <- aggregate(list(value=cb_data$value),list(cb_type=cb_data$cb_type),sum)
 cb_data2$value <- cb_data2$value/1e6
 cb_data2 <- reshape2::dcast(cb_data2, formula = 1 ~ cb_type, value.var = "value")
 cb_data2$primary_id <- files_table$primary_id[i]
#merge both  
cb_data_all <- append(cb_data_all,list(merge(cb_data1,cb_data2)))
 rm(cb_data1,cb_data2,cb_data)
}

cb_data_all <- plyr::rbind.fill(cb_data_all)

# Replace all NA values with 0
cb_data_all[is.na(cb_data_all)] <- 0
head(cb_data_all)

cb_data_all$Capex <- cb_data_all[,c("technical_cost")]
cb_data_all$Opex <- cb_data_all[,c("system_cost")] 
cb_data_all$Direct_Benefit <- rowSums(cb_data_all[,c("technical_savings","fuel_cost" ,"crop_value","lvst_value","ippu_value")]) 
cb_data_all$Indirect_Benefit <- rowSums(cb_data_all[,c( "air_pollution","congestion","ecosystem_services","env_pollution","land_pollution","road_safety","water_pollution","consumer_savings")]) 
cb_data_all[,"1"] <- NULL

  #write this file 
dir.out  <- r'(C:\Users\edmun\OneDrive\Edmundo-ITESM\3.Proyectos\59. Lousiana Project\LA_CaseStudy\Tableau\)'
write.csv(cb_data_all,paste0(dir.out,"cb_data.csv"),row.names=FALSE)


#now reshape CB_data per year  
#read all folders 
 dir.data <- r'(C:\Users\edmun\Downloads\lousiana_cb_strategias\)'
 files_table <- data.frame(file_name=list.dirs(path = dir.data, full.names = FALSE, recursive = FALSE))
 split_names <- do.call(rbind, strsplit(as.character(files_table$file_name), "_"))
 files_table$primary_id <- as.numeric(split_names[,2])
#now lets read inside each folder 
 target_cb_file <- "cost_benefit_results_"
cb_data_all <- list()
for (i in 1:nrow(files_table))
{ 
#read cb data 
# i <- 60
 cb_data <-read.csv(paste0(dir.data,"cb_",files_table$primary_id[i],"\\", target_cb_file,files_table$primary_id[i],".csv"))
 cb_data$region <- gsub("united_states_of_america","louisiana",cb_data$region)
 cb_data$future_id <- NULL
 cb_data$difference_value <- NULL 
 cb_data$difference_variable <- NULL 
 cb_data$X <- NULL 
 cb_data$primary_id <- files_table$primary_id[i]
 cb_data_all <- append(cb_data_all,list(cb_data))
 rm(cb_data)
}
cb_data_all <- do.call("rbind",cb_data_all)
dir.out  <- r'(C:\Users\edmun\OneDrive\Edmundo-ITESM\3.Proyectos\59. Lousiana Project\LA_CaseStudy\Tableau\)'
write.csv(cb_data_all,paste0(dir.out,"cb_data_care_model.csv"),row.names=FALSE)
head(cb_data_all)
