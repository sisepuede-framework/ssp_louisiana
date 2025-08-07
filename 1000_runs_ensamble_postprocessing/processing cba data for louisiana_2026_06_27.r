#processing cba data for louisiana  
dir.data <- "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample_cba/"
cb_data <- read.csv(paste0(dir.data,"cost_benefit_results_lousiana.csv")) #python cba output 
cb_chars <- data.frame(do.call(rbind, strsplit(as.character(cb_data$variable), ":")))
colnames(cb_chars) <- c("name","sector","cb_type","item_1","item_2")
cb_data <- cbind(cb_data,cb_chars)
cb_data$value <- 1*cb_data$value/1e9
cb_data$Year <- cb_data$time_period+2015

#agreggate 
cb_data <- aggregate(list(Cumulative = cb_data$value),list( cb_type = cb_data$cb_type,
                                                            strategy_code = cb_data$strategy_code,
                                                            primary_id = cb_data$primary_id,
                                                            future_id = cb_data$future_id, 
                                                            Year = cb_data$Year),sum, na.rm=TRUE)

#data table for meta-model 
library(reshape2)
wide_cb <- dcast(cb_data, primary_id + future_id + strategy_code + Year ~ cb_type, value.var = "Cumulative")
write.csv(wide_cb,paste0(dir.data,"wide_cb_data_lhc1000_2025_06_27.csv"),row.names=FALSE)


