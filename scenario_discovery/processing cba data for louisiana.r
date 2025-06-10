#processing cba data for louisiana  
cb_data <- read.csv("/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample_cba/cost_benefit_results_lousiana.csv")
cb_chars <- data.frame(do.call(rbind, strsplit(as.character(cb_data$variable), ":")))
colnames(cb_chars) <- c("name","sector","cb_type","item_1","item_2")
cb_data <- cbind(cb_data,cb_chars)
cb_data$value <- 1*cb_data$value/1e9
cb_data$Year <- cb_data$time_period+2015

#test 
subset(cb_data,cb_type=="technical_cost")[cb_data$future_id==46,]

#agreggate 
 cb_data <- aggregate(list(Cumulative = cb_data$value),list(Sector=cb_data$sector,
                                                            cb_type=cb_data$cb_type,
                                                            Strategy=cb_data$strategy,
                                                            future_id=cb_data$future_id),sum, na.rm=TRUE)

 cb_data_raw <- cb_data 
#now merge with experimental features  
 key_pt <- read.csv("/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample/pt.csv")

#merge with cba 
  dim(cb_data)
  cb_data <- merge(cb_data,key, by="future_id")
  dim(cb_data)

#write to test 
  write.csv(cb_data,"/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample_cba/cba_data_ensemble.csv",row.names=FALSE)




#what if we try another format by metling 
library(data.table)
library(reshape2)

vars <- subset(colnames(cb_data),grepl("X",colnames(cb_data))==TRUE)
ids <- subset(colnames(cb_data),!(colnames(cb_data)%in%vars))
cb_data_long <- melt(data.table(cb_data), id.vars = ids, measure.vars = vars) 


##
root<-'/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample/'
key <- read.csv(paste0(root,"variable_specification_to_sample_group.csv"))
key$variable <- paste0("X",key$sample_group)
key <- data.table(key)

tgroups<- unique(key$sample_group)
#create vars names 
all_vars <- list()
for (i in 1:length(tgroups))
{
#i<-1
vars<-data.frame(vars=paste(subset(key,sample_group==tgroups[i])$variable_specification,collapse=" : "),variable=paste0("X",tgroups[i]))
all_vars <- append(all_vars,list(vars))
}
all_vars <- do.call("rbind",all_vars)

#now merge 

dim(cb_data_long)
cb_data_long <- merge (cb_data_long,data.table(all_vars),by="variable")
dim(cb_data_long)

#write to test 
write.csv(cb_data_long,"/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample_cba/cba_data_ensemble_version3.csv",row.names=FALSE)
head(cb_data_long)
summary(cb_data_long$future_id)


#lets do prim on technical cost  

summaries <- list()
for (i in 1: length(unique(cb_data$Sector)))
{
  summaries <- append(summaries,
 list(sector=unique(cb_data$Sector)[i], summary(subset(cb_data_raw,cb_type=="technical_cost" & Sector==unique(cb_data$Sector)[i])$Cumulative)) 
  )
}
summaries
cb_prim1 <- subset(cb_data_raw,cb_type=="technical_cost" & Sector =="ippu")
summary(cb_prim1$Cumulative)

#add keys 
key <- read.csv(paste0(root,"variable_specification_to_sample_group.csv"))
#now merge with experimental features  
key_pt <- read.csv("/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample/pt.csv")


#firt we need to aggregate the data at future_id level 
cb_prim1 <- aggregate(list(Cumulative = cb_prim1$Cumulative),list(future_id=cb_prim1$future_id),sum, na.rm=TRUE)
summary(cb_prim1$Cumulative)

#now merge with pt 
#merge with cba 
  dim(cb_prim1)
  cb_prim1 <- merge(cb_prim1,key_pt, by="future_id")
  dim(cb_prim1)

#ok, now le'ts run our random forest analysis 
 pt <- cb_prim1
 v10<-as.numeric(quantile(pt$Cumulative,0.1))
 v20<-as.numeric(quantile(pt$Cumulative,0.2))
 v30<-as.numeric(quantile(pt$Cumulative,0.3))
 v40<-as.numeric(quantile(pt$Cumulative,0.4))
 v50<-as.numeric(quantile(pt$Cumulative,0.5))
 v60<-as.numeric(quantile(pt$Cumulative,0.6))
 v70<-as.numeric(quantile(pt$Cumulative,0.7))
 v80<-as.numeric(quantile(pt$Cumulative,0.8))
 v90<-as.numeric(quantile(pt$Cumulative,0.9))

#create flags 
pt$v10 <- ifelse(pt$Cumulative >=v10,1,0)
pt$v20 <- ifelse(pt$Cumulative >=v20,1,0)
pt$v30 <- ifelse(pt$Cumulative >=v30,1,0)
pt$v40 <- ifelse(pt$Cumulative >=v40,1,0)
pt$v50 <- ifelse(pt$Cumulative >=v50,1,0)
pt$v60 <- ifelse(pt$Cumulative >=v60,1,0)
pt$v70 <- ifelse(pt$Cumulative >=v70,1,0)
pt$v80 <- ifelse(pt$Cumulative >=v80,1,0)
pt$v90 <- ifelse(pt$Cumulative >=v90,1,0)

summary(pt[,c("v10","v20","v30","v40","v50","v60","v70","v80","v90")])

Xs<-subset(colnames(pt),!(colnames(pt)%in%c("Cumulative","future_id","primary_id","emission_co2e_TOTAL_50","net_zero_flag","selected_flag","cluster_flag","cluster_representative_flag")))
Xs<-subset(Xs,grepl("v",Xs)==FALSE)
summary(pt[,Xs])

#now do the same with random forest 
library(rpart)       # performing regression trees
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(caret)       # bagging

#set params
set.seed(55555)
ctrl <- trainControl(method = "cv",  number = 7) 

#models
 model_1<-as.formula(paste("as.factor(v10)","~",paste(Xs,collapse="+"),sep=""))
 model_2<-as.formula(paste("as.factor(v20)","~",paste(Xs,collapse="+"),sep=""))
 model_3<-as.formula(paste("as.factor(v30)","~",paste(Xs,collapse="+"),sep=""))
 model_4<-as.formula(paste("as.factor(v40)","~",paste(Xs,collapse="+"),sep=""))
 model_5<-as.formula(paste("as.factor(v50)","~",paste(Xs,collapse="+"),sep=""))
 model_6<-as.formula(paste("as.factor(v60)","~",paste(Xs,collapse="+"),sep=""))
 model_7<-as.formula(paste("as.factor(v70)","~",paste(Xs,collapse="+"),sep=""))
 model_8<-as.formula(paste("as.factor(v80)","~",paste(Xs,collapse="+"),sep=""))
 model_9<-as.formula(paste("as.factor(v90)","~",paste(Xs,collapse="+"),sep=""))
 models <- list(model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8,model_9)

#random forest
rdms <- list()
for (i in 1:length(models))
{
# i<-5
 bagged_cv <- train(models[[i]], data    = pt, method = "treebag", trControl = ctrl, importance = TRUE)
 st<-subset(varImp(bagged_cv)$importance,Overall>= as.numeric(quantile(varImp(bagged_cv)$importance[,1],0.75))) 
 st$group<-row.names(st)
 colnames(st)<-gsub("Overall",paste0("Overall","_",i),colnames(st))
 row.names(st)<-NULL
 rdms <- append(rdms,list(st))
} 
rdms<- Reduce(function(...) merge(...,all=TRUE), rdms)
rdms$Score <- rowSums(apply(rdms[,subset(colnames(rdms),colnames(rdms)!="group")],c(1,2),function(x){ifelse(is.na(x)==TRUE,0,1)}))
#add specific group vars
tgroups <- as.numeric(gsub("X","",unique(rdms$group)))

all_vars <- list()
for (i in 1:length(tgroups))
{
#i<-2
vars<-data.frame(vars=paste(subset(key,sample_group==tgroups[i])$variable_specification,collapse=","),group=paste0("X",tgroups[i]))
all_vars <- append(all_vars,list(vars))
}
all_vars <- do.call("rbind",all_vars)

#merge 
dim(rdms)
rdms<-merge(rdms,all_vars,by="group")
dim(rdms)
write.csv(rdms,paste0(root,"va_analysis/random_forest_report_cost_benefits_ippu.csv"),row.names=FALSE)
#writ the prim data  
write.csv(pt,"/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/Tableau/cb_ippu.csv",row.names=FALSE)



####
#now what if we do this with entc
#####
cb_prim1 <- subset(cb_data_raw,cb_type=="technical_cost" & Sector =="entc")
summary(cb_prim1$Cumulative)

#add keys 
key <- read.csv(paste0(root,"variable_specification_to_sample_group.csv"))
#now merge with experimental features  
key_pt <- read.csv("/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample/pt.csv")


#firt we need to aggregate the data at future_id level 
cb_prim1 <- aggregate(list(Cumulative = cb_prim1$Cumulative),list(future_id=cb_prim1$future_id),sum, na.rm=TRUE)
summary(cb_prim1$Cumulative)

#now merge with pt 
#merge with cba 
  dim(cb_prim1)
  cb_prim1 <- merge(cb_prim1,key_pt, by="future_id")
  dim(cb_prim1)

#ok, now le'ts run our random forest analysis 
 pt <- cb_prim1
 v10<-as.numeric(quantile(pt$Cumulative,0.1))
 v20<-as.numeric(quantile(pt$Cumulative,0.2))
 v30<-as.numeric(quantile(pt$Cumulative,0.3))
 v40<-as.numeric(quantile(pt$Cumulative,0.4))
 v50<-as.numeric(quantile(pt$Cumulative,0.5))
 v60<-as.numeric(quantile(pt$Cumulative,0.6))
 v70<-as.numeric(quantile(pt$Cumulative,0.7))
 v80<-as.numeric(quantile(pt$Cumulative,0.8))
 v90<-as.numeric(quantile(pt$Cumulative,0.9))

#create flags 
pt$v10 <- ifelse(pt$Cumulative >=v10,1,0)
pt$v20 <- ifelse(pt$Cumulative >=v20,1,0)
pt$v30 <- ifelse(pt$Cumulative >=v30,1,0)
pt$v40 <- ifelse(pt$Cumulative >=v40,1,0)
pt$v50 <- ifelse(pt$Cumulative >=v50,1,0)
pt$v60 <- ifelse(pt$Cumulative >=v60,1,0)
pt$v70 <- ifelse(pt$Cumulative >=v70,1,0)
pt$v80 <- ifelse(pt$Cumulative >=v80,1,0)
pt$v90 <- ifelse(pt$Cumulative >=v90,1,0)

summary(pt[,c("v10","v20","v30","v40","v50","v60","v70","v80","v90")])

Xs<-subset(colnames(pt),!(colnames(pt)%in%c("Cumulative","future_id","primary_id","emission_co2e_TOTAL_50","net_zero_flag","selected_flag","cluster_flag","cluster_representative_flag")))
Xs<-subset(Xs,grepl("v",Xs)==FALSE)
summary(pt[,Xs])

#now do the same with random forest 
library(rpart)       # performing regression trees
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(caret)       # bagging

#set params
set.seed(55555)
ctrl <- trainControl(method = "cv",  number = 7) 

#models
 model_1<-as.formula(paste("as.factor(v10)","~",paste(Xs,collapse="+"),sep=""))
 model_2<-as.formula(paste("as.factor(v20)","~",paste(Xs,collapse="+"),sep=""))
 model_3<-as.formula(paste("as.factor(v30)","~",paste(Xs,collapse="+"),sep=""))
 model_4<-as.formula(paste("as.factor(v40)","~",paste(Xs,collapse="+"),sep=""))
 model_5<-as.formula(paste("as.factor(v50)","~",paste(Xs,collapse="+"),sep=""))
 model_6<-as.formula(paste("as.factor(v60)","~",paste(Xs,collapse="+"),sep=""))
 model_7<-as.formula(paste("as.factor(v70)","~",paste(Xs,collapse="+"),sep=""))
 model_8<-as.formula(paste("as.factor(v80)","~",paste(Xs,collapse="+"),sep=""))
 model_9<-as.formula(paste("as.factor(v90)","~",paste(Xs,collapse="+"),sep=""))
 models <- list(model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8,model_9)

#random forest
rdms <- list()
for (i in 1:length(models))
{
# i<-5
 bagged_cv <- train(models[[i]], data    = pt, method = "treebag", trControl = ctrl, importance = TRUE)
 st<-subset(varImp(bagged_cv)$importance,Overall>= as.numeric(quantile(varImp(bagged_cv)$importance[,1],0.75))) 
 st$group<-row.names(st)
 colnames(st)<-gsub("Overall",paste0("Overall","_",i),colnames(st))
 row.names(st)<-NULL
 rdms <- append(rdms,list(st))
} 
rdms<- Reduce(function(...) merge(...,all=TRUE), rdms)
rdms$Score <- rowSums(apply(rdms[,subset(colnames(rdms),colnames(rdms)!="group")],c(1,2),function(x){ifelse(is.na(x)==TRUE,0,1)}))
#add specific group vars
tgroups <- as.numeric(gsub("X","",unique(rdms$group)))

all_vars <- list()
for (i in 1:length(tgroups))
{
#i<-2
vars<-data.frame(vars=paste(subset(key,sample_group==tgroups[i])$variable_specification,collapse=","),group=paste0("X",tgroups[i]))
all_vars <- append(all_vars,list(vars))
}
all_vars <- do.call("rbind",all_vars)

#merge 
dim(rdms)
rdms<-merge(rdms,all_vars,by="group")
dim(rdms)
write.csv(rdms,paste0(root,"va_analysis/random_forest_report_cost_benefits_entc.csv"),row.names=FALSE)
#writ the prim data  
write.csv(pt,"/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/Tableau/cb_entc.csv",row.names=FALSE)


#now what if we do all sectors, 

cb_prim1 <- subset(cb_data_raw,cb_type=="technical_cost")
summary(cb_prim1$Cumulative)

#add keys 
key <- read.csv(paste0(root,"variable_specification_to_sample_group.csv"))
#now merge with experimental features  
key_pt <- read.csv("/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample/pt.csv")


#firt we need to aggregate the data at future_id level 
cb_prim1 <- aggregate(list(Cumulative = cb_prim1$Cumulative),list(future_id=cb_prim1$future_id),sum, na.rm=TRUE)
summary(cb_prim1$Cumulative)

#now merge with pt 
#merge with cba 
  dim(cb_prim1)
  cb_prim1 <- merge(cb_prim1,key_pt, by="future_id")
  dim(cb_prim1)

#ok, now le'ts run our random forest analysis 
 pt <- cb_prim1
 v10<-as.numeric(quantile(pt$Cumulative,0.1))
 v20<-as.numeric(quantile(pt$Cumulative,0.2))
 v30<-as.numeric(quantile(pt$Cumulative,0.3))
 v40<-as.numeric(quantile(pt$Cumulative,0.4))
 v50<-as.numeric(quantile(pt$Cumulative,0.5))
 v60<-as.numeric(quantile(pt$Cumulative,0.6))
 v70<-as.numeric(quantile(pt$Cumulative,0.7))
 v80<-as.numeric(quantile(pt$Cumulative,0.8))
 v90<-as.numeric(quantile(pt$Cumulative,0.9))

#create flags 
pt$v10 <- ifelse(pt$Cumulative >=v10,1,0)
pt$v20 <- ifelse(pt$Cumulative >=v20,1,0)
pt$v30 <- ifelse(pt$Cumulative >=v30,1,0)
pt$v40 <- ifelse(pt$Cumulative >=v40,1,0)
pt$v50 <- ifelse(pt$Cumulative >=v50,1,0)
pt$v60 <- ifelse(pt$Cumulative >=v60,1,0)
pt$v70 <- ifelse(pt$Cumulative >=v70,1,0)
pt$v80 <- ifelse(pt$Cumulative >=v80,1,0)
pt$v90 <- ifelse(pt$Cumulative >=v90,1,0)

summary(pt[,c("v10","v20","v30","v40","v50","v60","v70","v80","v90")])

Xs<-subset(colnames(pt),!(colnames(pt)%in%c("Cumulative","future_id","primary_id","emission_co2e_TOTAL_50","net_zero_flag","selected_flag","cluster_flag","cluster_representative_flag")))
Xs<-subset(Xs,grepl("v",Xs)==FALSE)
summary(pt[,Xs])

#now do the same with random forest 
library(rpart)       # performing regression trees
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(caret)       # bagging

#set params
set.seed(55555)
ctrl <- trainControl(method = "cv",  number = 7) 

#models
 model_1<-as.formula(paste("as.factor(v10)","~",paste(Xs,collapse="+"),sep=""))
 model_2<-as.formula(paste("as.factor(v20)","~",paste(Xs,collapse="+"),sep=""))
 model_3<-as.formula(paste("as.factor(v30)","~",paste(Xs,collapse="+"),sep=""))
 model_4<-as.formula(paste("as.factor(v40)","~",paste(Xs,collapse="+"),sep=""))
 model_5<-as.formula(paste("as.factor(v50)","~",paste(Xs,collapse="+"),sep=""))
 model_6<-as.formula(paste("as.factor(v60)","~",paste(Xs,collapse="+"),sep=""))
 model_7<-as.formula(paste("as.factor(v70)","~",paste(Xs,collapse="+"),sep=""))
 model_8<-as.formula(paste("as.factor(v80)","~",paste(Xs,collapse="+"),sep=""))
 model_9<-as.formula(paste("as.factor(v90)","~",paste(Xs,collapse="+"),sep=""))
 models <- list(model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8,model_9)

#random forest
rdms <- list()
for (i in 1:length(models))
{
# i<-5
 bagged_cv <- train(models[[i]], data    = pt, method = "treebag", trControl = ctrl, importance = TRUE)
 st<-subset(varImp(bagged_cv)$importance,Overall>= as.numeric(quantile(varImp(bagged_cv)$importance[,1],0.75))) 
 st$group<-row.names(st)
 colnames(st)<-gsub("Overall",paste0("Overall","_",i),colnames(st))
 row.names(st)<-NULL
 rdms <- append(rdms,list(st))
} 
rdms<- Reduce(function(...) merge(...,all=TRUE), rdms)
rdms$Score <- rowSums(apply(rdms[,subset(colnames(rdms),colnames(rdms)!="group")],c(1,2),function(x){ifelse(is.na(x)==TRUE,0,1)}))
#add specific group vars
tgroups <- as.numeric(gsub("X","",unique(rdms$group)))

all_vars <- list()
for (i in 1:length(tgroups))
{
#i<-2
vars<-data.frame(vars=paste(subset(key,sample_group==tgroups[i])$variable_specification,collapse=","),group=paste0("X",tgroups[i]))
all_vars <- append(all_vars,list(vars))
}
all_vars <- do.call("rbind",all_vars)

#merge 
dim(rdms)
rdms<-merge(rdms,all_vars,by="group")
dim(rdms)
write.csv(rdms,paste0(root,"va_analysis/random_forest_report_cost_benefits_all_sectors.csv"),row.names=FALSE)
#writ the prim data  
write.csv(pt,"/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/Tableau/cb_all.csv",row.names=FALSE)

