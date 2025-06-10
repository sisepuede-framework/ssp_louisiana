
root <- "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample/"
pt <- read.csv(paste0(root,"pt.csv"))
key <- read.csv(paste0(root,"variable_specification_to_sample_group.csv"))


#see range of emissions above net zero 
summary(subset(pt,emission_co2e_TOTAL_50>=0)$emission_co2e_TOTAL_50)
v10<-as.numeric(quantile(subset(pt,emission_co2e_TOTAL_50>=0)$emission_co2e_TOTAL_50,0.1))
v20<-as.numeric(quantile(subset(pt,emission_co2e_TOTAL_50>=0)$emission_co2e_TOTAL_50,0.2))
v30<-as.numeric(quantile(subset(pt,emission_co2e_TOTAL_50>=0)$emission_co2e_TOTAL_50,0.3))
v40<-as.numeric(quantile(subset(pt,emission_co2e_TOTAL_50>=0)$emission_co2e_TOTAL_50,0.4))
v50<-as.numeric(quantile(subset(pt,emission_co2e_TOTAL_50>=0)$emission_co2e_TOTAL_50,0.5))
v60<-as.numeric(quantile(subset(pt,emission_co2e_TOTAL_50>=0)$emission_co2e_TOTAL_50,0.6))
v70<-as.numeric(quantile(subset(pt,emission_co2e_TOTAL_50>=0)$emission_co2e_TOTAL_50,0.7))
v80<-as.numeric(quantile(subset(pt,emission_co2e_TOTAL_50>=0)$emission_co2e_TOTAL_50,0.8))
v90<-as.numeric(quantile(subset(pt,emission_co2e_TOTAL_50>=0)$emission_co2e_TOTAL_50,0.9))

#create flags 
pt$v0 <- ifelse(pt$emission_co2e_TOTAL_50 >0,1,0)
pt$v10 <- ifelse(pt$emission_co2e_TOTAL_50 >=v10,1,0)
pt$v20 <- ifelse(pt$emission_co2e_TOTAL_50 >=v20,1,0)
pt$v30 <- ifelse(pt$emission_co2e_TOTAL_50 >=v30,1,0)
pt$v40 <- ifelse(pt$emission_co2e_TOTAL_50 >=v40,1,0)
pt$v50 <- ifelse(pt$emission_co2e_TOTAL_50 >=v50,1,0)
pt$v60 <- ifelse(pt$emission_co2e_TOTAL_50 >=v60,1,0)
pt$v70 <- ifelse(pt$emission_co2e_TOTAL_50 >=v70,1,0)
pt$v80 <- ifelse(pt$emission_co2e_TOTAL_50 >=v80,1,0)
pt$v90 <- ifelse(pt$emission_co2e_TOTAL_50 >=v90,1,0)
pt$vulnerability <- ifelse(pt$emission_co2e_TOTAL_50 >=70,1,0)

#summmaries  
summary(pt[,c("v10","v20","v30","v40","v50","v60","v70","v80","v90","vulnerability")])
summary(pt$vulnerability)
summary(pt$net_zero_flag)

Xs<-subset(colnames(pt),!(colnames(pt)%in%c("cb_value","future_id","primary_id","emission_co2e_TOTAL_50","net_zero_flag","selected_flag","cluster_flag","cluster_representative_flag","vulnerability")))
Xs<-subset(Xs,grepl("v",Xs)==FALSE)
summary(pt[,Xs])

#models
 model_1<-as.formula(paste("v10","~",paste(Xs,collapse="+"),sep=""))
 model_2<-as.formula(paste("v20","~",paste(Xs,collapse="+"),sep=""))
 model_3<-as.formula(paste("v30","~",paste(Xs,collapse="+"),sep=""))
 model_4<-as.formula(paste("v40","~",paste(Xs,collapse="+"),sep=""))
 model_5<-as.formula(paste("v50","~",paste(Xs,collapse="+"),sep=""))
 model_6<-as.formula(paste("v60","~",paste(Xs,collapse="+"),sep=""))
 model_7<-as.formula(paste("v70","~",paste(Xs,collapse="+"),sep=""))
 model_8<-as.formula(paste("v80","~",paste(Xs,collapse="+"),sep=""))
 model_9<-as.formula(paste("v90","~",paste(Xs,collapse="+"),sep=""))

models <- list(model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8,model_9)
# model_10<-as.formula(paste("vulnerability","~",paste(Xs,collapse="+"),sep=""))

#build summary report 

#linear regression
lms <- list()
for (i in 1:length(models))
{
 #i<-1
 lm.fit<-lm(models[[i]], data=pt)
 st <-  data.frame(summary(lm.fit)$coefficients)
 st <- subset(st,st[,4]<=0.05)
 st$group<-row.names(st)
 st <- subset(st,group!="(Intercept)")
 st <- st[,c("group","Estimate")]
 colnames(st)<-gsub("Estimate",paste0("Estimate","_",i),colnames(st))
 row.names(st)<-NULL
 lms <- append(lms,list(st))
} 
lms<- Reduce(function(...) merge(...,all=TRUE), lms)
lms$Score <- rowSums(apply(lms[,subset(colnames(lms),colnames(lms)!="group")],c(1,2),function(x){ifelse(is.na(x)==TRUE,0,1)}))
#add specific group vars
tgroups <- as.numeric(gsub("X","",unique(lms$group)))

all_vars <- list()
for (i in 1:length(tgroups))
{
#i<-2
vars<-data.frame(vars=paste(subset(key,sample_group==tgroups[i])$variable_specification,collapse=","),group=paste0("X",tgroups[i]))
all_vars <- append(all_vars,list(vars))
}
all_vars <- do.call("rbind",all_vars)

#merge 
dim(lms)
lms<-merge(lms,all_vars,by="group")
dim(lms)
write.csv(lms,paste0(root,"va_analysis\\lms_report.csv"),row.names=FALSE)
#now let's create a column with the actual group names  

#logistic regression
glms <- list()
for (i in 1:length(models))
{
 #i<-1
 glm.fit<-glm(models[[i]],data=pt,family=binomial)
 st <-  data.frame(summary(glm.fit)$coefficients)
 st <- subset(st,st[,4]<=0.05)
 st$group<-row.names(st)
 st <- subset(st,group!="(Intercept)")
 st <- st[,c("group","Estimate")]
 colnames(st)<-gsub("Estimate",paste0("Estimate","_",i),colnames(st))
 row.names(st)<-NULL
 glms <- append(glms,list(st))
} 
glms<- Reduce(function(...) merge(...,all=TRUE), glms)
glms$Score <- rowSums(apply(glms[,subset(colnames(glms),colnames(glms)!="group")],c(1,2),function(x){ifelse(is.na(x)==TRUE,0,1)}))
#add specific group vars
tgroups <- as.numeric(gsub("X","",unique(glms$group)))

all_vars <- list()
for (i in 1:length(tgroups))
{
#i<-2
vars<-data.frame(vars=paste(subset(key,sample_group==tgroups[i])$variable_specification,collapse=","),group=paste0("X",tgroups[i]))
all_vars <- append(all_vars,list(vars))
}
all_vars <- do.call("rbind",all_vars)

#merge 
dim(glms)
glms<-merge(glms,all_vars,by="group")
dim(glms)
write.csv(glms,paste0(root,"va_analysis\\logistic_report.csv"),row.names=FALSE)

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
# i<-3
 bagged_cv <- train(models[[i]], data    = pt, method = "treebag", trControl = ctrl, importance = TRUE)
 st<-subset(varImp(bagged_cv)$importance,Overall>=50)  
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
write.csv(rdms,paste0(root,"va_analysis\\random_forest_report.csv"),row.names=FALSE)
