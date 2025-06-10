#integrate all files 
root <- "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/"
data <- read.csv(paste0(root,"Emission_Trajectories/","sisepuede_results_sisepuede_run_ssp.csv"))

tv1_all <- subset(colnames(data),grepl("co2e_",colnames(data))==TRUE)
data$emission_co2e_TOTAL <- rowSums(data[,tv1_all])
data$Index<-paste0(data$primary_id,"_",data$region)

#definte subsector totals
sector_totals <-c("emission_co2e_subsector_total_agrc", 
                                                "emission_co2e_subsector_total_ccsq",
                                                "emission_co2e_subsector_total_entc",
                                                "emission_co2e_subsector_total_fgtv",
                                                "emission_co2e_subsector_total_frst",
                                                "emission_co2e_subsector_total_inen",
                                                "emission_co2e_subsector_total_ippu",
                                                "emission_co2e_subsector_total_lndu",
                                                "emission_co2e_subsector_total_lsmm",
                                                "emission_co2e_subsector_total_lvst",
                                                "emission_co2e_subsector_total_scoe",
                                                "emission_co2e_subsector_total_soil",
                                                "emission_co2e_subsector_total_trns",
                                                "emission_co2e_subsector_total_trww",
                                                "emission_co2e_subsector_total_waso")


#create file for prim boxes, emission targerts 
data_0<-subset(data,time_period==0)
data_0 <- data_0[,c("primary_id","region","emission_co2e_TOTAL")]
colnames(data_0) <- gsub("emission_co2e_TOTAL","emission_co2e_TOTAL_0",colnames(data_0))
data_50<-subset(data,time_period==35)
data_50 <- data_50[,c("primary_id","region","emission_co2e_TOTAL")]
colnames(data_50) <- gsub("emission_co2e_TOTAL","emission_co2e_TOTAL_50",colnames(data_50))
dim(data_50)
#merge both
dim(data_50)
data_50 <- merge(data_50,data_0,by=c("primary_id","region"))
dim(data_50)

#percent reduction 
data_50$reduction <- abs(data_50$emission_co2e_TOTAL_50)/data_50$emission_co2e_TOTAL_0
#threshold 
th<-0.25
data_50$net_zero_cases <- ifelse(data_50$reduction<= th,1,0)

#now for every country compute quantile 25
data_50new<-list()
for (i in 1:length(unique(data_50$region)))
{
#i<-1
pivot<-subset(data_50,region==unique(data_50$region)[i])
pivot$net_zero_cases <- ifelse(pivot$reduction<=quantile(pivot$reduction,th),1,0)
data_50new<-append(data_50new,list(pivot))
}
data_50new <- do.call("rbind",data_50new)
dim(data_50new)
dim(data_50)

data50 <- data_50new

#aggregate  
targets<-aggregate(data_50[,c("emission_co2e_TOTAL_50","net_zero_cases")],list(primary_id=data_50$primary_id),sum)
#region
targets$region_net_zero_case <- ifelse(abs(targets$emission_co2e_TOTAL_50)>=100,0,1)
#
summary(targets$net_zero_cases)

targets$targets <- ifelse(targets$net_zero_cases>=quantile(targets$net_zero_cases,0.80) & targets$region_net_zero_case >=0.5,1.0,0.0)
otargets<-targets
#write.csv(targets,paste0(root,"out\\","lever_targets.csv"),row.names=FALSE)

#try to cluster the resulting ids 
#load attribute ids 
 att <- read.csv("/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample/ATTRIBUTE_PRIMARY.csv")
 dim(targets)
 targets <- merge(targets,att[,c("primary_id","future_id")],by="primary_id")
 dim(targets)

#load lhs draws
 lhs <- read.csv("/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample/lhc_trials_lever_effects_by_sample_group.csv")
 lhs_0<-subset(lhs,future_id==1)
 lhs_0[,subset(colnames(lhs),colnames(lhs)!="future_id")]<-0.5
 lhs_0$future_id <- 0
 lhs <-  rbind(lhs,lhs_0)


 dim(targets)
 targets <- merge(targets,lhs,by="future_id")
 dim(targets)

#subset targets to selected targets  
 kgroups <- subset(targets,targets==1) 
 dim(kgroups)
 kvars<-subset(colnames(kgroups),!(colnames(kgroups)%in%c("future_id","primary_id","emission_co2e_TOTAL_50","net_zero_cases","region_net_zero_case" ,"targets")))
 
 set.seed(55555)
# exp<-expand.grid(nstart=c(20,50,100),centers=c(2,3,4,5,6,7,8,9,10))
exp<-expand.grid(nstart=c(20,50,100),centers=c(4,5,6,7,8,9,10))
 out<-apply(exp,1,function(x){kmeans_model <-kmeans(kgroups[,kvars], centers = as.numeric(x['centers']), nstart= as.numeric(x['nstart']));
                         data.frame(nstart=as.numeric(x['nstart']),centers=as.numeric(x['centers']), withinss=sum(as.numeric(kmeans_model$withinss )))})
 out <- do.call("rbind",out)
 out$id<-1:nrow(out)
 out <- out[order(out$withinss),]
 out$pct_withinss <- 1- out$withinss/max(out$withinss)
 out$diff<-c(diff(out$withinss),0)
# write.csv(out,paste0(root,"out\\","cluster_error.csv"),row.names=FALSE)
#after inspection we decide to keep 5 clusters 
 kmeans_model <-kmeans(kgroups[,kvars], centers = 5, nstart=50)
#assing observations to groups 
 kgroups$cluster <- kmeans_model$cluster
#now find the euclidian distance to centroids 
 dim(kgroups)
kgroups$euclidian <- 0
for (i in 1:nrow(kgroups))
{
 kgroups$euclidian[i] <- (sum((kgroups[i,kvars]-kmeans_model$centers[kgroups$cluster[i],kvars])^2))^0.5 
}
#based on euclidian distance find the id closest to the centroid
kgroups$Run <- 1000
for (i in 1:nrow(kgroups))
{ 
# i<-1
 min_dist <- min(subset(kgroups,cluster==kgroups$cluster[i])$euclidian)
 kgroups$Run[i] <- ifelse(kgroups$euclidian[i]==min_dist,1,0)
}

#merge with otargets 
head(otargets)
dim(otargets)
otargets<-merge(otargets,kgroups[,c("primary_id","cluster","Run")],all.x=TRUE)
dim(otargets)

#append to edit 
otargets$cluster<-ifelse(is.na(otargets$cluster)==TRUE,10,otargets$cluster)
otargets$Run<-ifelse(is.na(otargets$Run)==TRUE,0,otargets$Run)
summary(otargets)
#write.csv(otargets,paste0(root,"out\\","lever_targets.csv"),row.names=FALSE)

#now create a table for vulnerability analysis  
pt<-otargets[,c("primary_id","emission_co2e_TOTAL_50","cluster","Run")]
pt$net_zero_flag <- ifelse(abs(pt$emission_co2e_TOTAL_50) >= 100, 0,1)
pt$selected_flag <- ifelse(pt$cluster < 10,1,0)
pt$cluster_flag <- pt$cluster
pt$cluster_representative_flag <- pt$Run
#select variables of interest 
pt <- pt [,c("primary_id","emission_co2e_TOTAL_50","net_zero_flag","selected_flag","cluster_flag","cluster_representative_flag")]
#add LHS design 
dim(pt)
pt <- merge(pt,att[,c("primary_id","future_id")],by="primary_id")
dim(pt)

#load lhs draws
dim(pt)
pt <- merge(pt,lhs,by="future_id")
dim(pt)

#write the file  
write.csv(pt,paste0("/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample/","pt.csv"),row.names=FALSE)

