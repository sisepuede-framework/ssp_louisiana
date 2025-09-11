
#
root <- "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/59. Lousiana Project/LA_CaseStudy/LHC_sample/2025_02_11/"
file.name <- "sisepuede_results_sisepuede_run_2025-01-14T17;04;06.975301_WIDE_INPUTS_OUTPUTS.csv"
full_sim <- read.csv(paste0(root,file.name))

all_ids <- unique(full_sim$primary_id)

for (i in 1:length(all_ids))
{
#i<-1 
pivot <- subset(full_sim,primary_id==all_ids[i])
write.csv(pivot,paste0(root,"meta/",i,".csv"),row.names=FALSE)
}

