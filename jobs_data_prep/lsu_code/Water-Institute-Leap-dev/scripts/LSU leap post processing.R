primaries <- unique(shocks$power$primary_id[shocks$power$primary_id != 0]) #primary_id represents scenarios, 0 is baseline
vars_to_keep<-c('la_value_direct',
                'la_value_indirect',
                'la_value_induced',
                'la_earnings_direct',
                'la_earnings_indirect',
                'la_earnings_induced',
                'la_employment_direct',
                'la_employment_indirect',
                'la_employment_induced',
                'la_employment_total')

ccs_output<-data.frame()

for(id in primaries){
  temp<-data.frame(time=c(6:35))
  temp$primary_id<-id
  if(length(ccs_diff[[as.character(id)]])>0){
    temp<-merge(temp, ccs_diff[[as.character(id)]][c('time', vars_to_keep)], all.x=T)
    temp[is.na(temp)]<-0
  }
  else{
    temp[vars_to_keep]<-0
  }
  ccs_output<-rbind(ccs_output, temp)
}

power_output<-data.frame()

for(id in primaries){
  temp<-data.frame(time=c(6:35))
  temp$primary_id<-id
  if(length(power_diff[[as.character(id)]])>0){
    temp<-merge(temp, power_diff[[as.character(id)]][c('time', vars_to_keep)], all.x=T)
    temp[is.na(temp)]<-0
  }
  else{
    temp[vars_to_keep]<-0
  }
  power_output<-rbind(power_output, temp)
}

industry_output<-data.frame()

for(id in primaries){
  temp<-data.frame(time=c(6:35))
  temp$primary_id<-id
  if(length(industry_diff[[as.character(id)]])>0){
    temp<-merge(temp, industry_diff[[as.character(id)]][c('time', vars_to_keep)], all.x=T)
    temp[is.na(temp)]<-0
  }
  else{
    temp[vars_to_keep]<-0
  }
  industry_output<-rbind(industry_output, temp)
}

output_all<-as.matrix(industry_output[vars_to_keep])+as.matrix(ccs_output[vars_to_keep]+as.matrix(power_output[vars_to_keep]))

output<-cbind(ccs_output[,-which(colnames(ccs_output) %in% vars_to_keep)], output_all)

power_sector_direct_salary<-mean(baseline$power_impact$la_earnings_direct/baseline$power_impact$la_employment_direct)
power_sector_indirect_salary<-mean(baseline$power_impact$la_earnings_indirect/baseline$power_impact$la_employment_indirect)
industry_sector_direct_salary<-mean(baseline$industry_impact$la_earnings_direct/baseline$industry_impact$la_employment_direct)
industry_sector_indirect_salary<-mean(baseline$industry_impact$la_earnings_indirect/baseline$industry_impact$la_employment_indirect)
induced_salary<-mean(baseline$power_impact$la_earnings_induced/baseline$power_impact$la_employment_induced)

total_new_jobs<-sapply(power_output$la_employment_induced, max, 0)+
  sapply(power_output$la_employment_indirect, max, 0)+
  sapply(power_output$la_employment_direct, max, 0)+
  sapply(industry_output$la_employment_induced, max, 0)+
  sapply(industry_output$la_employment_indirect, max, 0)+
  sapply(industry_output$la_employment_direct, max, 0)+
  sapply(ccs_output$la_employment_induced, max, 0)+
  sapply(ccs_output$la_employment_indirect, max, 0)+
  sapply(ccs_output$la_employment_direct, max, 0)
  
total_new_earnings<-sapply((induced_salary*power_output$la_employment_induced), max, 0)+
  sapply((power_sector_indirect_salary*power_output$la_employment_indirect), max, 0)+
  sapply((power_sector_direct_salary*power_output$la_employment_direct), max, 0)+
  sapply((induced_salary*industry_output$la_employment_induced), max, 0)+
  sapply((industry_sector_indirect_salary*industry_output$la_employment_indirect), max, 0)+
  sapply((industry_sector_direct_salary*industry_output$la_employment_direct), max, 0)+
  sapply((induced_salary*ccs_output$la_employment_induced), max, 0)+
  sapply((industry_sector_indirect_salary*ccs_output$la_employment_indirect), max, 0)+
  sapply((industry_sector_direct_salary*ccs_output$la_employment_direct), max, 0)

output$average_salary_jobs_added<-total_new_earnings/total_new_jobs
output$average_salary_jobs_added[is.na(output$average_salary_jobs_added)]<-0

write.csv(output, 'lsu_output_1000_ensemble_10_22.csv', row.names=F)



