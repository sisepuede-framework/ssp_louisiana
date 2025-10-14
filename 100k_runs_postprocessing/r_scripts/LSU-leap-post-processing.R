primaries <- unique(shocks$power$primary_id[shocks$power$primary_id != 0]) #primary_id represents scenarios, 0 is baseline
vars_to_keep<-c('la_value_direct',
                'la_value_indirect',
                'la_value_induced',
                'la_earnings_direct',
                'la_earnings_indirect',
                'la_earnings_induced',
                'la_employment_direct',
                'la_employment_indirect',
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
write.csv(output, 'jobs_data_prep/lsu_code/Water-Institute-Leap-dev/output/lsu_output_1000_ensemble.csv', row.names=F)


