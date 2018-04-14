# rando
library(data.table)
library(lubridate)
library(dplyr)
library(date)
library(doMC)

air_visit <- fread("air_visit_data.csv")
air_visit$visit_date <- as.Date(parse_date_time(air_visit$visit_date,'%y-%m-%d'))

### train_sub -  use 2017 data to start with
train_sub <- air_visit[visit_date >= as.Date("2017-01-01"), ]
train_sub_wide <- dcast(
  train_sub, air_store_id ~ visit_date, value.var = "visitors", fill = 0)
train_ts <- ts(train_sub_wide[, 2:dim(train_sub_wide)[2]], frequency = 7) 
fcst_intv = 39  ### 39 days of forecast horizon
fcst_matrix <- matrix(NA,nrow=4*nrow(train_ts),ncol=fcst_intv)

### register cores for parallel processing in forecasting
registerDoMC(detectCores()-1)
fcst_matrix <- foreach(i=1:nrow(train_ts),.combine=rbind, .packages=c("forecast")) %dopar% { 
  fcst_ets <- forecast(ets(train_ts[i,]),h=fcst_intv)$mean
  fcst_nnet <- forecast(nnetar(train_ts[i,]),h=fcst_intv)$mean
  fcst_arima <- forecast(auto.arima(train_ts[i,]),h=fcst_intv)$mean
  fcst_ses <- forecast(HoltWinters(train_ts[i,], beta=FALSE, gamma=FALSE),h=fcst_intv)$mean
  fcst_matrix <- rbind(fcst_ets, fcst_nnet, fcst_arima, fcst_ses)
}

### sample code to extract single method forecasting 
# index_arima <- seq(3, nrow(fcst_matrix), by = 4)
# fcst_matrix_arima <- fcst_matrix[index_arima,] 

### mix 4 forecasting method results
fcst_matrix_mix <- aggregate(fcst_matrix,list(rep(1:(nrow(fcst_matrix)/4),each=4)),mean)[-1]

### post-processing the forecast table
fcst_matrix_mix[fcst_matrix_mix < 0] <- 0
colnames(fcst_matrix_mix) <- as.character(
  seq(from = as.Date("2017-04-23"), to = as.Date("2017-05-31"), by = 'day'))
fcst_df <- as.data.frame(cbind(train_sub_wide[, 1], fcst_matrix_mix)) 
colnames(fcst_df)[1] <- "air_store_id"

### melt the forecast data frame from wide to long format for final submission
fcst_df_long <- melt(
  fcst_df, id = 'air_store_id', variable.name = "fcst_date", value.name = 'visitors')
fcst_df_long$air_store_id <- as.character(fcst_df_long$air_store_id)
fcst_df_long$fcst_date <- as.Date(parse_date_time(fcst_df_long$fcst_date,'%y-%m-%d'))
fcst_df_long$visitors <- as.numeric(fcst_df_long$visitors)

### get & process the sample submission file
sample_sub <- fread("sample_submission.csv")
sample_sub$visitors <- NULL
sample_sub$store_id <- substr(sample_sub$id, 1, 20)
sample_sub$visit_date <- substr(sample_sub$id, 22, 31)
sample_sub$visit_date <- as.Date(parse_date_time(sample_sub$visit_date,'%y-%m-%d'))

### generate the final submission file
submission <- left_join(
  sample_sub, fcst_df_long, c("store_id" = "air_store_id", 'visit_date' = 'fcst_date'))
submission$visitors[is.na(submission$visitors)] <- 0
final_sub <- select(submission, c('id', 'visitors'))
write.csv(final_sub, "sub_ts_mix.csv", row.names = FALSE)