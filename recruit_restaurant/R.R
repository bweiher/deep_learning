library(tidyverse)
library(stringr)
library(prophet)
library(data.table)
library(lubridate)
library(bwmisc) # devtools::install_github('baweiher/bwmisc')
library(caret)

# prophet ------

sub <- read_csv('sample_submission.csv') %>% 
mutate(
  store_id = substr(id, 0,20),
  date = lubridate::ymd(substr(id , 22, 31))
)

c(min(sub$date), max(sub$date))
  
  
d <- read_csv('air_visit_data.csv')

holidays <- read_csv('date_info.csv') %>% 
  select(ds = calendar_date, holiday = holiday_flg) %>% 
  filter(holiday==1L) %>% 
  mutate(holiday = as.character(holiday))


stores <- unique(d$air_store_id)


c(max(d$visit_date), min(d$visit_date))



#predictions <- map_df(stores, function(x){
dlist <- list()
for(g in seq_along(stores)) {
  
  store <- stores[g]
  
data <- filter(d, air_store_id == store) %>% 
    select(y = visitors, ds = visit_date)

m <- prophet(df = data, 
             holidays = holidays, 
             yearly.seasonality=TRUE,
             daily.seasonality=TRUE)
  
future <- make_future_dataframe(m, periods = 40)
forecast <- predict(m, future)

p <- forecast %>%
  filter(ds > '2017-04-22') %>% 
  transmute(visit_date = ds, visits = yhat, store_id = store)

dlist[[g]] <- p

message(g / length(stores))
}


predictions <- bind_rows(dlist) %>% 
  as_tibble()


submission <-   select(sub, id) %>% 
  left_join(mutate(predictions, id = paste(store_id, visit_date, sep='_'))) %>% 
  select(id, visitors=visits)

submission %>%
   mutate(visitors = ifelse(is.na(visitors), 0, visitors),
          visitors = ifelse(visitors<0, 0, visitors)) %>% 
   write_csv('submission.csv')



# gbm ---- 

csvs <- str_subset(list.files(), '\\.csv$')
csv_names <- str_replace_all(csvs,'\\.csv','')
                   

dlist <- list()

for(g in seq_along(csvs)) {
  dlist[[csv_names[g]]] <- read_csv(csvs[g])
}

hr <- dlist[['hpg_reserve']] %>% 
  inner_join(dlist[['store_id_relation']])

ar <- dlist[['air_reserve']]

for( g in seq_along(list(ar,hr))) {
   list(ar,hr)[[g]]
}
  
