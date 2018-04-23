library(keras)
library(data.table)
library(tidyverse)
library(reticulate)
use_condaenv("r-reticulate")

d <- fread("../data/adfraud/train.csv",
                       drop = c("attributed_time"),
                       colClasses=list(numeric=1:5),  
                       col.names = c("ip", "app", "device", "os", "channel", "click_time", "is_attributed")
                 )[(.N - 50e6):.N] 



d[, yday := yday(click_time)]
d[, hour := hour(click_time)]
d[, day := wday(click_time)]

# d[, uniqueN(channel) , by = .(ip, hour, day)]

d[, qty :=  .N, by = .(ip, day, hour) ]
d[, ip_app_count := .N , by = .(ip, app)]
d[, ip_app_os_count := .N , by = .(ip, app, os, channel)]


drop_colz <- c("click_time", "ip")
d[, (drop_colz) := NULL]

ls() %>% str_subset("emb|in_|pred|max") -> del
rm(list = del)

emb_n <- 50
dense_n <- 1000

# in app
max_app <- d[,max(app)] + 1L

in_app <- layer_input(shape = 1L, name = 'app')
emb_app <- in_app %>% layer_embedding(max_app, emb_n)

# in channel
max_ch <- d[,max(channel)] + 1L
in_ch <- layer_input(shape = 1L, name = 'ch') 
emb_ch <-  in_ch %>% layer_embedding(max_ch, emb_n)

# in dev
max_dev <- d[, max(device)] + 1L
in_dev <- layer_input(shape = 1L, name = 'dev') 
emb_dev <- in_dev %>% layer_embedding(max_dev, emb_n)


# in os
max_os <- d[,max(os)] + 1L

in_os <- layer_input(shape = 1L, name = 'os')
emb_os <- in_os %>%  layer_embedding(max_os, emb_n)


# in hr
max_hr <- d[,max(hour)] + 1L
in_hr <-  layer_input(shape = 1L, name = 'h') 
emb_hr <- in_hr %>% layer_embedding(max_hr, emb_n)


# in yday
max_yday <-  d[,max(yday)] + 1L
in_yrday <- layer_input(shape = 1L, name = 'yday')
emb_yrday <- in_yrday %>% layer_embedding(max_yday, emb_n)

# in day
max_day <- d[,max(day)] + 1L
in_wday <- layer_input(shape = 1L, name = 'wday')  
emb_wday <- in_wday %>% layer_embedding(max_day, emb_n)

# in qty
max_qty <- d[,max(qty)] + 1L
in_qty <- layer_input(shape = 1L, name = 'qty')  
emb_qty <-  in_qty %>% layer_embedding(max_qty, emb_n)

# max_c1 
max_c1 <- d[,max(ip_app_count)] + 1L
in_c1 <- layer_input(shape = 1L, name = 'c1') 
emb_c1 <- in_c1 %>% layer_embedding(max_c1, emb_n)

# max_c2 
max_c2 <- d[,max(ip_app_os_count)] + 1L
in_c2 <- layer_input(shape = 1L, name = 'c2')
emb_c2 <- in_c2   %>% layer_embedding(max_c2, emb_n)


fe  <- layer_concatenate(
  list(
    emb_app,  emb_dev, emb_os , emb_ch, emb_yrday, emb_hr,  emb_wday,
    emb_qty, emb_c1, emb_c2
  )
)

predictions <- fe %>% 
    layer_spatial_dropout_1d(0.2) %>% 
    layer_flatten() %>% 
    layer_dense(units=dense_n, activation = 'relu') %>% 
    layer_dropout(0.2) %>% 
    layer_dense(units = dense_n, activation = 'relu') %>% 
    layer_dropout(0.2) %>% 
    layer_dense(units = 1, activation = 'sigmoid')


model <- keras_model(inputs = list(
  in_app,in_dev,in_os,in_ch,in_yrday,in_hr,in_wday,in_qty,in_c1,in_c2
  ), outputs = predictions)

batch_size = 20000
epochs = 2


  model %>% 
    compile(optimizer = "adam",
              loss = 'binary_crossentropy',
              metrics = c("accuracy")
      
    )
  
  
 d[, rn := .I - 1L]  

 set.seed(123)
 train <- d[sample(40000000)]  
 test <- d[!rn %in% test[,rn] ]
 rm(d) ; gc()
 test_y <-  test[,is_attributed]
 train_y <- train[,is_attributed]
 
 test[, is_attributed := NULL]
 train[, is_attributed := NULL]
 
 train[, rn:=NULL]
 
 
  history <- fit(
    object = model, 
    x = list(
      train$app, train$device, train$os, train$channel, train$hour,
      train$yday, train$day, train$qty, train$ip_app_count, 
      train$ip_app_count
    ),
    y = train_y,
    batch_size = 20000,
    verbose = 2, 
    shufflue = TRUE,
    epochs = 2 ,
    validation_split = 0.2
    )


  
test[, rn := NULL]
  
  
test_preds <- predict(model, list(
    test$app, test$device, test$os, test$channel, test$hour,
    test$yday, test$day, test$qty, test$ip_app_count, 
    test$ip_app_count
  ), batch_size=batch_size, verbose=2) %>% 
  as.vector()



predictions <- tibble(
  probs = test_preds,
  is_attributed = ifelse(probs > 0.5, 1, 0)
)


# Format test data and predictions for yardstick metrics
estimates <- tibble(
  truth      = as.factor(test_y) ,
  estimate   = as.factor(predictions$is_attributed) ,
  class_prob = predictions$probs
)

library(yardstick)

tibble(
  accuracy = estimates %>% metrics(truth, estimate) %>% pull(accuracy),
  precision = estimates %>% precision(truth, estimate),
  recall    = estimates %>% recall(truth, estimate),
 # auc = estimates %>% roc_auc(truth, class_prob),
  f1_statistic = estimates %>% f_meas(truth, estimate, beta = 1)
)


### predict on kaggle test set


test <- fread("../data/adfraud/test.csv",showProgress = FALSE)



test[, yday := yday(click_time)]
test[, hour := hour(click_time)]
test[, day := wday(click_time)]

# d[, uniqueN(channel) , by = .(ip, hour, day)]

test[, qty :=  .N, by = .(ip, day, hour) ]
test[, ip_app_count := .N , by = .(ip, app)]
test[, ip_app_os_count := .N , by = .(ip, app, os, channel)]


drop_colz <- c("click_time", "ip")
test[, (drop_colz) := NULL]
test[, click_id := NULL]

kaggle_preds <- predict(model, list(
  test$app, test$device, test$os, test$channel, test$hour,
  test$yday, test$day, test$qty, test$ip_app_count, 
  test$ip_app_count
), batch_size=batch_size, verbose=2) %>% 
  as.vector()

test_preds <- data.table(
  probs = kaggle_preds
)

test_preds[, is_attributed := ifelse(probs > 0.5, 1L, 0L)]
test_preds[, click_id := .I - 1L]
test_preds %>% select(click_id, is_attributed) %>% fwrite("../data/adfraud/sub2.csv")
