# TODO 
# ~ hyper param tuning
# ~ cross validation
library(keras)
library(data.table)
library(tidyverse)
library(yardstick)
library(fasttime)
library(reticulate)
library(caret)
# library(R6)
# library(lightgbm)
# library(recipes)

# try Keras or lightgbm!


# predict_for_kaggle <- FALSE

FLAGS <- flags(
  flag_numeric("n", 1024),
  flag_numeric("emb_n", 50)
)


start_time <- Sys.time()
use_condaenv("tf")


#setwd('deep_learning/adfraud/')
# read in train and test datasets
train <- fread("../data/adfraud/train.csv",
                       header=TRUE, 
                       showProgress = FALSE,
                       drop = c("attributed_time"),
                       colClasses = c(rep("integer", 5), rep("character",2), "integer"),
                       col.names = c("ip", "app", "device", "os", "channel", "click_time", "is_attributed")
                 )[121886955L:.N]

#train <- train[sample(10000)]
#d <- train

test <- fread("../data/adfraud/test.csv", 
              header=TRUE,
              showProgress = FALSE,
              drop = c("click_id"),
              colClasses = c(rep("integer", 6), "character"),
              col.names = c("ip", "app", "device", "os", "channel", "click_time")
              )[, is_attributed := NA_integer_]


# combine them and clear 
d <- rbindlist(list(test,train))
rm(test, train) ; gc()


# transform and build features
d[, click_time := fastPOSIXct(click_time,"GMT")]
d[, hour := hour(click_time)]
d[, day := wday(click_time)]
d[, yday := yday(click_time) ]
d[, qty :=  .N, by = .(ip, yday, hour) ]
d[, ip_app_count := .N , by = .(ip, app)]
d[, ip_app_os_count := .N , by = .(ip, app, os)]
#d[order(click_time), next_click_time :=  shift()]
# next clicktime DT feature

# drop redundant cols
drop_colz <- c("click_time", "ip")
d[, (drop_colz) := NULL]



# use labelencoder from scikit learn
pd <- import("pandas")
pp <- import("sklearn.preprocessing")
#  from sklearn.preprocessing import LabelEncoder

enc <- pp$LabelEncoder()
xformer <- enc$fit_transform

d[, app := xformer(app)]
d[, device := xformer(device)]
d[, os := xformer(os)]
d[, channel := xformer(channel)]
d[, hour := xformer(hour)]



# d[, (colnames(df)) := NULL]



# re-separate dfs 
train <- d[!is.na(is_attributed)]



# lightgbm  #### 
#run_light_gbm <- TRUE
#beepr::beep()

# if(isTRUE(run_light_gbm)){}
# 
# 
# cats <- c("app", "device", "os", "channel", "hour")
# 
# training <- as.matrix(training)
# training_data <- lgb.Dataset(data = training, 
#                              label = training_y, 
#                      categorical_feature = cats)
# 
# 
# cat("Creating the 'dvalid' for modeling...")
# validation <- as.matrix(validation)
# validation_data <- lgb.Dataset(data = validation, 
#                      label = validation_y, 
#                      categorical_feature = cats)
# 
# 
# # lgb.Dataset.save(training_data, "training.buffer")  
# #lgb.Dataset.save(validation_data, "validation.buffer")
# 
# # rm(training, validation) ; gc()
# # params for model
# 
# params <- list(objective = "binary", 
#               metric = "auc", 
#               learning_rate= 0.1,
#               num_leaves= 7,
#               max_depth= 4,
#               device = "gpu",
#               gpu_platform_id = 1,
#               gpu_device_id = 1,
#               nthread = 1,
#               min_child_samples= 100,
#               max_bin= 100,
#               subsample= 0.7, 
#               subsample_freq= 1,
#               colsample_bytree= 0.7,
#               min_child_weight= 0,
#               min_split_gain= 0,
#               scale_pos_weight= 99.7)
# 
# 
# model <- lgb.train(params = params, 
#                    data = training_data, 
#                    valids = list(validation = validation_data),
#                    nrounds = 1500, 
#                    verbose= 1, 
#                    early_stopping_rounds = 10, 
#                    eval_freq = 25)
# 
# # Try to find split_feature: 11
# # If you find it, it means it used a categorical feature in the first tree
# lgb.dump(model, num_iteration = 1)
# 
# 
# cat("Validation AUC @ best iter: ", max(unlist(model$record_evals[["validation"]][["auc"]][["eval"]])), "\n\n")


# preds <- predict(model, data = as.matrix(kaggle_test_data, n = model$best_iter))

# from sklearn.preprocessing import LabelEncoder



# params for nn #####
emb_n <- FLAGS$emb_n
dense_n <- FLAGS$n


layer_input_list <- list()
embedding_list <- list()

inputs <- colnames(train)  %>% .[!str_detect(.,"is_attributed")]

# for loop ~ build layers + embeddings
for(g in seq_along(inputs)){
  name <- inputs[g]
  max_val <- train[,max(eval(parse(text=name))) ] + 1L
  print(name)
  layer_input_list[[g]] <- layer_input(shape = 1L, name = name)
  embedding_list[[g]] <- layer_embedding(layer_input_list[[g]], input_dim = max_val, output_dim = emb_n)
}


# combine input layers
fe  <- embedding_list %>% 
  layer_concatenate()

predictions <- fe %>% 
  layer_spatial_dropout_1d(0.2) %>% 
  layer_flatten() %>% 
  layer_dense(units=dense_n, activation = 'relu') %>% 
  layer_dropout(0.2) %>% 
  layer_dense(units = dense_n, activation = 'relu') %>% 
  layer_dropout(0.2) %>% 
  layer_dense(units = 1, activation = 'sigmoid')


model <- keras_model(inputs =  layer_input_list, outputs = predictions)

batch_size <- 20000
epochs <- 2

   source_python("exp_decay.py")
   steps <- as.integer(nrow(train) / batch_size) * epochs
  


  model %>% 
    compile(optimizer = #"adam", 
              optimizer_adam(lr = 0.001, decay = exp_decay(init=0.001, fin=0.0001, steps=steps)),
              loss = 'binary_crossentropy',
              metrics = c("accuracy")
    )

 
 # py_dict(keys=c(0,1), values=c(0.01,0.99))  
  
  
  # split for model fitting and validation
  set.seed(123) # reproduce
  
  train_index <- createDataPartition(train[,is_attributed], p = 0.9, list = FALSE)
  training <- train[train_index, ]
  validation <- train[-train_index,]
  
  
  
  # # kaggle test set 
  kaggle_test_data <- d[is.na(is_attributed)][, is_attributed :=  NULL]
  rm(d,train) ; gc()
  
  training[, .(.N, sum(is_attributed))][, V2/N]
  validation[, .(.N, sum(is_attributed))][, V2/N]
  
  # get y and remove from dataframe
  training_y <- training[,is_attributed]
  training[, is_attributed := NULL]
  
  validation_y <- validation[,is_attributed]
  validation[, is_attributed := NULL]
  
  
  
  
history <- fit(
    object = model, 
    x = map(training, as.vector), # this becomes a numpy array w/ reticulate transformation
    y = training_y,
    batch_size = batch_size,
    class_weight = list("0"=0.01, "1"=0.99),
    verbose = 2, 
    shuffle = TRUE,
    epochs = epochs,
    callbacks = callback_tensorboard("logs/run_a")
    )


# make predictions on the validation dataset withheld from training
y_validation <- validation[, is_attributed]
validation[, is_attributed := NULL]

test_preds <- predict(model, map(validation, as.vector), batch_size=batch_size, verbose=2) %>%
  as.vector()


# rm(test) ; gc()

options(yardstick.event_first = FALSE)

# calculate metrics  

#metrics <- map_df(seq(0.2, 0.9, .1), function(x){
x <-  0.5    

    estimates <- tibble(
      class_prob = test_preds,
      truth =  as.factor(y_validation),
      estimate = as.factor(ifelse(class_prob > x, 1L, 0))
    )
    
    
    tibble(
      accuracy = estimates %>% metrics(truth, estimate) %>% pull(accuracy),
      precision = estimates %>% precision(truth, estimate), # when the model predicts "yes", how often is it actually "yes". 
      recall    = estimates %>% recall(truth, estimate), # specificity /  when the actual value is "yes" how often is the model correct
      #auc = estimates %>% roc_auc(truth, class_prob), # warning long computation time
      f1_statistic = estimates %>% f_meas(truth, estimate, beta = 1)
    ) %>% 
      mutate(
        threshold = x
      )
    
 # }
#)


  end_time <- Sys.time()
 
  metrics %>% 
   mutate(
     time_mins = as.numeric(difftime(end_time, start_time, units='mins'))
  ) %>% 
    write_csv("metrics.csv")


rm(d) ; gc()


# write to kaggle

kaggle_preds <- predict(model, map(kaggle_test_data, as.vector), batch_size=batch_size, verbose=2) %>%
  as.vector()


data.table(
  is_attributed = kaggle_preds
)[, click_id := .I - 1L] %>% 
  select(click_id, is_attributed) %>% 
  fwrite("sub8.csv")
