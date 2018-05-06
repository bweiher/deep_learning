# new_lightgbm
# library(keras)
library(data.table)
library(tidyverse)
# library(yardstick)
library(fasttime)
# library(reticulate)
library(caret)
library(R6)
library(lightgbm)

# read in train and test datasets
d <- fread("../data/adfraud/train_sample.csv",
               header=TRUE, 
               showProgress = FALSE,
               drop = c("attributed_time"),
               colClasses = c(rep("integer", 5), rep("character",2), "integer"),
               col.names = c("ip", "app", "device", "os", "channel", "click_time", "is_attributed")
)


d[, click_time := fastPOSIXct(click_time,"GMT")]
d[, hour := hour(click_time)]
d[, day := wday(click_time)]
d[, yday := yday(click_time) ]
d[, qty :=  .N, by = .(ip, yday, hour) ]
d[, ip_app_count := .N , by = .(ip, app)]
d[, ip_app_os_count := .N , by = .(ip, app, os)]


# drop redundant cols
drop_colz <- c("click_time", "ip")
d[, (drop_colz) := NULL]


train_index <- createDataPartition(d[,is_attributed], p = 0.9, list = FALSE)

training <- d[train_index, ]
validation <- d[-train_index,]


# get y and remove from dataframe
training_y <- training[,is_attributed]
training[, is_attributed := NULL]

validation_y <- validation[,is_attributed]
validation[, is_attributed := NULL]



cats <- c("app", "device", "os", "channel", "hour")

training <- as.matrix(training)
training_data <- lgb.Dataset(data = training, 
                             label = training_y, 
                             categorical_feature = cats)


cat("Creating the 'dvalid' for modeling...")
validation <- as.matrix(validation)
validation_data <- lgb.Dataset(data = validation, 
                               label = validation_y, 
                               categorical_feature = cats)


# lgb.Dataset.save(training_data, "training.buffer")  
#lgb.Dataset.save(validation_data, "validation.buffer")

# rm(training, validation) ; gc()
# params for model

params <- list(objective = "binary", 
               metric = "auc", 
               learning_rate= 0.1,
               num_leaves= 7,
               max_depth= 4,
               device = "gpu",
               gpu_platform_id = 1,
               gpu_device_id = 1,
               nthread = 1,
               min_child_samples= 100,
               max_bin= 100,
               subsample= 0.7, 
               subsample_freq= 1,
               colsample_bytree= 0.7,
               min_child_weight= 0,
               min_split_gain= 0,
               scale_pos_weight= 99.7)


model <- lgb.train(params = params, 
                   data = training_data, 
                   valids = list(validation = validation_data),
                   nrounds = 1500, 
                   verbose= 1, 
                   early_stopping_rounds = 10, 
                   eval_freq = 25)