# TODO 
# ~ hyper param tuning
# ~ cross validation
library(keras)
library(data.table)
library(tidyverse)
library(yardstick)
library(fasttime)
library(reticulate)


predict_for_kaggle <- FALSE

FLAGS <- flags(
  flag_numeric("n", 1000)
)


start_time <- Sys.time()
use_condaenv("r-tensorflow")


#setwd('deep_learning/adfraud/')

# read in train and test datasets
train <- fread("../data/adfraud/train.csv",
                       header=TRUE, 
                       showProgress = FALSE,
                       drop = c("attributed_time"),
                       colClasses = c(rep("integer", 5), rep("character",2), "integer"),
                       col.names = c("ip", "app", "device", "os", "channel", "click_time", "is_attributed")
                 )[121886955L:.N]

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
d[, click_time := fastPOSIXct(click_time)]
d[, hour := hour(click_time)]
d[, day := wday(click_time)]
d[, qty :=  .N, by = .(ip, day, hour) ]
d[, ip_app_count := .N , by = .(ip, app)]
d[, ip_app_os_count := .N , by = .(ip, app, os, channel)]

# drop redundant cols
drop_colz <- c("click_time", "ip")
d[, (drop_colz) := NULL]

# params for nn
emb_n <- 50
dense_n <- FLAGS$n


layer_input_list <- list()
embedding_list <- list()

inputs <- colnames(d) %>% .[!str_detect(.,"is_attributed")]

# for loop ~ build layers + embeddings
for(g in seq_along(inputs)){
  name <- inputs[g]
  max_val <- d[,max(eval(parse(text=name))) ] + 1L
  #print(name)
  layer_input_list[[g]] <- layer_input(shape = 1L, name = name)
  embedding_list[[g]] <- layer_input_list[[g]] %>%
    layer_embedding(input_dim = max_val, output_dim = emb_n)
}


# separate dfs 
train <- d[!is.na(is_attributed)]

# create validation set
# TODO cv , better sampling
set.seed(123)
t_nrows <- train[,.N] * 0.85
train[, rn := .I]

training <- train[sample(t_nrows)]
validation <- train[!rn %in% training[,rn]]
rm(train) ; gc()

validation[, rn :=  NULL]
training[, rn := NULL]

# training[, .(.N, sum(is_attributed))][, V2/N]
# validation[, .(.N, sum(is_attributed))][, V2/N]

# get y and remove from dataframe
training_y <- training[,is_attributed]
training[, is_attributed := NULL]

# # kaggle test set 
kaggle_test_data <- d[is.na(is_attributed)][, is_attributed :=  NULL]
rm(d) ; gc()


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

  # source_python("exp_decay.py")
  # steps <- as.integer(nrow(train) / batch_size) * epochs
  


  model %>% 
    compile(optimizer = "adam",
              #optimizer_adam(lr = 0.001,decay = exp_decay(init=0.001, fin=0.0001, steps=steps)),
              loss = 'binary_crossentropy',
              metrics = c("accuracy")
    )

 
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

metrics <- map_df(seq(0.2, 0.9, .1), function(x){
    
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
    
  }
)


  end_time <- Sys.time()
 
  metrics %>% 
   mutate(
     time_mins = as.numeric(difftime(end_time, start_time, units='mins'))
  ) %>% 
    write_csv("metrics.csv")


rm(d) ; gc()

