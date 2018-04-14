# r
library(tidyverse)
library(data.table)
library(tensorflow)
library(keras)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)

# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

use_condaenv("r-tensorflow")

test <- read_csv("test.csv")
train <- read_csv("train_sample.csv")


train <- train %>% 
  mutate(
    hr = hour(click_time),
    dow = wday(click_time),
    mday = mday(click_time)
  ) %>% 
  select(is_attributed, everything(),
         -click_time,
         -attributed_time)




# Split test/training sets
set.seed(100)
train_test_split <- initial_split(train)

# Retrieve train and test sets
train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split) 


# # Create recipe
rec_obj <- recipe(is_attributed ~ ., data = train_tbl) %>%
  step_discretize(app, options = list(cuts = 6)) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)
# 

y_train_vec <- pull(train_tbl, is_attributed)
y_test_vec <- pull(test_tbl, is_attributed)
# Building our Artificial Neural Network
model_keras <- keras_model_sequential()

model_keras %>% 
  
  # First hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(train_tbl)) %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Second hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Output layer
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  
  # Compile ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )


keras_model


# Fit the keras model to the training data
history <- fit(
  object           = model_keras, 
  x                = as.matrix(train_tbl), 
  y                = y_train_vec,
  batch_size       = 50, 
  epochs           = 10,
  validation_split = 0.30
)



# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(test_tbl)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(test_tbl)) %>%
  as.vector()


# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)




# Precision
model_quality <-  tibble(
  accuracy = estimates_keras_tbl %>% metrics(truth, estimate) %>% pull(accuracy),
  precision = estimates_keras_tbl %>% precision(truth, estimate),
  recall    = estimates_keras_tbl %>% recall(truth, estimate),
  auc = estimates_keras_tbl %>% roc_auc(truth, class_prob),
  f1_statistic = estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)
)




