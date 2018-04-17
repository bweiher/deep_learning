# r
library(tidyverse)
library(data.table)
library(tensorflow)
library(keras)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)
library(lime)
library(lubridate)

# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

#use_condaenv("r-tensorflow")

test <- read_csv("../data/adfraud/test.csv")
train <- fread("../data/adfraud/smaller_train.csv")
# 
# is_attributed <- train[is_attributed == 1L]
# n <- nrow(is_attributed) * 2
# 
# train <- train[is_attributed == 0L]
# smaller_train <- sample_n(tbl = train[is_attributed != 1L], size = n)
# 
# train <- bind_rows(is_attributed, sample_n(train[is_attributed == 0], n))
# 
# rm(is_attributed)
# 


get_df_ready <- function(datatable){

  setDT(datatable)

  datatable[, `:=` (
  click_time = ymd_hms(click_time),
  app = as.factor(app),
  device = as.factor(device),
  os = as.factor(device),
  channel = as.factor(channel)
  )]

  datatable[, `:=` (
  hr = hour(click_time),
  dow = wday(click_time))]


  datatable[, hr_count := .N , by = .(dow, hr)]
  datatable[, ip_count := uniqueN(ip), by = .(dow, hr)]
  datatable[, click_time :=  NULL]
  datatable[, attributed_time := NULL]
  
  setcolorder(datatable, c("is_attributed","ip", "app", "device", "os", "channel","hr", "dow", "hr_count", "ip_count"))
}


get_df_ready(train)




# train %>% sample_n(50000)
# train  %>% ggplot(aes(hr_count)) + geom_histogram()
# train  %>% ggplot(aes(ip_count)) + geom_histogram()
# %>%
#     # mutate(
#     #   
#     # )
# #    group_by()
#     select(is_attributed, everything(),
#            -click_time,
#            -attributed_time
#            )
# }


# train <- get_df_ready(train)

# 
# train <- train %>% sample_n(2e+06) %>% setDT()

# train %>%
#   mutate(
#     wday = wday(click_time),
# 
#   )

# Split test/training sets
set.seed(100)
train_test_split <- initial_split(train)

# Retrieve train and test sets
train_tbl <- training(train_test_split)
test_tbl  <-  testing(train_test_split) 


# # Create recipe
rec_obj <- recipe(is_attributed ~ ., data = train_tbl) %>%
  # step_dummy(all_nominal(), -all_outcomes()) %>% 
   step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)

# 
y_train_vec <- pull(train_tbl, is_attributed)
y_test_vec <- pull(test_tbl, is_attributed)

x_train_tbl <- bake(rec_obj, newdata = train_tbl) %>% select(-is_attributed) 


x_test_tbl <- bake(rec_obj, newdata = test_tbl) %>% select(-is_attributed)

# Building our Artificial Neural Network
model_keras <- keras_model_sequential() %>% 
  
  # First hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  
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
fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 50, 
  epochs           = 4,
  validation_split = 0.30
)

touch("model")
save_model_hdf5(model_keras, filepath = 'model/model.hdf5')

# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()


# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) ,
  estimate   = as.factor(yhat_keras_class_vec) ,
  class_prob = yhat_keras_prob_vec
)




# Precision
model_quality <-  tibble(
  accuracy = estimates_keras_tbl %>% metrics(truth, estimate) %>% pull(accuracy),
  precision = estimates_keras_tbl %>% precision(truth, estimate),
  recall    = estimates_keras_tbl %>% recall(truth, estimate),
  # auc = estimates_keras_tbl %>% roc_auc(truth, class_prob),
  f1_statistic = estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)
)



# use model to predict on kaggle test data

test <- fread('../data/adfraud/test.csv')

test <- test %>% 
 # head(5) %>% 
  mutate(
    attributed_time = NA_character_ , is_attributed = NA_integer_
  ) %>% 
  get_df_ready()


# bake test
test_x <- bake(rec_obj, newdata = test) %>% select(-is_attributed) 


# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(test_x)) %>%
  as.vector()

# # Predicted Class Probability
# yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(test_x)) %>%
#   as.vector()


# use correlation analysis on top features

# use lime to understand model
library(lime)


# # Setup lime::model_type() function for keras
model_type.keras.models.Sequential <- function(x, ...) {
  "classification"
}

# 
# Setup lime::predict_model() function for keras
predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  data.frame(Yes = pred, No = 1 - pred)
}


# Test our predict_model() function
predict_model(x = model_keras, newdata = head(x_test_tbl), type = 'raw') %>%
  tibble::as_tibble()


# Run lime() on training set
explainer <- lime::lime(
  x              = x_train_tbl,
  model          = model_keras,
  bin_continuous = FALSE
)

#
# Now we run the explain() function, which returns our explanation.
# This can take a minute to run so we limit it to just the first ten
# rows of the test data set. We set n_labels = 1 because we care about
# explaining a single class. Setting n_features = 4 returns the top
# four features that are critical to each case. Finally, setting
# kernel_width = 0.5 allows us to increase the "model_r2" value by
# shrinking the localized evaluation.

# Run explain() on explainer
explanation <- lime::explain(
  x_test_tbl[1:10, ],
  explainer    = explainer,
  n_labels     = 1,
  n_features   = 4,
  kernel_width = 0.5
)


plot_features(explanation) +
  labs(title = "LIME Feature Importance Visualization",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

library(corrr)

corrr_analysis <- x_train_tbl %>%
  mutate(fraud = y_train_vec) %>%
  correlate() %>%
  focus(fraud) %>%
  rename(feature = rowname) %>%
  arrange(abs(fraud)) %>%
  mutate(feature = as_factor(feature))

corrr_analysis <- rename(corrr_analysis, Churn = fraud)

library(tidyquant)
corrr_analysis %>%
  ggplot(aes(x = Churn, y = fct_reorder(feature, desc(Churn)))) +
  geom_point() +
  # Positive Correlations - Contribute to churn
  geom_segment(aes(xend = 0, yend = feature),
               color = palette_light()[[2]],
               data = corrr_analysis %>% filter(Churn > 0)) +
  geom_point(color = palette_light()[[2]],
             data = corrr_analysis %>% filter(Churn > 0)) +
  # Negative Correlations - Prevent churn
  geom_segment(aes(xend = 0, yend = feature),
               color = palette_light()[[1]],
               data = corrr_analysis %>% filter(Churn < 0)) +
  geom_point(color = palette_light()[[1]],
             data = corrr_analysis %>% filter(Churn < 0)) +
  # Vertical lines
  geom_vline(xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  # Aesthetics
  theme_tq() +
  labs(title = "Churn Correlation Analysis",
       subtitle = "Positive Correlations (contribute to churn), Negative Correlations (prevent churn)",
       y = "Feature Importance")

# make predictions on the test data

sample_sub <- fread('../data/adfraud/sample_submission.csv')



sub <- data.table(
  is_attributed = yhat_keras_class_vec
)


sub[, click_id :=  1L:.N]

setcolorder(sub, c("click_id", "is_attributed"))
sub[, click_id := as.integer(click_id)]

fwrite(submission, "../data/adfraud/sub.csv")
sub[,.N] == 18790469L 


submission <- fread("../data/adfraud/sub.csv")
submission[, click_id := click_id - 1L]
