# ann 

library(tidyverse)
library(keras)
library(tensorflow)
library(rsample)
library(corrr)
library(recipes)
library(yardstick)

#keras::use_condaenv('r-tensorflow')

FLAGS <- flags(
  flag_numeric("units" , 16),
  flag_numeric("epochs", 30),
  flag_numeric("dropout1", .1),
  flag_numeric("dropout2", .1)
)


d <- read_csv("titanic.csv") %>% 
  mutate(
    title = map_chr(name, ~str_split(., " ")[[1]][2]),
    title = case_when(
      !title %in% c("Mr.", "Miss.","Mrs.","Master.") ~ "Other",
      TRUE ~ title
    ),
    family_size = sibsp + 1L
    ) %>% 
  select(-body, -name, -ticket, -cabin, -home.dest, -boat) %>% 
  group_by(title, sex) %>% 
  mutate(
    fare = replace_na(mean(fare, na.rm=TRUE)),
    embarked = replace_na(embarked, "Unkown")
  ) %>% 
  ungroup() %>% 
  mutate_if(is.integer, as.double)

#  %>% 
  #select( -body, -boat, -cabin, -home.dest, -embarked, -ticket)
# 
# d %>% 
#   select_if(is.numeric) %>% 
#   gather(metric, value, -survived) %>% 
#   ggplot(aes(x=value, group = survived, fill=as.factor(survived))) +
#   geom_density() + 
#   facet_wrap(~metric, scales ='free')
# 

# d %>% 
#   select(survived, fare) %>% 
#   mutate(
#     log = log(fare),
#     log = ifelse(is.infinite(log), 0, log)
#   ) %>% 
#   correlate() %>% 
#   focus(survived)



# d %>% map(~sum(is.na(.)))

# d <- d %>% 
#   group_by()
#   mutate(
#     age = ifelse(is.na(age), mean(age, na.rm=TRUE), age),
#     fare = ifelse(is.na(fare), mean(fare, na.rm=TRUE), fare)
#   )



sd <- initial_split(d, prop = 0.8)

train <- training(sd)
testing <- testing(sd)



# Create recipe
rec <- recipe(survived ~ ., data = train) %>% 
 # step_log(fare) %>% 
#  step_discretize(fare, options = list(cuts = 3)) %>% 
  step_meanimpute(age) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train)


# Predictors
x_train_tbl <- bake(rec, newdata = train) %>% select(-survived)
x_test_tbl  <- bake(rec, newdata = testing) %>% select(-survived)

# Response variables for training and testing sets
y_train_vec <- pull(train, survived)
y_test_vec <-  pull(testing, survived)



# Building our Artificial Neural Network
model_keras <- keras_model_sequential()

model_keras %>% 
  
  # First hidden layer
  layer_dense(
    units              = FLAGS$units, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = FLAGS$dropout1) %>%
  
  # Second hidden layer
  layer_dense(
    units              = FLAGS$units, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = FLAGS$dropout2) %>%
  
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

# keras_model




# Fit the keras model to the training data
history <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 50, 
  epochs           = FLAGS$epochs,
  validation_split = 0.30,
  callbacks = callback_tensorboard(log_dir = "logs")
)



# # Print a summary of the training history
# print(history)
# 
# plot(history) 


# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
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

# read_csv('data.csv') %>% 
#   filter(run!='First!') %>% 
#   bind_rows(model_quality) %>% 
#   mutate(run = row_number()) %>% 
#   write_csv("data.csv")
# 
# 


# library(glue)
# library(crayon)
# 
# for(g in seq_along(model_quality)){
#  metric <-  names(model_quality[g])
#  val <- model_quality[[g]]
#  glue_col("
#             
# {red {metric}}
# {green {round(val,3)}}
#           ") %>% 
#    print()
#  
# }
