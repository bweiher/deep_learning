# ann 
library(tidyverse)
library(keras)
library(tensorflow)
library(rsample)
library(corrr)
library(recipes)
library(yardstick)

#keras::use_condaenv('r-tensorflow')
start_time <- Sys.time()

# run 4/36 (flags = list(units = 32, epochs = 45, dropout1 = 0.3, dropout2 = 0.1
FLAGS <- flags(
  flag_numeric("units" , 256),
  flag_numeric("epochs", 56),
  flag_numeric("dropout1", .4),
  flag_numeric("dropout2", .6)
)

# comp data 
train <- read_csv('../data/titanic/train.csv')
test <-  read_csv("../data/titanic/test.csv")

train_maps <- train %>% select(name = Name, Survived)
test_maps <- test %>% transmute(name = Name, Survived = NA_integer_)

get_comp_data_ready <- function(dataframe){
  colnames(dataframe) <- str_to_lower(colnames(dataframe))
  dataframe %>%
    mutate(
      title = map_chr(name, ~str_split(., " ")[[1]][2]),
      title = case_when(
        !title %in% c("Mr.", "Miss.","Mrs.","Master.") ~ "Other",
        TRUE ~ title
      ),
      family_size = sibsp + 1L
    ) %>%
    select(-name, -ticket, -cabin, -passengerid) %>%
    group_by(title, sex) %>%
    mutate(
      fare = replace_na(mean(fare, na.rm=TRUE)),
      embarked = replace_na(embarked, "Unkown")#,
    #  age = replace_na(mean(age, na.rm=TRUE))
    ) %>%
    ungroup() %>%
    mutate_if(is.integer, as.double)
}

# test <-  get_comp_data_ready(test)
# train <- get_comp_data_ready(train)

d <- read_csv("../data/titanic/titanic.csv") %>% 
  mutate(
    title = map_chr(name, ~str_split(., " ")[[1]][2]),
    title = case_when(
      !title %in% c("Mr.", "Miss.","Mrs.","Master.") ~ "Other",
      TRUE ~ title
    ),
    family_size = sibsp + 1L + parch
    ) %>% 
  select(-body, -ticket, -cabin, -home.dest, -boat) %>% 
  group_by(title, sex) %>% 
  mutate(
    embarked = replace_na(embarked, "Unkown")
  ) %>% 
  ungroup() %>% 
  mutate_if(is.integer, as.double)


d <- d %>% semi_join(train_maps) %>% select(-name)
test_maps


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



sd <- initial_split(d, prop = 0.85)

train <- training(sd)
testing <- testing(sd)



# Create recipe
rec <- recipe(survived ~ ., data = train) %>% 
 # step_log(fare) %>% 
  step_meanimpute(fare) %>% 
  step_discretize(fare, options = list(cuts = 4)) %>% 
  step_meanimpute(age) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train)


# Predictors
x_train_tbl <- bake(rec, newdata = train) %>% select(-survived)
x_test_tbl  <- #tryCatch(
  bake(rec, newdata = testing) %>% select(-survived)#,
  #error = function(e){NA}



if(!is.na(x_test_tbl)){
  


# Response variables for training and testing sets
y_train_vec <- pull(train, survived)
y_test_vec <-  pull(testing, survived)



# Building our Artificial Neural Network
model_keras <- keras_model_sequential() %>% 
  
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
  truth      = as.factor(y_test_vec) ,
  estimate   = as.factor(yhat_keras_class_vec) ,
  class_prob = yhat_keras_prob_vec
)


# metric qualiuty on unseen data 
tibble(
  accuracy = estimates_keras_tbl %>% metrics(truth, estimate) %>% pull(accuracy),
  precision = estimates_keras_tbl %>% precision(truth, estimate),
  recall    = estimates_keras_tbl %>% recall(truth, estimate),
  auc = estimates_keras_tbl %>% roc_auc(truth, class_prob),
  f1_statistic = estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)
) %>% 
  mutate(
    train_time = difftime(Sys.time(), start_time, units='secs') %>% as.numeric
  )# %>% 
  #write_csv('data.csv')

#colnames(test) <- colnames(test) %>% str_to_lower()
unseen_data <- bake(rec, newdata = get_comp_data_ready(test)) 

# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(unseen_data)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()


read_csv("../data/titanic/gender_submission.csv")

submission <- read_csv("../data/titanic/test.csv") %>% 
  select(PassengerId) %>% 
  add_column(Survived = yhat_keras_class_vec) %>% 
  mutate(Survived = as.integer(Survived))


write_csv(submission, "../data/titanic/ann_submission.csv") 


rm(list = ls())
}


# save_model_hdf5(model_keras, filepath = getwd())
