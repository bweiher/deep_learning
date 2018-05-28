library(tidyverse)
library(data.table)
library(rsample)
library(lightgbm)
set.seed(0)

#---------------------------
# '[2801]:	validation's auc:0.776327'

train <- fread("application_train.csv")
test <-  fread("application_test.csv")


# train[,.N, by=TARGET][, pct := N/sum(N)][]

# fns
get_chr_colnames <- function(df){
  b_names <- map(df, is.character)
  b_names[b_names == TRUE] %>% names  
}


integerize_chr <- function(dt){
  for (j in get_chr_colnames(dt)) set(dt, j = j, value = as.integer(as.factor(dt[[j]])))
}

bureau <- fread("bureau.csv") 
cred_card_bal <- fread("credit_card_balance.csv")
pos_cash_bal <- fread("POS_CASH_balance.csv") 
prev <- fread("previous_application.csv")
install_payments <- fread("installments_payments.csv") # 


cats <- c(
  get_chr_colnames(bureau),
  get_chr_colnames(cred_card_bal),
  get_chr_colnames(pos_cash_bal),
  get_chr_colnames(prev)
  
  )

integerize_chr(bureau)
integerize_chr(cred_card_bal)
integerize_chr(pos_cash_bal)
integerize_chr(prev)



#---------------------------
cat("Preprocessing...\n")


avg_bureau <- bureau %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(funs(mean(., na.rm = TRUE),
                     max(., na.rm=TRUE),
                     min(., na.rm=TRUE))) %>% 
  mutate(buro_count = bureau %>%  
           group_by(SK_ID_CURR) %>% 
           count() %>% 
           pull(n))

avg_cred_card_bal <- cred_card_bal %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(funs(mean(., na.rm = TRUE))) %>% 
  mutate(card_count = cred_card_bal %>%  
           group_by(SK_ID_CURR) %>% 
           count() %>% 
           pull(n))

avg_pos_cash_bal <- pos_cash_bal %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(funs(mean(., na.rm = TRUE))) %>% 
  mutate(pos_count = pos_cash_bal %>%  
           group_by(SK_ID_CURR) %>% 
           count()  %>% 
           pull(n))

avg_prev <- prev %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(funs(mean(., na.rm = TRUE))) %>% 
  mutate(nb_app = prev %>%  
           group_by(SK_ID_CURR) %>% 
           count() %>% 
           pull(n))


train_index <- 1:nrow(train)
target <- pull(train, TARGET)

d <- train %>% 
  select(-TARGET) %>% 
  bind_rows(test) %>%
  left_join(avg_bureau, by = "SK_ID_CURR") %>% 
  left_join(avg_cred_card_bal, by = "SK_ID_CURR") %>% 
  left_join(avg_pos_cash_bal, by = "SK_ID_CURR") %>% 
  left_join(avg_prev, by = "SK_ID_CURR") %>% 
  setDT()

c(d %>% select_if(is.character) %>% colnames,
  cats) -> cats
  
integerize_chr(d)


d <- d[,lapply(.SD,function(x){ifelse(is.na(x),-999,x)})]             
                             

data <- d[train_index]
kaggle_data <- d[-train_index]

rm(list = 
base::setdiff(
  ls(), c("data", "kaggle_data", "target", "cats")
)
)
gc()


data[, TARGET := target]
# recipes! 
# 
# rec <- recipe(TARGET ~ ., data = data) %>% 
#   step_scale(all_predictors(), -all_outcomes()) %>% 
#   step_center(all_predictors(), -all_outcomes()) %>% 
#   prep(data = data)
# 
# baked_data <- bake(rec, newdata=data)


split <- initial_split(data, prop = 0.90)
training <- rsample::training(split)
testing <- rsample::testing(split)

training_y <- pull(training, TARGET)
validation_y <- pull(testing, TARGET)

training <- select(training, -TARGET) 
validation <- select(testing, -TARGET) 

sub <- read_csv("sample_submission.csv") %>% 
  select(-TARGET)

# light gbm --------------------------------------------------------


training_lgb <-   lgb.Dataset(data = as.matrix(training), label = training_y)
validation_lgb <- lgb.Dataset(data = as.matrix(validation), label = validation_y)



params <- list(objective = "binary", 
               metric = "auc", 
               learning_rate= 0.02,
               num_leaves= 45,
               max_depth= 7,
               num_iterations = 5000 , 
               nthread = 8, # use actual cpu cores for best speed
             #  min_child_samples= 100,
               max_bin= 200,
               subsample= 0.9, 
               subsample_freq= 1,
               colsample_bytree= 0.8,
              # min_child_weight= 0,
                min_split_gain= 0.01,
              # scale_pos_weight= 99.7,
                reg_alpha=.1,
                reg_lambda=.1,
                boosting = "dart"
               )


model <- lgb.train(params = params, 
                   data = training_lgb, 
                   valids = list(validation = validation_lgb),
                   nrounds = 2000, 
                   verbose= 1, 
                   early_stopping_rounds = 200, #, 
                   eval_freq = 100
                   )






imp <- lightgbm::lgb.importance(model)

lgb.plot.importance(imp)
lgb.plot.interpretation(imp)


lgm_preds <-  sub %>%  
  mutate(TARGET = predict(model, as.matrix(kaggle_data)))
  
  
  # %>% 
  # write_csv("sub10.csv")
  

# xgboost --------------------------------------------------------
  
  library(xgboost)
  xg_train <-  xgb.DMatrix(data.matrix(training), label =  training_y)
  xg_test <-  xgb.DMatrix(data.matrix(testing), label =  validation_y)
  
  cat("Preparing data...\n")

  cols <- colnames(xg_train)
  
  kaggle_data_gbm <- xgb.DMatrix(data.matrix(kaggle_data))

  cat("Training model...\n")
  p <- list(
    objective = "binary:logistic",
            booster = "gbtree",
            eval_metric = "auc",
            nthread = 8,
            eta = 0.025,
            max_depth = 6,
            min_child_weight = 19,
            gamma = 0,
            subsample = 0.8,
            colsample_bytree = 0.632,
            alpha = 0,
            lambda = 0.05,
            nrounds = 2000
            )
  
  m_xgb <- xgb.train(params = p, 
                     data = xg_train,
                     nrounds =  p$nrounds, 
                     watchlist = list(val = xg_test), 
                     print_every_n = 50, 
                     early_stopping_rounds = 200)
  
  xgb.importance(cols, model=m_xgb) %>% 
  xgb.plot.importance(top_n = 30)
  
  

  
  xgboost_preds <- sub %>% 
    mutate(
      TARGET = predict(m_xgb, kaggle_data_gbm)
    )
  
  
 model_preds <-  lgm_preds %>% 
    rename(lgm_targ = TARGET) %>% 
    inner_join(xgboost_preds)
 
 
 model_preds %>% 
   select(
     SK_ID_CURR, TARGET
   ) %>% 
   write_csv("xg.csv")
  
 # .774 highest score 
 model_preds %>% 
   transmute(
     SK_ID_CURR , TARGET = (lgm_targ + TARGET) /  2
   ) %>% 
   write_csv("avg.csv")
 
  # h2o ----------
  library(h2o)

  h2o.init()
 
  training_h2o <- training %>% 
    mutate(TARGET = ifelse(TARGET == 1, "1", "0") %>% as.factor) %>% 
             as.h2o()
  
  testing_h2o <- testing %>%
    mutate(TARGET = ifelse(TARGET == 1, "1", "0") %>% as.factor) %>% 
    as.h2o()
  

  
  # model <- h2o.getModel('DRF_0_AutoML_20180526_231831')
  x <- setdiff(colnames(training),"TARGET")
  
  
  # train models
  models <- h2o.automl(
    x = x,
    y = "TARGET",
    training_frame = training_h2o, # used for training
    validation_frame = testing_h2o, # used for early stopping of models / grid searches
    #leaderboard_frame = test, # not used for anything besides leaderboard scoring
    max_runtime_secs = 600,
    project_name = "airbnb",
    stopping_metric = "AUC"
  )
  
  models@leaderboard
  
  top_model <- models@leader
  
  perf <- h2o.performance(top_model)
  
  h2o.auc(top_model)
  
  
  h_h2o <- as.h2o(kaggle_data)
  preds <- predict(top_model, h_h2o) #%>% as.vector()
  
  
   read_csv("sample_submission.csv") %>%  
    mutate(SK_ID_CURR = as.integer(SK_ID_CURR),
           TARGET = preds[,"p0"] %>% as.vector()) %>% 
     write_csv("h2o.csv")
