library(tidyverse)
library(data.table)
library(rsample)
library(lightgbm)
set.seed(0)

#---------------------------


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


integerize_chr(bureau)
integerize_chr(cred_card_bal)
integerize_chr(pos_cash_bal)
integerize_chr(prev)



#---------------------------
cat("Preprocessing...\n")


avg_bureau <- bureau %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(funs(mean(., na.rm = TRUE))) %>% 
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
  
integerize_chr(d)



data <- d[train_index]
kaggle_data <- d[-train_index]

rm(list = 
base::setdiff(
  ls(), c("data", "kaggle_data", "target")
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
training <- training(split)
testing <- testing(split)



training_y <- pull(training, TARGET)
training <- select(training, -TARGET) %>% as.matrix()

validation_y <- pull(testing, TARGET)
validation <- select(testing, -TARGET) %>% as.matrix()


training <- lgb.Dataset(data = training, label = training_y)
validation <- lgb.Dataset(data = validation, label = validation_y)




params <- list(objective = "binary", 
               metric = "auc", 
               learning_rate= 0.01,
               num_leaves= 30,
               max_depth= 7,
               num_iterations = 5000 , 
               nthread = 8, # use actual cpu cores for best speed
             #  min_child_samples= 100,
               max_bin= 100,
               subsample= 0.8, 
               subsample_freq= 1,
               colsample_bytree= 0.8,
              # min_child_weight= 0,
               # min_split_gain= 0.01,
               scale_pos_weight= 99.7     

               )


model <- lgb.train(params = params, 
                   data = training, 
                   valids = list(validation = validation),
                   nrounds = 2000, 
                   verbose= 1, 
                   early_stopping_rounds = 100, #, 
                   eval_freq = 25
                   )




  read_csv("sample_submission.csv") %>%  
  mutate(SK_ID_CURR = as.integer(SK_ID_CURR),
         TARGET = predict(model, as.matrix(kaggle_data))) %>% 
  write_csv("sub5.csv")
