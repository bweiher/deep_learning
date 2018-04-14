# titanic sparklyr

library(sparklyr)
library(tidyverse)

sc <- spark_connect(master = "local")

data <- read_csv("titanic.csv") %>% 
  mutate(survived = ifelse(survived==1L, "yes", "no"),
         title =  map_chr(name, ~unlist(str_split(., ' '))[2]),
         title = case_when(
           title %in% c('Mr.','Miss.','Mrs.','Master.') ~ title ,
           TRUE ~ 'other'
         )
  )




data %>% 
  select_if(is.numeric) %>% 
  gather(metric, value) %>% 
  ggplot(aes(x=value)) +
  geom_histogram() +
  facet_wrap(~metric, scales='free')

glimpse(data)


#data <- read_csv("https://raw.githubusercontent.com/baweiher/kaggle/master/titanic/titanic.csv")


d <- copy_to(sc, data)
d <- sdf_partition(d, train = 0.8, validation = 0.2, seed= 123)
 

# SQL transformers 
sql <-   d$train %>% 
  mutate(
    age = if_else(is.na(age), mean(age, na.rm=TRUE), age),
    family_size = sibsp + parch + 1L,
    title = as.integer(title)
  ) %>% 
  mutate(age = case_when(
    age < 10 ~ 1,
    age > 10 & age <= 19 ~ 2 ,
    age > 19 ~ 3
  )) %>% 
  filter(!is.na(embarked)) %>% 
  mutate(family_size = as.numeric(family_size),
         pclass = as.numeric(pclass),
         is_female = as.numeric(sex == 'female')) %>% 
  group_by(pclass) %>% 
  mutate(fare = ifelse(is.na(fare), mean(fare, na.rm=TRUE), fare)) %>% # mean within group
  ungroup() %>% 
  mutate_if(is.integer, as.numeric) %>% 
  select(survived, 
         pclass, 
         age, 
         sibsp, 
         parch, 
         fare,
         family_size,
         is_female)


cols <- colnames(sql)

pipeline <- ml_pipeline(sc) %>% 
   ft_dplyr_transformer(sql) %>% 
   ft_bucketizer("family_size", "family_sizes", splits = c(1,2,5,12)) %>% 
   ft_string_indexer("survived", "label")
 
 
# test what pipeline looks like
 pipeline_model <- pipeline %>% 
   ml_fit(d$train)
 
 pipeline_model %>%
   ml_transform(d$validation) %>% # transformation of test data
   glimpse()
 
 
 vector_assembler <- ft_vector_assembler(
   sc, 
   input_cols = c("family_sizes",setdiff(cols, "survived")), 
   output_col = "features"
 )


 # obtain the labels from the fitted StringIndexerModel
 labels <- pipeline_model %>%
   ml_stage("string_indexer") %>%
   ml_labels()
 
 # convert the predicted numeric values back to class labels
 index_to_string <- ft_index_to_string(sc, "prediction", "predicted_label", 
                                       labels = labels)

 models <- list(
   #'gb' = ml_gradient_boosted_trees(sc),
  'logit' = ml_logistic_regression(sc),
   'rf' = ml_random_forest_classifier(sc),
  'dt' = ml_decision_tree_classifier(sc),
  'nb' = ml_naive_bayes(sc)#,
  #'perc' = ml_multilayer_perceptron_classifier(sc,layers = c(4, 5, 2))
 )
 
 
 lift_data <- list()
 perf_data <- list()
 
 for(g in seq_along(models)){

   model <- models[[g]]
   m_name <- names(models[g])
   
 
 
 # ML PIPELINE
 prod_ml_pipe <- ml_pipeline(
   pipeline, # SQL / Spark Transformers
   vector_assembler, # Feature Transposing
   model,
   index_to_string # Predictor
 )
 
 
 trained_model <- prod_ml_pipe %>% 
   ml_fit(d$train)
 
 
 predictions <- trained_model %>% 
   ml_transform(d$validation)
 
 
 ml_score <- select(predictions, survived, prediction)
 
 
 
 # Calculate AUC and accuracy
 perf_data[[g]] <- perf_metrics <- tibble(
   AUC = 100 * ml_binary_classification_eval(ml_score, "survived", "prediction"),
   Accuracy =  100 * ml_classification_eval(ml_score, "prediction", "survived", "accuracy")
 ) %>% 
   mutate(
     model = m_name
   )
 
 message(m_name)
 
 
 
 }
 
 

 
 perf_data %>% 
   bind_rows() 
 
 old %>% 
   setNames(c('old_AUC','old_Accuracy', 'model')) %>% 
   inner_join(bind_rows(perf_data))
 # 
 # old
 # 
 #  
 # 
 #   perf_data %>% 
 #   bind_rows() %>% 
 #   gather(metric, value, -model) %>% 
 #   ggplot(aes(x=metric, y=value,  fill=model)) +
 #   geom_bar(stat='identity') +
 #   facet_wrap(~model)
 
