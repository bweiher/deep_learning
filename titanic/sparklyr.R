library(sparklyr)
library(dplyr)

# If needed, install Spark locally via `spark_install()`
sc <- spark_connect(master = "local")
iris_tbl <- copy_to(sc, iris)

# split the data into train and validation sets
iris_data <- iris_tbl %>%
  sdf_partition(train = 2/3, validation = 1/3, seed = 123)


pipeline <- ml_pipeline(sc) %>%
  ft_dplyr_transformer(
    iris_data$train %>%
      mutate(Sepal_Length = log(Sepal_Length),
             Sepal_Width = Sepal_Width ^ 2)
  ) %>%
  ft_string_indexer("Species", "label")

#pipeline


pipeline_model <- pipeline %>%
  ml_fit(iris_data$train)


# # pipeline_model is a transformer
# pipeline_model %>%
#   ml_transform(iris_data$validation) %>%
#   glimpse()


# define stages
# vector_assember will concatenate the predictor columns into one vector column
vector_assembler <- ft_vector_assembler(
  sc, 
  input_cols = setdiff(colnames(iris_data$train), "Species"), 
  output_col = "features"
)


logistic_regression <- ml_sv(sc)
gbm <- ml_random_forest_classifier(sc)

# obtain the labels from the fitted StringIndexerModel
labels <- pipeline_model %>%
  ml_stage("string_indexer") %>%
  ml_labels()


# IndexToString will convert the predicted numeric values back to class labels
index_to_string <- ft_index_to_string(sc, "prediction", "predicted_label", 
                                      labels = labels)



# construct a pipeline with these stages
prediction_pipeline <- ml_pipeline(
  pipeline, # pipeline from previous section
  vector_assembler, 
  logistic_regression,
  index_to_string
)

# fit to data and make some predictions
prediction_model <- prediction_pipeline %>%
  ml_fit(iris_data$train)


predictions <- prediction_model %>%
  ml_transform(iris_data$validation)


predictions %>%
  select(label = Species, pred = predicted_label) %>%
  group_by_all() %>% 
  count() %>% 
  ungroup() %>% 
  mutate(x = label == pred) %>% 
  group_by(x) %>% 
  summarise(n = sum(n, na.rm=TRUE)) %>% 
  ungroup() %>% 
  mutate(share = n / sum(n)) %>% 
  filter(x== TRUE) %>% 
  collect() %>% 
  pull(share) -> acc

print(acc)



  


ml_save(prediction_model, '~/model/prediction_model')



# k-means

library(ggplot2)
model <- ml_bisecting_kmeans(iris_tbl, Species ~ Petal_Length + Petal_Width, k = 3, seed = 123)

predictions <- ml_predict(model, iris_tbl) %>%
  collect() %>%
  mutate(cluster = as.factor(prediction))

ggplot(predictions, aes(
  x = Petal_Length, 
  y = Petal_Width, 
  color = predictions$cluster)
) + 
  geom_point() +
  facet_wrap(~Species)


# fp 

# create an item purchase history dataset
items <- data.frame(items = c("1,2,5", "1,2,3,5", "1,2"), stringsAsFactors = FALSE)

# parse into vector column
items_tbl <- copy_to(sc, items) %>%
  mutate(items = split(items, ","))

# fit the model
fp_model <- items_tbl %>%
  ml_fpgrowth(min_support = 0.5, min_confidence = 0.6)

# use the model to predict related items based on
#  learned association rules
fp_model %>%
  ml_transform(items_tbl) %>%
  collect() %>%
  mutate_all(function(x) sapply(x, paste0, collapse = ","))


# dates
copy_to(sc, nycflights13::flights) %>%
  select(carrier, flight, time_hour)

