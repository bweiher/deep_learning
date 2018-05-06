 
m <- max(x_train_tbl$MonthlyCharges)

model %>% layer_embedding(m, m/max(m))



model_keras %>% 
  
  # First hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = FLAGS$dropout1 ) %>%
  
  # Second hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = FLAGS$dropout1) %>%
  
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
  epochs           = FLAGS$runs,
  validation_split = 0.30,
  callbacks = callback_tensorboard("logs/run_a")
) 


# Print a summary of the training history
print(history)
plot(history) 


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


options(yardstick.event_first = FALSE)

# estimates_keras_tbl %>% conf_mat(truth, estimate)

#estimates_keras_tbl %>% metrics(truth, estimate)

# We can also get the ROC Area Under the Curve (AUC) measurement. 
# AUC is often a good metric used to compare different classifiers and to
# compare to randomly guessing (AUC_random = 0.50). 
# Our model has AUC = 0.85, which is much better than randomly guessing. 
# Tuning and testing different classification algorithms may yield even better results.
#estimates_keras_tbl %>% roc_auc(truth, class_prob)


# F1-Statistic
#estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)

# Precision
metrics <- tibble(
  accuracy = estimates_keras_tbl %>% metrics(truth, estimate) %>% pull(accuracy),
  precision = estimates_keras_tbl %>% precision(truth, estimate),
  recall    = estimates_keras_tbl %>% recall(truth, estimate),
  auc = estimates_keras_tbl %>% roc_auc(truth, class_prob),
  f1_statistic = estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)
)


# 
# # Setup lime::model_type() function for keras
# model_type.keras.models.Sequential <- function(x, ...) {
#   "classification"
# }
# 
# # Setup lime::predict_model() function for keras
# predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
#   pred <- predict_proba(object = x, x = as.matrix(newdata))
#   data.frame(Yes = pred, No = 1 - pred)
# }
# 
# 
# # Test our predict_model() function
# predict_model(x = model_keras, newdata = x_test_tbl, type = 'raw') %>%
#   tibble::as_tibble()
# 
# 
# # Run lime() on training set
# explainer <- lime::lime(
#   x              = x_train_tbl, 
#   model          = model_keras, 
#   bin_continuous = FALSE
# )
# 
# # 
# # Now we run the explain() function, which returns our explanation. 
# # This can take a minute to run so we limit it to just the first ten
# # rows of the test data set. We set n_labels = 1 because we care about 
# # explaining a single class. Setting n_features = 4 returns the top
# # four features that are critical to each case. Finally, setting 
# # kernel_width = 0.5 allows us to increase the "model_r2" value by
# # shrinking the localized evaluation.
# 
# # Run explain() on explainer
# explanation <- lime::explain(
#   x_test_tbl[1:10, ], 
#   explainer    = explainer, 
#   n_labels     = 1, 
#   n_features   = 4,
#   kernel_width = 0.5
# )
# 
# 
# plot_features(explanation) +
#   labs(title = "LIME Feature Importance Visualization",
#        subtitle = "Hold Out (Test) Set, First 10 Cases Shown")
# 
# 
# 
# 
# 
# plot_explanations(explanation) +
#   labs(title = "LIME Feature Importance Heatmap",
#        subtitle = "Hold Out (Test) Set, First 10 Cases Shown")
# 
# 
# # Feature correlations to Churn
# corrr_analysis <- x_train_tbl %>%
#   mutate(Churn = y_train_vec) %>%
#   correlate() %>%
#   focus(Churn) %>%
#   rename(feature = rowname) %>%
#   arrange(abs(Churn)) %>%
#   mutate(feature = as_factor(feature)) 
# corrr_analysis
# 
# 
# # Correlation visualization
# corrr_analysis %>%
#   ggplot(aes(x = Churn, y = fct_reorder(feature, desc(Churn)))) +
#   geom_point() +
#   # Positive Correlations - Contribute to churn
#   geom_segment(aes(xend = 0, yend = feature), 
#                color = palette_light()[[2]], 
#                data = corrr_analysis %>% filter(Churn > 0)) +
#   geom_point(color = palette_light()[[2]], 
#              data = corrr_analysis %>% filter(Churn > 0)) +
#   # Negative Correlations - Prevent churn
#   geom_segment(aes(xend = 0, yend = feature), 
#                color = palette_light()[[1]], 
#                data = corrr_analysis %>% filter(Churn < 0)) +
#   geom_point(color = palette_light()[[1]], 
#              data = corrr_analysis %>% filter(Churn < 0)) +
#   # Vertical lines
#   geom_vline(xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +
#   geom_vline(xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
#   geom_vline(xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
#   # Aesthetics
#   theme_tq() +
#   labs(title = "Churn Correlation Analysis",
#        subtitle = "Positive Correlations (contribute to churn), Negative Correlations (prevent churn)",
#        y = "Feature Importance")



