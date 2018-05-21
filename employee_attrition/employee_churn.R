library(tidyverse)
library(h2o)
library(corrr)
library(lime)
library(recipes)

file <- 'attrition.xlsx'

d <- readxl::read_excel(file)
dict <- readxl::read_excel(file, sheet = 2, col_names = F) 
skimr::skim(d)

# find useless cols with 1 distinct value or all unique values
# these aren't useful for modeling 
distincts <- d %>% 
  summarize_all(n_distinct) %>% 
  gather() %>% 
  arrange(value) %>% 
  mutate(nrow=nrow(d))

distincts %>% print(n=Inf)

remove_cols <- distincts %>% 
  filter(value == 1 | value == nrow(d)) %>% # in recipes step_zv gets the val == 1 
  pull(key)  

cols <- setdiff(colnames(d), remove_cols)
d <- select(d, cols) %>% 
  select(Attrition, everything()) 
# exploratory data analysis ----- 

d %>% bwmisc::countp(Attrition) # count for target var

# correlation analysis to find factors associated w/ churn
fn <- function(x) x %>% as.factor %>% as.numeric


# shows correlation between Attrition and variable
corr <- d %>% 
  mutate_if(is.character, fn) %>% 
  # sample_frac(size = 1, replace=T) %>%  #to do bootstrapping
  correlate() %>% 
  focus(Attrition) %>% 
  arrange(desc(abs(Attrition)))


corr %>% 
  fashion()  # > 0 ; higher likelihood of leaving



# look at some of the interesting features
# overtime
corr %>% filter(rowname=="OverTime")
d %>% 
  select(Attrition, OverTime) %>% 
  group_by_all() %>% 
  count() %>% 
  ggplot(aes(x=Attrition,y=n, fill=OverTime)) +
  geom_bar(stat="identity", position = position_fill())


# employee age
corr %>% filter(rowname=="Age")
d %>% 
  ggplot(aes(x=Attrition, y=Age)) +
  geom_jitter() +
  geom_violin(alpha = 0.75)


# total working years
corr %>% filter(rowname=="TotalWorkingYears")
d %>% 
  ggplot(aes(x=TotalWorkingYears)) +
  geom_density() +
  facet_wrap(~Attrition, ncol=1L) +
  labs(title = "Total Working Years is Correlated w/ Attrition")

  # Marital Status
corr %>% filter(rowname=="MaritalStatus")
d %>% 
  select(MaritalStatus) %>% 
  mutate(x2 = fn(MaritalStatus)) %>% 
  distinct()  # as x2 rises, more likely to leave 

d %>% 
  select(Attrition, MaritalStatus) %>% 
  group_by_all() %>% 
  count() %>% 
  ggplot(aes(x=Attrition, y=n, fill = MaritalStatus))  +
  geom_bar(stat ="identity", position = position_fill()) 




# model building and data prep/preprocessing  ------ 



# convert other 
data <- d %>% 
  mutate_if(is.character, as.factor) %>% 
  select(Attrition, everything()) 


recipe <- data %>%
  recipe(formula = Attrition ~ .) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(),-all_outcomes()) %>%
  prep(data = data)


baked_data <- bake(recipe, newdata = data) 


baked_data %>% ggplot(aes(x=Attrition, y=TotalWorkingYears)) + geom_jitter() + geom_boxplot()
data %>% ggplot(aes(x=Attrition, y=TotalWorkingYears)) + geom_jitter() + geom_boxplot()


list(baked_data, data) %>% 
  map_df(
    ~select_if(., is.numeric) %>% 
      summarise_all(.funs = funs(mean, sd))
  ) %>% 
  glimpse()


h2o.init() # initialize h2o cluster
# view flow at 127.0.0.1:54321

h2o_data <- as.h2o(baked_data)
split <- h2o.splitFrame(h2o_data, c(0.65, 0.175), seed = 123L)
train <- h2o.assign(split[[1]], "train" ) 
valid <- h2o.assign(split[[2]], "valid" ) 
test  <- h2o.assign(split[[3]], "test" )  


x <- d %>% select(-Attrition) %>% colnames

# train models
models <- h2o.automl(
  x = x, 
  y = "Attrition",
  training_frame    = train,
  validation_frame = valid,
  leaderboard_frame = test,
  max_runtime_secs  = 45
)


models@leaderboard %>% 
  as_tibble() %>% 
  print(n=Inf)

top_model <- models@leader


# make predictions on test dataset
pred_h2o <- h2o.predict(object = models, newdata = test)


# get predictions and truth for test dataset
test_preds <- as.data.frame(pred_h2o) %>% 
  as_tibble() %>% 
  add_column(truth = test[,"Attrition"] %>% as.vector()) %>% 
  mutate(truth = as.factor(truth)) %>% 
  select(prediction = predict, truth,  prob =  Yes)


# look at model metrics -----
perf <- h2o.performance(top_model, test)
# logloss used when your model outputs a probability for each class

test_preds %>% count(truth)

# http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
# confusion matrix for binary classificaiton
h2o.confusionMatrix(perf)

# or 
conf <- test_preds %>% select(truth, prediction) %>% table()
 
 
tn <- conf[1]
tp <- conf[4]
fp <- conf[3] # Type I error ; prediction=Yes ; truth=No
fn <- conf[2] # type II error ; prediction=No  ; truth=Yes
total <- nrow(test_preds)

accuracy <- (tp + tn) / total
misclassification_rate <- 1 - accuracy 
recall <- tp / (tp + fn)  # TPR / Sensitivity
fpr <-   fp / (tn + fp) # when its actually no, how often does it predict yes
specificity <- tn / (fp+ tn) # when its actually no, how often does it predict no (1-fpr)
precision <- tp / (tp + fp) # when we predict yes, how often is it correct 
null_error_rate  <- (fn + tp) / total # how often you would be wrong if you always predicted the majority class

null_error_rate 

model_metrics <- tibble(
  accuracy,
  #misclassification_rate,
  recall,
  false_positive_rt = fpr,
  #specificity, 
  precision,
  #null_error_rate,
  auc = perf@metrics$AUC
)

glimpse(model_metrics) 


# understand thresholds are used here ----
# cutoffs for predictions
test_preds %>% 
  group_by(prediction) %>% 
  summarise(
    min = min(prob),
    max = max(prob)
  )

test_preds %>% 
  ggplot(aes(x=prediction,y=prob, color=prediction)) +
  geom_jitter()


# you can tweak thresholds to optimize for precision or recall
perfm <- perf@metrics$max_criteria_and_metric_scores %>% as.data.frame()
perfm %>% filter(metric == "max f1") %>% pull(threshold) -> threshold_val

h2o.F1(perf) %>%
  as.tibble() %>%
  ggplot(aes(x=threshold, y=f1)) +
  geom_point() +
  geom_vline(xintercept = threshold_val, linetype=2) +
  labs(title = " the threshold that maximizes F1 score is used by default")

# create a ROC curve to show how the model compares against random guessing
roc_data <- left_join(h2o.tpr(perf), h2o.fpr(perf), by="threshold") %>%
  mutate(random_guess = fpr) %>%
  as_tibble() %>% 
  mutate(max = near(threshold, threshold_val, 0.0001)) 


#  ROC Curve ~ evaluate model performance at all possible thresholds

# plots true positive rate (y-axis) against the false positive rate (x-axis)
# allows us to compare how model looks against random guessing 
roc_data %>%
  ggplot(aes(x = fpr)) +
  geom_point(aes(y = tpr, color = "TPR"), alpha = 0.25) +
  geom_line(aes(y = random_guess, color = "Random Guess"), size = 1, linetype = 2) +
  labs(title = "ROC Curve",
       y = "tpr / recall  ",
       x =  "fpr ",
       subtitle = "Model is performing much better than random guessing") +
  annotate("text", x = 0.25, y = 0.65, label = "Better than guessing") +
  annotate("text", x = 0.75, y = 0.25, label = "Worse than guessing") +
  geom_point(data = filter(roc_data, max==TRUE), aes(x=fpr, y=tpr),  shape = 3, size = 3)


# other model metrics
perf@metrics$AUC
h2o.auc(perf) # area under the curve, perfect classification = 1.0


# threshold needs to be configured bc
# precision = how many of our `yes` guesses are accurate?
# recall = what proportion of actual `yes` did we predict `yes`?
# low precision = cost of an intervention  (intervenining for  a false positive)
# low recall = cost of a lost employee (failing to predict a true negative outcome)



 # expain model predictions ----
class(top_model)


# Test our predict_model() function
predict_model(x = top_model, newdata = as.data.frame(test[,-1]), type = 'raw') %>%
  tibble::as_tibble()

# Run lime() on training set
explainer <- lime::lime(
  as.data.frame(train[,-1]), 
  model          = top_model, 
  bin_continuous = FALSE)


# Run explain() on explainer
explanation <- lime::explain(
  as.data.frame(test[1:10,-1]), 
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 6,
  kernel_width = 0.5)

plot_features(explanation) +
  labs(title = "HR Predictive Analytics: LIME Feature Importance Visualization",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

plot_explanations(explanation) +
  labs(title = "LIME Feature Importance Heatmap",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")


# analyze features from lime -----

# Focus on critical features of attrition
feats <- d %>%
  select(Attrition, JobSatisfaction, JobRole, OverTime) %>%
  mutate(Case = row_number()) %>% 
  select(Case, everything()) %>%   
  mutate_if(is.character, as.factor)


feats %>% 
  ggplot(aes(
    x = Attrition, y= JobSatisfaction
  )) +
  geom_jitter() +
  geom_violin()


feats %>% 
  ggplot(aes(x=JobSatisfaction)) +
  geom_density() +
  facet_wrap(~Attrition, ncol=1L)


feats %>% 
 group_by(OverTime) %>% 
 countp(Attrition)
