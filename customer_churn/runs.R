# runs

library(tfruns)
tuning_run(file = 'churn.R', 
           flags = list(
             dropout1 = c(.01, .05, .1, .25, .35, .4, .5),
             runs = c(16,32,48,64)
           )
)
