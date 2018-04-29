# run

library(tfruns)
# run various combinations of dropout1 and dropout2
runs <- tuning_run("rpython.R", flags = list(
  n = c(1000, 2048)
))
