library(tfruns)
tuning_run(file = 'ann.R', 
           flags = list(
              units = c(16,32,48, 64),
              epochs = c(30, 45),
              dropout1 = c(0.1, 0.2, 0,3),
              dropout2 = c(0.1, 0.2, 0,3)
           )
)
