
library(data.table)
library(tidyverse)


d <- fread("../data/adfraud/train_sample.csv",
                       drop = c("attributed_time"),
                       colClasses=list(numeric=1:5),  
                       #skiprows = range(1, 131886954), 
                       col.names = c("ip", "app", "device", "os", "channel", "click_time", "is_attributed")
                 )



d[, hour := hour(click_time)]
d[, day := wday(click_time)]

d[,.(max(click_time),min(click_time))]

d[, uniqueN(channel) , by = .(ip, hour, day)]
