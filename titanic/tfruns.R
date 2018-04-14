library(tfruns)
library(bwmisc)
library(tidyverse)

tuning_run(file = 'ann.R', 
           flags = list(
              units = c(256,512),
              epochs = c(56),
              dropout1 = c( .5, .6), 
              dropout2= c(.5, .6)
           )
)





# write a function to retrieve all the data we've done predictions in on the test set.
read_in_csv_along_paths <- function(dir, datas){
  run_dirs <- list.files(dir)
  # paste is vectorized
  paths <- glue::glue("{dir}/{run_dirs}/{datas}")
  exists <- purrr::map_lgl(paths, file.exists)
  
   v <- paths[exists]

   purrr::map_df(v, ~readr::read_csv(.)) %>% 
     dplyr::mutate(run_dir = stringr::str_replace_all(v,"/data.csv",""))
   
}




runs <- read_in_csv_along_paths(dir = 'runs', 'data.csv') %>% 
  arrange(desc(f1_statistic)) %>% 
  full_join(ls_runs())
  
  


runs %>% arrange(desc(f1_statistic)) %>% View
