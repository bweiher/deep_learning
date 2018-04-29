# run

library(tfruns)

# run various combinations of dropout1 and dropout2
runs <- tuning_run("rpython.R", flags = list(
  n = c(1000, 2000, 4000)
))




run_dirs <- list.files('runs')

run_metrics <- map_df(run_dirs, function(x){
  
  x <- paste0("runs/", x, "/metrics.csv")
  read_csv(x) %>% 
    mutate(
      run_dir = str_replace_all(x, "/metrics.csv", "")
    )
  
  
})

library(hrbrthemes)

d <- left_join(
  run_metrics, ls_runs(),  by = "run_dir"
) 


d %>% 
  select(accuracy:time_mins, flag_n) %>% 
  gather(key, value, -threshold, -flag_n) %>%
  mutate(flag_n = as.factor(flag_n)) %>% 
  distinct() %>% 
  ggplot(aes(x=threshold, y=value, color=flag_n)) +
  geom_point() +
  facet_wrap(~key, scales = 'free_y') +
  theme_ipsum_tw() +
  theme(legend.position = 'bottom') +
  geom_line()


