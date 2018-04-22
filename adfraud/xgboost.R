library(data.table)
library(xgboost)

#---------------------------
cat("Loading data...\n")
train <- fread("../data/adfraud/train.csv", drop = c("attributed_time"), showProgress=F)[(.N - 50e6):.N] 
test <- fread("../data/adfraud/test.csv", drop = c("click_id"), showProgress=F)

set.seed(0)
train <- train[sample(.N, 30e6), ]

#---------------------------
cat("Preprocessing...\n")
y <- train$is_attributed
tri <- 1:nrow(train)
tr_te <- rbind(train, test, fill = T)

rm(train, test); gc()

tr_te[, hour := hour(click_time)
      ][, click_time := NULL
        ][, ip_f := .N, by = "ip"
          ][, app_f := .N, by = "app"
            ][, channel_f := .N, by = "channel"
              ][, device_f := .N, by = "device"
                ][, os_f := .N, by = "os"
                  ][, app_f := .N, by = "app"
                    ][, ip_app_f := .N, by = "ip,app"
                      ][, ip_dev_f := .N, by = "ip,device"
                        ][, ip_os_f := .N, by = "ip,os"
                          ][, ip_chan_f := .N, by = "ip,channel"
                            ][, c("ip", "is_attributed") := NULL]

#---------------------------
cat("Preparing data...\n")
tr_te <- mutate_if(tr_te, is.integer, as.numeric)


# dddd












dtest <- xgb.DMatrix(data = data.matrix(tr_te[-tri]))
ttr_te <- tr_te[tri]; gc()
tri <- caret::createDataPartition(y, p = 0.9, list = F)
dtrain <- xgb.DMatrix(data = data.matrix(tr_te[tri]), label = y[tri])
dval <- xgb.DMatrix(data = data.matrix(tr_te[-tri]), label = y[-tri])
cols <- colnames(tr_te)

rm(tr_te, y, tri); gc()

#---------------------------
cat("Training model...\n")
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 8,
          eta = 0.07,
          max_depth = 4,
          min_child_weight = 24,
          gamma = 36.7126,
          subsample = 0.9821,
          colsample_bytree = 0.3929,
          colsample_bylevel = 0.6818,
          alpha = 72.7519,
          lambda = 5.4826,
          max_delta_step = 5.7713,
          scale_pos_weight = 94,
          nrounds = 2000)

m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 150)

(imp <- xgb.importance(cols, model=m_xgb))
xgb.plot.importance(imp, top_n = 30)

#---------------------------
cat("Creating submission file...\n")
subm <- fread("../input/sample_submission.csv") 
subm[, is_attributed := round(predict(m_xgb, dtest), 6)]
fwrite(subm, paste0("dt_xgb_", m_xgb$best_score, ".csv"))