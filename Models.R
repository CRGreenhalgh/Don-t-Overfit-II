library(tidyverse)
library(caret)
library(LiblineaR)
library(Matrix)
library(glmnet)
library(randomForest)
library(RRF)

overfit_train <- read_csv("train.csv")
overfit_test <- read_csv("test.csv")
overfit_train$target <- as.factor(overfit_train$target)

#### XGBoost ####
tc <- trainControl(method = "cv", number = 10)

tg <- expand.grid(nrounds = seq(100,250, by = 50),
                  max_depth = 2:9,
                  eta = seq(.05, .5, by = .05),
                  gamma = 0,
                  subsample = 1,
                  colsample_bytree = 1,
                  min_child_weight = 1) 

overfit_xgb <- train(form = target ~.,
                      data = overfit_train %>% select(-id),
                      method = "xgbTree",
                      trControl = tc,
                      tuneGrid = tg,
                      tuneLength = 4)  
beepr::beep()

plot(overfit_xgb)
overfit_xgb$results
xgb_preds1 <- predict(overfit_rf, newdata = overfit_test)
xgb_preds_df1 <- data.frame(id = overfit_test$id, target = xgb_preds1)

write_csv(x = xgb_preds_df1, path = "./XGBPreds2.csv")



#### Random Forest ####
overfit_rf <- train(form = target~.,
                 data=(overfit_train %>% select(-id)),
                 method = "ranger",
                 trControl=trainControl(method="repeatedcv",
                                        number=10, #Number of pieces of your data
                                        repeats=3) #repeats=1 = "cv"
)

overfit_rf$results
rf_preds1 <- predict(overfit_rf, newdata = overfit_test)
rf_preds_df1 <- data.frame(id = overfit_test$id, target = rf_preds1)

write_csv(x = rf_preds_df1, path = "./RFPreds1.csv")

#### Regularized logistic regression ####
tc <- trainControl(method = "cv", number = 10)

tg <- expand.grid(cost = 1:3,
                  loss = "L1",
                  epsilon = seq(0.001, 0.10, by = 0.005)) 
  
overfit_rlr <- train(form = target~.,
                     data = overfit_train %>% select(-id),
                     method = "regLogistic",
                     trControl = tc,
                     tuneGrid = tg,
                     tuneLength = 2)

plot(overfit_rlr)                     
overfit_rlr$results
rlr_preds1 <- predict(overfit_rlr, newdata = overfit_test)
rlr_preds_df1 <- data.frame(id = overfit_test$id, target = rlr_preds1)

write_csv(x = rlr_preds_df1, path = "./RLRPreds1.csv")

#### GLMNET ####

tc <- trainControl(method = "cv", number = 10)

tg <- expand.grid(alpha = 1,
                  lambda = seq(0.015, 0.035, by = 0.0001)) 

overfit_glmnet <- train(form = target~.,
                     data = overfit_train %>% select(-id),
                     method = "glmnet",
                     trControl = tc,
                     tuneGrid = tg,
                     tuneLength = 2)

plot(overfit_glmnet)
glmnet_vi <- varImp(overfit_rrf, scale = FALSE)
glmnet_vi

overfit_glmnet$bestTune
glmnet_preds1 <- predict(overfit_rlr, newdata = overfit_test)
glmnet_preds_df1 <- data.frame(id = overfit_test$id, target = glmnet_preds1)

write_csv(x = rlr_preds_df1, path = "./glmnetPreds1.csv")
#### Regularized Random Forest  ####

tc <- trainControl(method = "cv", number = 10)

tg <- expand.grid(mtry = seq(20, 300, by = 20),
                  coefReg = seq(1, 1, by = .2),
                  coefImp = seq(0, .5, by = .1)
)

tictoc::tic()
overfit_rrf <- train(form = target~.,
                        data = overfit_train %>% dplyr::select(-id),
                        method = "RRF",
                        trControl = tc,
                        tuneGrid = tg,
                        tuneLength = 2)
tictoc::toc()
beepr::beep()
plot(overfit_rrf)                     
overfit_rrf$results
rrf_preds1 <- predict(overfit_rrf, newdata = overfit_test)
rrf_preds_df1 <- data.frame(id = overfit_test$id, target = rrf_preds1)

write_csv(x = rrf_preds_df1, path = "./RRFPreds1.csv")
#### RF with variable selection ####

tc <- trainControl(method = "cv", number = 10)

tg <- expand.grid(mtry = seq(0, 300, by = 10),
                  coefReg = seq(1, 1, by = .5),
                  coefImp = seq(0, .5, by = .1)
)

tictoc::tic()
overfit_rrf <- train(form = target~.,
                     data = overfit_train %>% dplyr::select(-id),
                     method = "RRF",
                     trControl = tc,
                     tuneGrid = tg,
                     tuneLength = 2)
tictoc::toc()
beepr::beep()
plot(overfit_rrf)  

rrf_vi <- varImp(overfit_rrf, scale = FALSE)
rrf_vi

updated_rff <- overfit_train %>% select(c(target,"33","65","217","117","91","199","214", "82", "295", "258"
))

tg <- expand.grid(mtry = seq(0, 10, by = 1),
                  coefReg = seq(1, 1, by = .5),
                  coefImp = seq(0, .5, by = .05)
)

tictoc::tic()
overfit_rrf <- train(form = target~.,
                     data = updated_rff,
                     method = "RRF",
                     trControl = tc,
                     tuneGrid = tg,
                     tuneLength = 4)
tictoc::toc()
beepr::beep()
plot(overfit_rrf)

overfit_rrf$results
rrf_preds1 <- predict(overfit_rrf, newdata = overfit_test)
rrf_preds_df1 <- data.frame(id = overfit_test$id, target = rrf_preds1)

write_csv(x = rrf_preds_df1, path = "./RRFPreds1.csv")

#### LASSO VI ####
tc <- trainControl(method = "cv", number = 10)

tg <- expand.grid(alpha = c(0.5,.75,1),
                  lambda = seq(0.001, 0.06, by = 0.00005)
)

overfit_lasso <- train(form = target~.,
                      data = overfit_train,
                      method = "glmnet",
                      trControl= tc,
                      tuneGrid = tg
)

plot(overfit_lasso)  
lasso_vi <- varImp(overit_lasso)
lasso_vi$importance %>% filter(Overall > 0) %>% arrange(desc(Overall))

overfit_vi_train <- overfit_train %>% select(c(target, "33", "65", "217", "91", "199", "117", "73", "295", "189", "80",
                             "194", "258", "108", "133", "82", "16", "134", "43", "90", "298", "237"#, "226",
          #                   "252", "165", "129","101", "127"
))
overfit_lasso_vi <- train(form = target~.,
                      data = overfit_vi_train,
                      method = "glmnet",
                      trControl= tc,
                      tuneGrid = tg
)
plot(overit_lasso)
lasso_preds1 <- predict(overfit_lasso_vi, newdata = overfit_test)
lasso_preds_df1 <- data.frame(id = overfit_test$id, target = lasso_preds1)

write_csv(x = lasso_preds_df1, path = "./lassoPreds1.csv")

