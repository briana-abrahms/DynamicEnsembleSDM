#' Script to calculate Area Under the receiver operator Curve (AUC) and True Skill Statistic (TSS) metrics for
#' GAMM, BRT and Ensemble models, from Abrahms et al. 2019, Diversity and Distributions.
#' Written by Briana Abrahms, Stephanie Brodie, and Heather Welch, 2019.
#'
#' Workflow:
#' 0. Setup: load libraries, assign global objects and filepaths
#' 1. Functions for calculating evaluation metrics for the following training/testing datasets:
#'    1a. Full dataset
#'    1b. k-fold
#'    1c. Leave One Year Out


# 0. ------------------- > Setup ####

#Load libraries
library(mlbench)
library(dismo)
library(gbm)
library(lubridate)
library(mgcv)
library(qmap)
library(dplyr)

#Load global objects
DataInput_Fit <- read.csv("tracks.csv") 
DataInput_Fit$ptt <- as.factor(DataInput_Fit$ptt)
DataInput_Fit$month <- lubridate::month(DataInput_Fit$dt)
DataInput_Fit<-na.omit(DataInput_Fit)

#Load models
source("model_fit.R")


# 1. ------------------- > Functions for calculating evaluation metrics ####

#### 1a. ------------------- > Full tracking dataset #########

eval_100_BRT <- function(model, months = c(1:12)) {
  Evaluations_100_BRT <- as.data.frame(matrix(data=0,nrow=1,ncol=2))
  colnames(Evaluations_100_BRT) <- c("AUC","TSS")
  DataInput_test <- DataInput_Fit[DataInput_Fit$month %in% months, ]
  preds<-predict.gbm(model, DataInput_test,n.trees=model$gbm.call$best.tree, type="response")
  d <- cbind(DataInput_test$presabs, preds)
  pres <- as.numeric(d[d[,1]==1,2])
  abs <- as.numeric(d[d[,1]==0,2])
  e <- evaluate(p=pres, a=abs)
  Evaluations_100_BRT[1,1] <- e@auc
  Evaluations_100_BRT[1,2] <- max(e@TPR + e@TNR-1)
  return(Evaluations_100_BRT)
} 

#demo BRT: eval_100_BRT(brt)

eval_100_GAMM <- function(dataInput, formula, months = c(1:12)) {
  Evaluations_100_GAMM <- as.data.frame(matrix(data=0,nrow=1,ncol=2))
  colnames(Evaluations_100_GAMM) <- c("AUC","TSS")
  DataInput_train <- dataInput[dataInput$month %in% months,]
  DataInput.100 <- mgcv::gam(formula=formula,data=DataInput_train, family=binomial,method="REML")
  DataInput_test <- dataInput[dataInput$month %in% months,]
  preds<-as.data.frame(predict.gam(DataInput.100, DataInput_test, se=TRUE, type="response"))
  d <- cbind(DataInput_test$presabs, preds)
  pres <- as.numeric(d[d[,1]==1,2])
  abs <- as.numeric(d[d[,1]==0,2])
  e <- evaluate(p=pres, a=abs)
  Evaluations_100_GAMM[1,1] <- e@auc
  Evaluations_100_GAMM[1,2] <- max(e@TPR + e@TNR-1)
  return(Evaluations_100_GAMM)
} 

#demo GAMM: eval_100_GAMM(DataInput_Fit, formula = presabs~s(sst, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ssh_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+s(ptt, bs = "re"))

eval_100_Ensemble <- function(dataInput, BRT_model, GAMM_formula, months = c(1:12)) {
  #setup
  Evaluations_100_Ensemble <- as.data.frame(matrix(data=0,nrow=1,ncol=2))
  colnames(Evaluations_100_Ensemble) <- c("AUC","TSS")
  DataInput_train <- dataInput[dataInput$month %in% months,]
  DataInput_test <- dataInput[dataInput$month %in% months,]
  DataInput.100 <- mgcv::gam(formula=GAMM_formula,data=DataInput_train, family=binomial,method="REML")
  
  #ensemble average predictions from models
  preds_GAMM<-as.data.frame(predict.gam(DataInput.100, DataInput_test, se=TRUE, type="response"))
  preds_BRT<-predict.gbm(BRT_model, DataInput_test,n.trees=BRT_model$gbm.call$best.tree, type="response")
  preds_Ensemble <- as.data.frame(rowMeans(cbind(preds_BRT, preds_GAMM$fit), na.rm = TRUE))
    
  #evaluate
  d <- cbind(DataInput_test$presabs, preds_Ensemble)
  pres <- as.numeric(d[d[,1]==1,2])
  abs <- as.numeric(d[d[,1]==0,2])
  e <- evaluate(p=pres, a=abs)
  Evaluations_100_Ensemble[1,1] <- e@auc
  Evaluations_100_Ensemble[1,2] <- max(e@TPR + e@TNR-1)
  return(Evaluations_100_Ensemble)
} 

#demo BRT-GAMM ensemble: eval_100_Ensemble(DataInput_Fit, BRT_model = brt, GAMM_formula = presabs~s(sst, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ssh_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+s(ptt, bs = "re"))



#### 1b. ------------------- >  K-fold data (k=5) #########

eval_kfold_BRT <- function(dataInput, gbm.x, gbm.y, lr=0.05, months=c(1:12)){
  DataInput <- dataInput[dataInput$month %in% months,]
  DataInput$Kset <- kfold(DataInput,5) #randomly allocate k groups
  Evaluations_kfold_BRT <- as.data.frame(matrix(data=0,nrow=5,ncol=4))
  colnames(Evaluations_kfold_BRT) <- c("k","Deviance","AUC","TSS")
  counter=1
  for (k in 1:5){
    print(k)
    DataInput_train <- DataInput[DataInput$Kset!=k,]
    DataInput_test <- DataInput[DataInput$Kset==k,]
    DataInput.kfolds <- gbm.step(data=DataInput_train, gbm.x= gbm.x, gbm.y = gbm.y, 
                                 family="bernoulli", tree.complexity=3,
                                 learning.rate = lr, bag.fraction = 0.6)
    preds <- predict.gbm(DataInput.kfolds, DataInput_test,
                         n.trees=DataInput.kfolds$gbm.call$best.trees, type="response")
    dev <- calc.deviance(obs=DataInput_test$presabs, pred=preds, calc.mean=TRUE)
    d <- cbind(DataInput_test$presabs, preds)
    pres <- as.numeric(d[d[,1]==1,2])
    abs <- as.numeric(d[d[,1]==0,2])
    e <- dismo::evaluate(p=pres, a=abs)
    Evaluations_kfold_BRT[counter,1] <- k
    Evaluations_kfold_BRT[counter,2] <- dev
    Evaluations_kfold_BRT[counter,3] <- e@auc
    Evaluations_kfold_BRT[counter,4] <- max(e@TPR + e@TNR-1)
    counter=counter+1 
  }
  return(Evaluations_kfold_BRT)}

#demo BRT: eval_kfold_BRT(DataInput_Fit, gbm.x=c("curl","ild", "ssh", "sst","sst_sd", "ssh_sd", "z", "z_sd", "EKE","slope","aspect","BV"), "presabs")

eval_kfold_GAMM <- function(dataInput, months=c(1:12),formula){
  DataInput <- dataInput[dataInput$month %in% months,]
  DataInput$ptt <- as.numeric(DataInput$ptt) #as needed
  DataInput$Kset <- kfold(DataInput,5) #randomly allocate k groups
  Evaluations_kfold_GAMM <- as.data.frame(matrix(data=0,nrow=5,ncol=3))
  colnames(Evaluations_kfold_GAMM) <- c("k","AUC","TSS")
  counter=1
  for (k in 1:5){
    print(k)
    DataInput_train <- DataInput[DataInput$Kset!=k,]
    DataInput_test <- DataInput[DataInput$Kset==k,]
    DataInput.kfolds <- mgcv::gam(formula=formula,data=DataInput_train, family=binomial,method="REML")
    preds <- as.data.frame(predict.gam(DataInput.kfolds, DataInput_test,se=TRUE, type="response"))
    d <- cbind(DataInput_test$presabs, preds)
    pres <- as.numeric(d[d[,1]==1,2])
    abs <- as.numeric(d[d[,1]==0,2])
    e <- evaluate(p=pres, a=abs)
    Evaluations_kfold_GAMM[counter,1] <- k
    Evaluations_kfold_GAMM[counter,2] <- e@auc
    Evaluations_kfold_GAMM[counter,3] <- max(e@TPR + e@TNR-1)
    counter=counter+1 
  }
  return(Evaluations_kfold_GAMM)}

#demo GAMM: eval_kfold_GAMM(DataInput_Fit, formula = presabs~s(sst, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ssh_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+s(ptt, bs = "re"))

eval_kfold_Ensemble <- function(dataInput, gbm.x, gbm.y, lr=0.05, GAMM_formula, months = c(1:12)) {
  #setup
  DataInput <- dataInput[dataInput$month %in% months,]
  DataInput$ptt <- as.numeric(DataInput$ptt) #as needed
  DataInput$Kset <- kfold(DataInput,5) #randomly allocate k groups
  Evaluations_kfold_Ensemble <- as.data.frame(matrix(data=0,nrow=5,ncol=3))
  colnames(Evaluations_kfold_Ensemble) <- c("k","AUC","TSS")
  counter=1
  for (k in 1:5){
    print(k)
    DataInput_train <- DataInput[DataInput$Kset!=k,]
    DataInput_test <- DataInput[DataInput$Kset==k,]
    DataInput.kfold.gam <- mgcv::gam(formula=GAMM_formula,data=DataInput_train, family=binomial,
                                  method="REML")
    DataInput.kfold.brt <- gbm.step(data=DataInput_train, gbm.x= gbm.x, gbm.y = gbm.y, 
                                 family="bernoulli", tree.complexity=3,
                                 learning.rate = lr, bag.fraction = 0.6)
  
    #ensemble average predictions from models
    preds_GAMM<-as.data.frame(predict.gam(DataInput.kfold.gam, DataInput_test, se=TRUE, type="response"))
    preds_BRT<- predict.gbm(DataInput.kfold.brt, DataInput_test,
                            n.trees=DataInput.kfold.brt$gbm.call$best.trees, type="response")
    preds_Ensemble <- as.data.frame(rowMeans(cbind(preds_BRT, preds_GAMM$fit), na.rm = TRUE))
    
    #evaluate
    d <- cbind(DataInput_test$presabs, preds_Ensemble)
    pres <- as.numeric(d[d[,1]==1,2])
    abs <- as.numeric(d[d[,1]==0,2])
    e <- evaluate(p=pres, a=abs)
    Evaluations_kfold_Ensemble[counter,1] <- k
    Evaluations_kfold_Ensemble[counter,2] <- e@auc
    Evaluations_kfold_Ensemble[counter,3] <- max(e@TPR + e@TNR-1)
    counter=counter+1 
    }
    return(Evaluations_kfold_Ensemble)
} 

#demo BRT-GAMM ensemble: eval_kfold_Ensemble(DataInput_Fit, gbm.x=c("curl","ild", "ssh", "sst","sst_sd", "ssh_sd", "z", "z_sd", "EKE","slope","aspect","BV"), "presabs", GAMM_formula = presabs~s(sst, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ssh_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+s(ptt, bs = "re"))



#### 1c. ------------------- > Leave Year Out data #########

eval_LOO_BRT <- function(dataInput, gbm.x, gbm.y, lr=0.05, months=c(1:12)){
  DataInput <- dataInput[dataInput$month %in% months,]
  DataInput$Year <- lubridate::year(DataInput$dt)
  Evaluations_LOO_BRT <- as.data.frame(matrix(data=0,nrow=1,ncol=4))
  colnames(Evaluations_LOO_BRT) <- c("k","Deviance","AUC","TSS")
  counter=1
  for (y in unique(DataInput$Year)){
    if(any(months %in% DataInput[DataInput$Year==y & DataInput$presabs==1,]$month)==FALSE) next #skip year if no months in dataset are in months vector
    print(y)
    DataInput_train <- DataInput[DataInput$Year!=y,]
    DataInput_test <- DataInput[DataInput$Year==y,]
    DataInput.loo <- gbm.step(data=DataInput_train, gbm.x= gbm.x, gbm.y = c("presabs"), 
                              family="bernoulli", tree.complexity=3,
                              learning.rate = lr, bag.fraction = 0.6)
    preds <- predict.gbm(DataInput.loo, DataInput_test,
                         n.trees=DataInput.loo$gbm.call$best.trees, type="response")
    dev <- calc.deviance(obs=DataInput_test$presabs, pred=preds, calc.mean=TRUE)
    d <- cbind(DataInput_test$presabs, preds)
    pres <- as.numeric(d[d[,1]==1,2])
    abs <- as.numeric(d[d[,1]==0,2])
    e <- dismo::evaluate(p=pres, a=abs)
    
    Evaluations_LOO_BRT[counter,1] <- y
    Evaluations_LOO_BRT[counter,2] <- dev
    Evaluations_LOO_BRT[counter,3] <- e@auc
    Evaluations_LOO_BRT[counter,4] <- max(e@TPR + e@TNR-1)
    counter=counter+1 
  }
  return(Evaluations_LOO_BRT)}

#demo BRT: eval_LOO_BRT(DataInput_Fit, gbm.x=c("curl","ild", "ssh", "sst","sst_sd", "ssh_sd", "z", "z_sd", "EKE","slope","aspect","BV"), "presabs")

eval_LOO_GAMM <- function(dataInput, formula, months=c(1:12)){
  DataInput <- dataInput[dataInput$month %in% months,]
  DataInput$ptt <- as.numeric(DataInput$ptt)
  DataInput$Year <- lubridate::year(DataInput$dt)
  Evaluations_LOO_GAMM <- as.data.frame(matrix(data=0,nrow=1,ncol=3))
  colnames(Evaluations_LOO_GAMM) <- c("k","AUC","TSS")
  counter=1
  for (y in unique(DataInput$Year)){
    #skip year if no months in dataset are in months vector
    if(any(months %in% DataInput[DataInput$Year==y & DataInput$presabs==1,]$month)==FALSE |
       any(months %in% DataInput[DataInput$Year==y & DataInput$presabs==0,]$month)==FALSE) 
      next 
    DataInput_train <- DataInput[DataInput$Year!=y,]
    DataInput_test <- DataInput[DataInput$Year==y,]
    print(y)
    DataInput.loo <- mgcv::gam(formula=formula,data=DataInput_train, family=binomial,method="REML")
    preds <- as.data.frame(predict.gam(DataInput.loo, DataInput_test,se=TRUE, type="response"))
    d <- cbind(DataInput_test$presabs, preds)
    pres <- as.numeric(d[d[,1]==1,2])
    abs <- as.numeric(d[d[,1]==0,2])
    e <- evaluate(p=pres, a=abs)
    Evaluations_LOO_GAMM[counter,1] <- y
    Evaluations_LOO_GAMM[counter,2] <- e@auc
    Evaluations_LOO_GAMM[counter,3] <- max(e@TPR + e@TNR-1)
    counter=counter+1 
  }
  return(Evaluations_LOO_GAMM)}

#demo GAMM: eval_LOO_GAMM(DataInput_Fit, formula = presabs~s(sst, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ssh_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+s(ptt, bs = "re"))

eval_LOO_Ensemble <- function(dataInput, gbm.x, gbm.y, lr=0.05, GAMM_formula, months = c(1:12)) {
  #setup
  DataInput <- dataInput[dataInput$month %in% months,]
  DataInput$ptt <- as.numeric(DataInput$ptt) #as needed
  DataInput$Year <- lubridate::year(DataInput$dt)
  Evaluations_LOO_Ensemble <- as.data.frame(matrix(data=0,nrow=1,ncol=3))
  colnames(Evaluations_LOO_Ensemble) <- c("y","AUC","TSS")
  counter=1
  for (y in unique(DataInput$Year)){
    #skip year if no months in dataset are in months vector
    if(any(months %in% DataInput[DataInput$Year==y & DataInput$presabs==1,]$month)==FALSE |
       any(months %in% DataInput[DataInput$Year==y & DataInput$presabs==0,]$month)==FALSE) 
      next 
    DataInput_train <- DataInput[DataInput$Year!=y,]
    DataInput_test <- DataInput[DataInput$Year==y,]
    print(y)
    DataInput.LOO.gam <- mgcv::gam(formula=GAMM_formula,data=DataInput_train, family=binomial,
                                     method="REML")
    DataInput.LOO.brt <- gbm.step(data=DataInput_train, gbm.x= gbm.x, gbm.y = gbm.y, 
                                    family="bernoulli", tree.complexity=3,
                                    learning.rate = lr, bag.fraction = 0.6)
    
    #ensemble average predictions from models
    preds_GAMM<-as.data.frame(predict.gam(DataInput.LOO.gam, DataInput_test, se=TRUE, type="response"))
    preds_BRT<- predict.gbm(DataInput.LOO.brt, DataInput_test,
                            n.trees=DataInput.LOO.brt$gbm.call$best.trees, type="response")
    preds_Ensemble <- as.data.frame(rowMeans(cbind(preds_BRT, preds_GAMM$fit), na.rm = TRUE))
    
    #evaluate
    d <- cbind(DataInput_test$presabs, preds_Ensemble)
    pres <- as.numeric(d[d[,1]==1,2])
    abs <- as.numeric(d[d[,1]==0,2])
    e <- evaluate(p=pres, a=abs)
    Evaluations_LOO_Ensemble[counter,1] <- y
    Evaluations_LOO_Ensemble[counter,2] <- e@auc
    Evaluations_LOO_Ensemble[counter,3] <- max(e@TPR + e@TNR-1)
    counter=counter+1 
  }
  return(Evaluations_LOO_Ensemble)
} 

#demo BRT-GAMM ensemble: eval_LOO_Ensemble(DataInput_Fit, gbm.x=c("curl","ild", "ssh", "sst","sst_sd", "ssh_sd", "z", "z_sd", "EKE","slope","aspect","BV"), "presabs", GAMM_formula = presabs~s(sst, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ssh_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+s(ptt, bs = "re"))
