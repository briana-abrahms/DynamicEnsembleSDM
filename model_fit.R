#' Code to fit Generalized Additive Mixed Models (GAMMs) and Boosted Regression Trees (BRTs)
#' for blue whale habitat suitability from Abrahms et al. 2019, Diversity and Distributions.
#' This file can be used as source code to feed into the model_evaluation.R script.
#' Written by Briana Abrahms, 2019.
#' 
#' Workflow:
#' 0. Setup: load libraries
#' 1. Load tracking data
#' 2. Fit GAMMs
#' 3. Fit BRT


# 0. ------------------- > Load libraries ####
library(mgcv) #GAMMs
library(dismo) #BRTs
library(lubridate) #datetimes


# 1. ------------------- > Load and prep data ####
# Tracking data provided is a subset of 10 randomly selected individuals
# Response variable name is 'presabs' for Presence-Absence
tracks <- read.csv("tracks.csv") 
tracks$ptt <- as.factor(tracks$ptt)
tracks$month <- lubridate::month(tracks$dt)
tracks<-na.omit(tracks)


# 2. ------------------- > Fit candidate GAMMs ####
# Individuals (variable name 'ptt') are nested as a random effect

# sst,z,z_sd,ssh_sd,ild,eke (Model 1)
gam.mod1 <- mgcv::gam(presabs~s(sst, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ssh_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+s(ptt, bs = "re"),family=binomial, data=tracks, method = "REML", select = T)

# sst,z,z_sd,ssh_sd,ild,eke,te(lon,lat) (Model 2)
gam.mod2 <- mgcv::gam(presabs ~ s(sst, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ssh_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+te(lon,lat,bs="ts")+s(ptt, bs = "re"),family=binomial, data=tracks, method = "REML", select = T)

# z,z_sd,ssh_sd,ild,eke,te(sst,lat) (Model 3)
gam.mod3 <- mgcv::gam(presabs ~ s(z, bs="ts")+s(z_sd, bs="ts")+s(ssh_sd, bs="ts")+s(ild, bs="ts")+s(EKE, bs="ts")+te(sst, lat,bs="ts")+s(ptt, bs = "re"),family=binomial, data=tracks, method = "REML", select = T)

# sst, z,z_sd
gam.mod4 <- mgcv::gam(presabs ~ s(sst, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ptt, bs = "re"),family=binomial, data=tracks, method = "REML", select = T)

# curl,sst,ssh,ssh_sd, sst_sd,z,z_sd,ild,eke
gam.mod5 <- mgcv::gam(presabs ~ s(curl, bs="ts")+s(sst, bs="ts")+s(ssh, bs="ts")+s(ssh_sd, bs="ts")+s(sst_sd, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+ s(ild, bs="ts")+s(EKE, bs="ts")+s(ptt, bs = "re"),family=binomial, data=tracks, method = "REML", select = T)

# sst, ssh_sd, z, z_sd, ild, eke, slope
gam.mod6 <- mgcv::gam(presabs~s(sst, bs="ts")+s(ssh_sd, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+ s(slope, bs="ts")+s(ptt, bs = "re"),family=binomial, data=tracks, method = "REML", select = T)

# sst, ssh_sd, z, z_sd, ild, eke, aspect
gam.mod7 <- mgcv::gam(presabs~s(sst, bs="ts")+s(ssh_sd, bs="ts")+s(z, bs="ts")+s(z_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+s(aspect, bs="ts")+s(ptt, bs = "re"),family=binomial, data=tracks, method = "REML", select = T)

# sst, ssh_sd, z*slope, z_sd, ild, eke
gam.mod8 <- mgcv::gam(presabs~s(sst, bs="ts")+s(ssh_sd, bs="ts")+s(z, slope,bs="ts")+ s(z_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+s(ptt, bs = "re"),family=binomial, data=tracks, method = "REML", select = T)

# bv, ssh_sd, z*slope, z_sd, ild, eke
gam.mod9 <- mgcv::gam(presabs~s(BV, bs="ts")+s(ssh_sd, bs="ts")+s(z, slope,bs="ts")+ s(z_sd, bs="ts")+s(ild, bs="ts")+ s(EKE, bs="ts")+s(ptt, bs = "re"),family=binomial, data=tracks, method = "REML", select = T)

# sst, ssh_sd, z, z_sd, ild, eke, month
gam.mod10<-mgcv::gam(presabs~s(sst,bs="ts")+s(ssh_sd,bs="ts")+s(z,bs="ts")+s(z_sd,bs="ts")+s(ild,bs="ts")+ s(EKE, bs="ts")+s(month, bs="cc") + s(ptt, bs = "re"),family=binomial, data=tracks, method = "REML", select = T)


# 3. ------------------- > Fit BRT
#Includes all covariates: sst, sst_sd, ssh, ssh_sd,z,z_sd,ild, eke, curl, bv, slope, aspect

brt <- dismo::gbm.step(data=tracks, 
               gbm.x=c("curl","ild", "ssh", "sst","sst_sd", "ssh_sd", "z", "z_sd", "EKE","slope","aspect","BV"), 
               gbm.y="presabs", ### response variable
               family = "bernoulli",
               tree.complexity = 3, ### complexity of the interactions that the model will fit
               learning.rate = 0.05,  ### optimized to end up with >1000 trees
               bag.fraction = 0.6) ### recommended by Elith, amount of input data used each time
