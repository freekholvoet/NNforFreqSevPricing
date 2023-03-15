# ----- SETUP R -----

# All setup is done in this section
# Installing and loading all packages, setting up tensorflow and keras
# Reading in data and small data prep
# Define metrics for later use

## ----- Install packages needed -----

#library(reticulate)
#use_python("C:/Users/Frynn/.conda/envs/tf_noGpu/python")
#reticulate::use_condaenv("my_env")

# Make sure the installation of the GBM package is the one from Harry Southworth which includes Gamma deviance for GBM
#devtools::install_github("harrysouthworth/gbm")

used_packages <- c("sp", "vip","ggplot2",
                   "pdp","cplm","mltools",
                   "data.table", "tidyverse",
                   "gtools", "beepr",
                   "gridExtra", "cowplot", "RColorBrewer",
                   "fuzzyjoin", "colorspace", "sf",
                   "tmap", "rgdal","egg", 
                   "tcltk", "xtable","progress",
                   "doParallel","pbapply", "gam")
suppressMessages(packages <- lapply(used_packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x)
    library(x, character.only = TRUE)
  }
}))

## ----- Read in data -----

# Data files input and results from Henckaerts et al. 2019
#setwd("~/Dropbox/Freek research project/Code Freek/Code_FH")
#setwd("C:/Users/u0086713/Dropbox/Freek research project/Code Freek/Code_FH")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Location of the extra data files
location_datasets <- "/home/lynn/Dropbox/MTPL Data Sets"
#location_datasets <- "C:/Users/u0086713/Dropbox/MTPL Data Sets"

# Read in Functions File
source("Functions.R")

# Read in raw input data for each country
load("data_AUS_prepared.RData")
load("data_BE_prepared.RData")
load("data_FR_prepared.RData")
load("data_NOR_prepared.RData")

# Read in Functions File
source("Functions.R")

## ----- Prediction en loss functions -----

predict_model <- function(object, newdata) UseMethod('predict_model')

predict_model.gbm <- function(object, newdata) {
  predict(object, newdata, n.trees = object$n.trees, type = 'response')
}

dev_poiss <- function(ytrue, yhat) {
  -2 * mean(dpois(ytrue, yhat, log = TRUE) - dpois(ytrue, ytrue, log = TRUE), na.rm = TRUE)
}

dev_gamma <- function(ytrue, yhat, wcase) {
  -2 * mean(wcase * (log(ytrue/yhat) - (ytrue - yhat)/yhat), na.rm = TRUE)
}

# -----
# ----- GBM -----

# For each data set, and each test fold, we tune a GBM
# With optimal tuning parameters we fit a GBM to calculate out-of-sample predictions and performance

## ----- Tuning Functions -----

# Define grid op tuning parameter options
tuning_grid <- expand.grid(T = seq(100,5000,by=200), d = 1:10)

# Function for tuning frequency models in cross-validation
tune_GBM_freq <- function(data, data_feat, tune_grid, data_folds){
  print(paste("Now tuning for data set",deparse(substitute(data))))
  
  cl <- parallel::makeCluster(parallel::detectCores()-1)
  clusterExport(cl, c('data', 'data_feat', 'tune_grid', 'data_folds'), envir=environment())
  clusterExport(cl, c('predict_model', 'predict_model.gbm', 'dev_poiss'))
  clusterEvalQ(cl, {library(tidyverse)})
  
  results <- parallel::parLapply(cl,data_folds, function(fold){
    
    # Get the data folds in the validation set
    val_folds <- setdiff(data_folds,fold)
    # Extend the tuning grid with the validation fold numbers
    val_tune_grid <- tidyr::expand_grid(tune_grid,Val_fold = val_folds)
    
    val_errors <- apply(val_tune_grid,1,function(param){
      
      data_trn <- data %>% filter(fold_nr != fold & fold_nr != param[3])
      data_val <- data %>% filter(fold_nr == param[3])
      
      gbm_fit <- gbm::gbm(
        formula = as.formula(paste('nclaims ~ offset(log(expo)) +', paste(data_feat, collapse = ' + '))),
        data = data_trn,
        distribution = 'poisson',
        n.trees = param[1],
        interaction.depth = param[2],
        shrinkage = 0.01,
        bag.fraction = 0.75,
        n.minobsinnode = 0.01 * 0.75 * nrow(data),
        verbose = FALSE
      )
      return(dev_poiss(data_val$nclaims, gbm_fit %>% predict_model(newdata = data_val) * data_val$expo))
    })
    bind_cols(val_tune_grid, "Var_Errors.{fold}" := val_errors)
  }) %>% reduce(full_join, by = c("T","d","Val_fold")) %>% arrange(T,d,Val_fold)
  stopCluster(cl)
  return(results)
}

# Function for tuning severity models in cross-validation
tune_GBM_sev <- function(data, data_feat, tune_grid, data_folds){
  print(paste("Now tuning for data set",deparse(substitute(data))))
  print(paste("The time is:",Sys.time()))
  
  results <- lapply(data_folds, function(fold){
    
    # Get the data folds in the validation set
    val_folds <- setdiff(data_folds,fold)
    # Extend the tuning grid with the validation fold numbers
    val_tune_grid <- tidyr::expand_grid(tune_grid,Val_fold = val_folds)
    
    cl <- parallel::makeCluster(parallel::detectCores()-1)
    clusterExport(cl, c('data', 'data_feat', 'tune_grid', 'data_folds','fold','val_folds','val_tune_grid'), envir=environment())
    clusterExport(cl, c('predict_model', 'predict_model.gbm', 'dev_gamma'))
    clusterEvalQ(cl, {library(tidyverse)})

    val_errors <- parallel::parApply(cl,val_tune_grid,1,function(param){
      
      data_trn <- data %>% filter(nclaims > 0) %>% filter(fold_nr != fold & fold_nr != param[3])
      data_val <- data %>% filter(nclaims > 0) %>% filter(fold_nr == param[3])
      
      gbm_fit <- gbm::gbm(
        formula = as.formula(paste('average  ~ ', paste(data_feat, collapse = ' + '))),
        data = data_trn,
        weights = nclaims,
        distribution = 'gamma',
        n.trees = param[1],
        interaction.depth = param[2],
        shrinkage = 0.01,
        bag.fraction = 0.75,
        n.minobsinnode = 0.01 * 0.75 * nrow(data),
        verbose = FALSE
      )
      return(dev_gamma(data_val$average, gbm_fit %>% predict_model(newdata = data_val), data_val$nclaims))
    })
    stopCluster(cl)
    return(bind_cols(val_tune_grid, "Var_Errors.{fold}" := val_errors))
  })
  output <- results %>% reduce(full_join, by = c("T","d","Val_fold")) %>% arrange(T,d,Val_fold)
  print(paste("Run is finished at:",Sys.time()))
  return(output)
}

## ----- Tuning for all data sets -----

# Tuning for all data folds - Frequency model
AUS_tuning_GBM_freq <- tune_GBM_freq(data_AUS, feat_AUS, tuning_grid, data_folds = 1:6)
save(AUS_tuning_GBM_freq, file='AUS_tuning_GBM_freq')

FR_tuning_GBM_freq <- tune_GBM_freq(data_FR, feat_FR, tuning_grid, data_folds = 1:6)
save(FR_tuning_GBM_freq, file='FR_tuning_GBM_freq')

NOR_tuning_GBM_freq <- tune_GBM_freq(data_NOR, feat_NOR, tuning_grid, data_folds = 1:6)
save(NOR_tuning_GBM_freq, file='NOR_tuning_GBM_freq')

# Tuning for all data sets - Frequency model
AUS_tuning_GBM_sev <- tune_GBM_sev(data_AUS, feat_AUS, tuning_grid, data_folds = 1:6)
save(AUS_tuning_GBM_sev, file='AUS_tuning_GBM_sev')

FR_tuning_GBM_sev <- tune_GBM_sev(data_FR %>% filter(!is.na(average)), feat_FR, tuning_grid, data_folds = 1:6)
save(FR_tuning_GBM_sev, file='FR_tuning_GBM_sev')

NOR_tuning_GBM_sev <- tune_GBM_sev(data_NOR, feat_NOR, tuning_grid, data_folds = 1:6)
save(NOR_tuning_GBM_sev, file='NOR_tuning_GBM_sev')

# -----
# ----- OUT OF SAMPLE -----

# Read in optimal tuning parameters
# Fit a GBM for each dataset and testfold to calculate OOS

## ----- Read in tuning results -----

load('AUS_tuning_GBM_freq')
load('AUS_tuning_GBM_sev')

load('FR_tuning_GBM_freq')
load('FR_tuning_GBM_sev')

load('NOR_tuning_GBM_freq')
load('NOR_tuning_GBM_sev')

## ----- Get optimal tuning parameters -----

## Function to get optimal tuning parameters from tuning runs
get_opt_param <- function(tuning_results){
  summariser_per_testfold <- tuning_results %>% 
    group_by(T,d) %>% 
    summarise(across(everything(),funs(mean(., na.rm = TRUE)))) %>% 
    select(!Val_fold_mean) %>% 
    rename_with(~ c(1:6)[which(paste0('Var_Errors.', 1:6,'_mean') == .x)], .cols = paste0('Var_Errors.', 1:6,'_mean')) %>% 
    gather(- T, -d, key = 'Test_Fold', value = 'Cross_Val_Error') %>% mutate(Test_Fold = as.numeric(Test_Fold))
  
  lapply(1:6, function(x){
    summariser_per_testfold %>% 
      filter(Test_Fold == x) %>% 
      ungroup() %>% 
      slice(which.min(Cross_Val_Error))
    })
}

# Subtract optimal tuning parameters for each data set
opt_tuning_freq_AUS <- get_opt_param(AUS_tuning_GBM_freq)
opt_tuning_sev_AUS <- get_opt_param(AUS_tuning_GBM_sev)

opt_tuning_freq_FR <- get_opt_param(FR_tuning_GBM_freq)
opt_tuning_sev_FR <- get_opt_param(FR_tuning_GBM_sev)

opt_tuning_freq_NOR <- get_opt_param(NOR_tuning_GBM_freq)
opt_tuning_sev_NOR <- get_opt_param(NOR_tuning_GBM_sev)

## ----- Fit a GBM and calculate OOS performance -----

fit_freq_GBM <- function(data, data_feat, parameters, data_folds){
  lapply(data_folds, function(fold){
    
    data_trn <- data %>% filter(fold_nr != fold)

    param <- parameters[[fold]][,1:2]
    
    gbm_fit <- gbm::gbm(
      formula = as.formula(paste('nclaims ~ offset(log(expo)) +', paste(data_feat, collapse = ' + '))),
      data = data_trn,
      distribution = 'poisson',
      n.trees = param[1],
      interaction.depth = param[2],
      shrinkage = 0.01,
      bag.fraction = 0.75,
      n.minobsinnode = 0.01 * 0.75 * nrow(data_trn),
      verbose = FALSE
    )
    
    data_GBM <- bind_cols(data, prediction = gbm_fit %>% predict_model(newdata = data) * data$expo)
    
    return(list(data_GBM, dev_poiss(data_GBM %>% filter(fold_nr==fold) %>% pull(nclaims), 
                                    data_GBM %>% filter(fold_nr==fold) %>% pull(prediction)),
                gbm_fit))
  })
}
fit_sev_GBM <- function(data, data_feat, parameters, data_folds){
  lapply(data_folds, function(fold){
    
    data_trn <- data %>% filter(fold_nr != fold)
    
    param <- parameters[[fold]][,1:2]
    
    gbm_fit <- gbm::gbm(
      formula = as.formula(paste('average  ~ ', paste(data_feat, collapse = ' + '))),
      data = data_trn,
      weights = nclaims,
      distribution = 'gamma',
      n.trees = param[1],
      interaction.depth = param[2],
      shrinkage = 0.01,
      bag.fraction = 0.75,
      n.minobsinnode = 0.01 * 0.75 * nrow(data_trn),
      verbose = FALSE
    )
    
    data_GBM <- bind_cols(data, prediction = gbm_fit %>% predict_model(newdata = data))
    
    return(list(data_GBM, dev_gamma(data_GBM %>% filter(fold_nr==fold) %>% pull(average), 
                                    data_GBM %>% filter(fold_nr==fold) %>% pull(prediction), 
                                    data_GBM %>% filter(fold_nr==fold) %>% pull(nclaims)),
                gbm_fit))
  })
}

oos_freq_GBM_AUS <- fit_freq_GBM(data_AUS, feat_AUS, opt_tuning_freq_AUS, 1:6)
oos_sev_GBM_AUS <- fit_sev_GBM(data_AUS %>% filter(nclaims > 0) %>% filter(!is.na(average)), feat_AUS, opt_tuning_freq_AUS, 1:6)

oos_freq_GBM_FR <- fit_freq_GBM(data_FR, feat_FR, opt_tuning_freq_FR, 1:6)
oos_sev_GBM_FR <- fit_sev_GBM(data_FR %>% filter(nclaims > 0) %>% filter(!is.na(average)), feat_FR, opt_tuning_freq_FR, 1:6)

oos_freq_GBM_NOR <- fit_freq_GBM(data_NOR, feat_NOR, opt_tuning_freq_NOR, 1:6)
oos_sev_GBM_NOR <- fit_sev_GBM(data_NOR %>% filter(nclaims > 0) %>% filter(!is.na(average)), feat_NOR, opt_tuning_freq_NOR, 1:6)

save(oos_freq_GBM_AUS, file = 'oos_freq_GBM_AUS')
save(oos_sev_GBM_AUS, file = 'oos_sev_GBM_AUS')
save(oos_freq_GBM_FR, file = 'oos_freq_GBM_FR')
save(oos_sev_GBM_FR, file = 'oos_sev_GBM_FR')
save(oos_freq_GBM_NOR, file = 'oos_freq_GBM_NOR')
save(oos_sev_GBM_NOR, file = 'oos_sev_GBM_NOR')

## ----- Put all OOS into table -----

# Add the Belgian OOS from data sets Henckaerts
load("NClaims_datasets.RData")
load("ClaimAmount_datasets.RData")

load('oos_freq_GBM_AUS')
load('oos_sev_GBM_AUS')
load('oos_freq_GBM_FR')
load('oos_sev_GBM_FR')
load('oos_freq_GBM_NOR')
load('oos_sev_GBM_NOR')

oos_freq_GBM_BE <- sapply(1:6, function(x) dev_poiss(NC_all_testfolds_CANNgbm[[x]]$testset$response %>% pull(nclaims), 
                                                     NC_all_testfolds_CANNgbm[[x]]$testset$data %>% pull(prediction)) )
oos_sev_GBM_BE <- sapply(1:6, function(x) dev_gamma(CA_all_testfolds_CANNgbm[[x]]$testset$response %>% pull(average), 
                                                    CA_all_testfolds_CANNgbm[[x]]$testset$data %>% pull(prediction),
                                                    CA_all_testfolds_CANNgbm[[x]]$testset$weights %>% pull(nclaims)) )

# Combine all out-of-sample performances into a table
oos_all_GBMs <- bind_rows(
  bind_cols(Fold = 1:6, Freq = oos_freq_GBM_BE, Sev = oos_sev_GBM_BE) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'BE'),
  bind_cols(Fold = 1:6, Freq = sapply(1:6, function(x) oos_freq_GBM_AUS[[x]][[2]]), Sev = sapply(1:6, function(x) oos_sev_GBM_AUS[[x]][[2]])) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'AUS'),
  bind_cols(Fold = 1:6, Freq = sapply(1:6, function(x) oos_freq_GBM_FR[[x]][[2]]), Sev = sapply(1:6, function(x) oos_sev_GBM_FR[[x]][[2]])) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'FR'),
  bind_cols(Fold = 1:6, Freq = sapply(1:6, function(x) oos_freq_GBM_NOR[[x]][[2]]), Sev = sapply(1:6, function(x) oos_sev_GBM_NOR[[x]][[2]])) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'NOR')
) %>% mutate(Model = 'GBM') %>% select(Model,Data,Problem,Fold,OOS)
save(oos_all_GBMs, file = 'oos_all_GBMs')

oos_all_GBMs %>% ggplot(aes(Fold, OOS, color = Problem)) + geom_line() + facet_wrap(~Data)

# -----
# ----- THE END -----
# -----
