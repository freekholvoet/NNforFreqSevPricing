# ----- SETUP R -----

# All setup is done in this section
# Installing and loading all packages, setting up tensorflow and keras
# Reading in data and small data prep
# Define metrics for later use

## ----- Install packages needed -----

#library(reticulate)
#use_python("C:/Users/Frynn/.conda/envs/tf_noGpu/python")
#reticulate::use_condaenv("my_env")

used_packages <- c("sp", "vip","ggplot2",
                   "pdp","cplm","mltools",
                   "data.table", "keras", "tensorflow",
                   "reticulate", "tidyverse",
                   "gtools", "beepr", "gbm",
                   "gridExtra", "cowplot", "RColorBrewer",
                   "fuzzyjoin", "colorspace", "sf",
                   "tmap", "rgdal","egg", 
                   "tcltk", "xtable","progress",
                   "doParallel")
suppressMessages(packages <- lapply(used_packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x)
    library(x, character.only = TRUE)
  }
}))

#install.packages('devtools')
#devtools::install_github('henckr/maidrr')
#library("maidrr")

## ---- Setup Keras and Tensorflow -----

# Running on local laptop PC
tensorflow::use_condaenv( "tf_noGpu")
conda_python(envname = "tf_noGpu")

# Running on local Desktop PC
#tensorflow::use_condaenv( "my_env")
#conda_python(envname = "my_env")

# Disable graphical plot of model training (to much memory, can cause crash)
options(keras.view_metrics = FALSE)
#options(keras.view_metrics = TRUE)

# Number of significant digits
options(pillar.sigfig = 5)

## ----- Read in Data -----

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

# Load prepared data sets for neural networks
load("ClaimAmount_all_data_sets.RData")
load("NClaims_all_data_sets.RData")

# Read in scaled weights of pre-trained autoencoders
load('AE_weights_scaled_AUS')
load('AE_weights_scaled_BE')
load('AE_weights_scaled_FR')
load('AE_weights_scaled_NOR')

# Read in optimal tuning parameters for each country, each model type
load('optimal_tuning_param_AUS')
load('optimal_tuning_param_BE')
load('optimal_tuning_param_FR')
load('optimal_tuning_param_NOR')

## ----- Loss metrics -----

dev_poiss <- function(y, yhat, w = 1, scaled = TRUE){
  sf <- ifelse(scaled, 1/length(y[!is.na(y)]), 1)
  if(!is.matrix(yhat)) return(-2*sf*sum(w*(dpois(y,yhat,log=TRUE) - dpois(y,y,log=TRUE)), na.rm = TRUE))
  return(-2*sf*colSums(w*(dpois(y,yhat,log=TRUE) - dpois(y,y,log=TRUE)), na.rm = TRUE))
}

dev_poiss_2 <- function(ytrue, yhat) {
  -2 * mean(dpois(ytrue, yhat, log = TRUE) - dpois(ytrue, ytrue, log = TRUE), na.rm = TRUE)
}

poisson_metric <- function(y_true, y_pred){
  K <- backend()
  loss <- 2*K$mean(y_pred - y_true - y_true * (K$log(y_pred) - K$log(y_true+ 0.00000001)))
  loss
}

metric_poisson_metric <- custom_metric("poisson_metric", function(y_true, y_pred) {
  poisson_metric(y_true, y_pred)
})

dev_gamma <- function(y, yhat, weight){
  sum(2*weight*(((y-yhat)/yhat)-log(y/yhat)))/length(y)
}

gamma_metric <- function(){
  gamma <- function(y_true, y_pred) 2 * k_mean(((y_true - y_pred) / y_pred) - k_log(y_true / y_pred))
}

# -----
# ----- FIT MODELS FOR INTERPRETATION -----

# Fit the appropriate model to use for interpretation

## ----- Fitting optimal models -----

# Australian
NC_opt_AUS <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = NC_data_AUS_GBM[[fold]], 
                     flags_list = AUS_NC_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                     cat_vars = cat_AUS,
                     output_modelinfo = TRUE)
})

CA_opt_AUS <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = CA_data_AUS_GBM[[fold]], 
                     flags_list = AUS_CA_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                     cat_vars = cat_AUS,
                     output_modelinfo = TRUE)
})

# Belgian
NC_opt_BE <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = NC_data_BE_GBM[[fold]], 
                     flags_list = BE_NC_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_BE[[fold]],
                     cat_vars = cat_BE,
                     output_modelinfo = TRUE)
})

CA_opt_BE <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = CA_data_BE_GBM[[fold]], 
                     flags_list = BE_CA_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_BE[[fold]],
                     cat_vars = cat_BE,
                     output_modelinfo = TRUE)
})

# French
NC_opt_FR <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = NC_data_FR_GBM[[fold]],
                     flags_list = FR_NC_CANN_GBM_flex[[fold]],
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_FR[[fold]],
                     cat_vars = cat_FR,
                     output_modelinfo = TRUE)
})

CA_opt_FR <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = CA_data_FR_GBM[[fold]],
                     flags_list = FR_CA_CANN_GBM_flex[[fold]],
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_FR[[fold]],
                     cat_vars = cat_FR,
                     output_modelinfo = TRUE)
})

# Norwegian
NC_opt_NOR <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = NC_data_NOR_GBM[[fold]], 
                     flags_list = NOR_NC_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                     cat_vars = cat_NOR,
                     output_modelinfo = TRUE)
})

CA_opt_NOR <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = CA_data_NOR_GBM[[fold]], 
                     flags_list = NOR_CA_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                     cat_vars = cat_NOR,
                     output_modelinfo = TRUE)
})

# -----
# ----- INTERPRETATION PREPERATION -----

# Load in the used models and other information

## ----- Loading necessary info -----

# Load fitted GBMs for AUS, FR and NOR
load('oos_freq_GBM_AUS')
load('oos_sev_GBM_AUS')
load('oos_freq_GBM_FR')
load('oos_sev_GBM_FR')
load('oos_freq_GBM_NOR')
load('oos_sev_GBM_NOR')
gbm_fits<-readRDS("./Data/mfits_gbm.rds") #Gbm's as constructed in Henckaerts et al. (2019)
gbm_fits <- gbm_fits[c(1:6,13:18)] # We do not use the log_normal GBM fits

# Define variables to be scaled
scale_AUS <- setdiff(names(data_AUS),c(cat_AUS,"expo","id","nclaims","average","fold_nr"))
scale_FR <- setdiff(names(data_FR),c(cat_FR,"expo","id","nclaims","average","fold_nr"))
scale_NOR <- setdiff(names(data_NOR),c(cat_NOR,"expo","id","nclaims","average","fold_nr"))
scale_BE <- c("ageph","bm","agec","power","long","lat")

## ----- Determine info about continuous variables -----

# Australian
NC_scaleinfo_AUS <- expand.grid(Variable = scale_AUS, Testfold = 1:6) %>% as_tibble %>% rowwise %>%
  mutate(u = (data_AUS %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% mean),
         sd = (data_AUS %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% sd)) %>% 
  mutate(min = (data_AUS %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% min),
         max = (data_AUS %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% max))

CA_scaleinfo_AUS <- expand.grid(Variable = scale_AUS, Testfold = 1:6) %>% as_tibble %>% rowwise %>%
  mutate(u = (data_AUS %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% mean),
         sd = (data_AUS %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% sd)) %>% 
  mutate(min = (data_AUS %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% min),
         max = (data_AUS %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% max))

# Belgian
NC_scaleinfo_BE <- expand.grid(Variable = scale_BE, Testfold = 1:6) %>% as_tibble %>% rowwise %>%
  mutate(u = (data_BE %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% mean),
         sd = (data_BE %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% sd)) %>% 
  mutate(min = (data_BE %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% min),
         max = (data_BE %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% max))

CA_scaleinfo_BE <- expand.grid(Variable = scale_BE, Testfold = 1:6) %>% as_tibble %>% rowwise %>%
  mutate(u = (data_BE %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% mean),
         sd = (data_BE %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% sd)) %>% 
  mutate(min = (data_BE %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% min),
         max = (data_BE %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% max))

# French
NC_scaleinfo_FR <- expand.grid(Variable = scale_FR, Testfold = 1:6) %>% as_tibble %>% rowwise %>%
  mutate(u = (data_FR %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% mean),
         sd = (data_FR %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% sd)) %>% 
  mutate(min = (data_FR %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% min),
         max = (data_FR %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% max))

CA_scaleinfo_FR <- expand.grid(Variable = scale_FR, Testfold = 1:6) %>% as_tibble %>% rowwise %>%
  mutate(u = (data_FR %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% mean),
         sd = (data_FR %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% sd)) %>% 
  mutate(min = (data_FR %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% min),
         max = (data_FR %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != Testfold) %>% pull(Variable) %>% max))

# Norwegian
NC_scaleinfo_NOR <- NULL
CA_scaleinfo_NOR <- NULL

## ----- Variable labels ----

vars_with_label_AUS <- bind_cols(Variable = c("VehValue", "VehAge", "VehBody", "Gender", "DrivAge"), 
                                xlabels = c("Vehicle value", "Vehicle age", "Vehicle build type", "Policyholder gender","Policyholder age"))

vars_with_label_BE <- bind_cols(Variable = c("ageph", "bm", "agec", "power", "sex", "coverage", "fuel", "use", "fleet", "long", "lat", "latlong"), 
                             xlabels = c("Policyholder age", "Bonus-malus scale", "Vehicle age", 
                                         "Vehicle power", "Policyholder gender", "Coverage type", 
                                         "Fuel type", "Vehicle usage", "Fleet car", 
                                         "Longitude of policy holder residence", "Latitude of policy holder residence", "Postalcode policyholder"))

vars_with_label_FR <- bind_cols(Variable = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"), 
                                 xlabels = c("Vehicle power", "Vehicle age","Policyholder age", "Bonus-malus scale", "Vehicle brand", "Fuel type",
                                             "Area of residence", "Population density of area", "District of residence"))

vars_with_label_NOR <- bind_cols(Variable = c("Male", "Young", "DistLimit", "GeoRegion"), 
                                xlabels = c("Policyholder gender", "Policyholder age", "Distance limit", "Population density of region"))

## ----- GBM prediction function -----

# Functions for easier GBM prediction
predict_model <- function(object, newdata) UseMethod('predict_model')

predict_model.gbm <- function(object, newdata) {
  predict(object, newdata, n.trees = object$n.trees, type = 'response')
}

## ----- Belgium specific postal code preparation ------

# For Belgium we need to add the postal code back to the data
# We want interpretation effects on postal code level, not on latitude or longitude separately

# Add the Postal code back to Belgian data; we want effect of Postal code, not of lat-long separately
data_readin<-readRDS("./Data/Data.rds") #Dataset

data_BE_PC <- data_BE %>% left_join(data_readin %>% select(id,postcode), by = 'id') %>% select(!c(lat, long))

# Complete list of all postal codes in Belgium with lat long of center (for PDP use)
belgium_shape <- readOGR('./shape file Belgie postcodes/npc96_region_Project1.shp') %>% spTransform(CRS('+proj=longlat +datum=WGS84'))
all_latlong <- bind_cols(belgium_shape@data %>% as_tibble %>% select(postcode = POSTCODE), 
                         sp::coordinates(belgium_shape) %>% as_tibble %>% rename(lat = V2, long = V1) )

latlong_per_postalcode <- all_latlong
#latlong_per_postalcode <- data_readin %>% select(postcode, lat, long) %>% unique %>% as_tibble %>% arrange(postcode)

# -----
# ----- PREDICTION FUNCTIONS -----

# Define country and test set specific prediction functions for use with the MAIDRR package 

## ----- Australia -----

NC_pred_AUS_GBM_1 <- function(object, newdata){
  
  fold <- 1
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), NC_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_AUS_GBM_2 <- function(object, newdata){
  
  fold <- 2
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), NC_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_AUS_GBM_3 <- function(object, newdata){
  
  fold <- 3
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), NC_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_AUS_GBM_4 <- function(object, newdata){
  
  fold <- 4
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), NC_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_AUS_GBM_5 <- function(object, newdata){
  
  fold <- 5
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), NC_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_AUS_GBM_6 <- function(object, newdata){
  
  fold <- 6
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), NC_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_AUS_GBM <- list(NC_pred_AUS_GBM_1, NC_pred_AUS_GBM_2, NC_pred_AUS_GBM_3, NC_pred_AUS_GBM_4, NC_pred_AUS_GBM_5, NC_pred_AUS_GBM_6)

CA_pred_AUS_GBM_1 <- function(object, newdata){
  
  fold <- 1
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), CA_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_AUS_GBM_2 <- function(object, newdata){
  
  fold <- 2
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), CA_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_AUS_GBM_3 <- function(object, newdata){
  
  fold <- 3
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), CA_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_AUS_GBM_4 <- function(object, newdata){
  
  fold <- 4
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), CA_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_AUS_GBM_5 <- function(object, newdata){
  
  fold <- 5
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), CA_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_AUS_GBM_6 <- function(object, newdata){
  
  fold <- 6
  
  cat_vars <- c("VehAge", "VehBody", "Gender", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("VehValue")) %>% 
    scale_withPar(c("VehValue"), CA_scaleinfo_AUS %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_AUS[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_AUS_GBM <- list(CA_pred_AUS_GBM_1, CA_pred_AUS_GBM_2, CA_pred_AUS_GBM_3, CA_pred_AUS_GBM_4, CA_pred_AUS_GBM_5, CA_pred_AUS_GBM_6)

## ----- Belgium -----

NC_pred_BE_GBM_1 <- function(object, newdata){
  
  fold <- 1
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), NC_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold]] %>% predict_model(newdata = newdata %>% 
                                                     left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>% data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_BE_GBM_2 <- function(object, newdata){
  
  fold <- 2
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), NC_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold]] %>% predict_model(newdata = newdata %>% 
                                                         left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  object %>% predict(train_mat, type = "response") %>% mean
}
NC_pred_BE_GBM_3 <- function(object, newdata){
  
  fold <- 3
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), NC_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold]] %>% predict_model(newdata = newdata %>% 
                                                         left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  object %>% predict(train_mat, type = "response") %>% mean
}
NC_pred_BE_GBM_4 <- function(object, newdata){
  
  fold <- 4
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), NC_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold]] %>% predict_model(newdata = newdata %>% 
                                                         left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  object %>% predict(train_mat, type = "response") %>% mean
}
NC_pred_BE_GBM_5 <- function(object, newdata){
  
  fold <- 5
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), NC_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold]] %>% predict_model(newdata = newdata %>% 
                                                         left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  object %>% predict(train_mat, type = "response") %>% mean
}
NC_pred_BE_GBM_6 <- function(object, newdata){
  
  fold <- 6
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), NC_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold]] %>% predict_model(newdata = newdata %>% 
                                                         left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  object %>% predict(train_mat, type = "response") %>% mean
}
NC_pred_BE_GBM <- list(NC_pred_BE_GBM_1, NC_pred_BE_GBM_2, NC_pred_BE_GBM_3, NC_pred_BE_GBM_4, NC_pred_BE_GBM_5, NC_pred_BE_GBM_6)

CA_pred_BE_GBM_1 <- function(object, newdata){
  
  fold <- 1
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), CA_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold+6]] %>% predict_model(newdata = newdata %>% left_join(latlong_per_postalcode,by=c("postcode")))
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_BE_GBM_2 <- function(object, newdata){
  
  fold <- 2
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), CA_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold+6]] %>% predict_model(newdata = newdata %>% 
                                                         left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  object %>% predict(train_mat, type = "response") %>% mean
}
CA_pred_BE_GBM_3 <- function(object, newdata){
  
  fold <- 3
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), CA_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold+6]] %>% predict_model(newdata = newdata %>% 
                                                         left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  object %>% predict(train_mat, type = "response") %>% mean
}
CA_pred_BE_GBM_4 <- function(object, newdata){
  
  fold <- 4
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), CA_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold+6]] %>% predict_model(newdata = newdata %>% 
                                                         left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  object %>% predict(train_mat, type = "response") %>% mean
}
CA_pred_BE_GBM_5 <- function(object, newdata){
  
  fold <- 5
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), CA_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold+6]] %>% predict_model(newdata = newdata %>% 
                                                         left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  object %>% predict(train_mat, type = "response") %>% mean
}
CA_pred_BE_GBM_6 <- function(object, newdata){
  
  fold <- 6
  
  cat_vars <- c("coverage", "fuel", "sex", "use", "fleet")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    left_join(latlong_per_postalcode,by=c("postcode")) %>% 
    dplyr::select(c("ageph", "bm", "agec", "power", "long", "lat")) %>% 
    scale_withPar(c("ageph", "bm", "agec", "power", "long", "lat"), CA_scaleinfo_BE %>% filter(Testfold == fold))
  
  new_prediction <- gbm_fits[[fold+6]] %>% predict_model(newdata = newdata %>% 
                                                         left_join(latlong_per_postalcode,by=c("postcode"))) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  object %>% predict(train_mat, type = "response") %>% mean
}
CA_pred_BE_GBM <- list(CA_pred_BE_GBM_1, CA_pred_BE_GBM_2, CA_pred_BE_GBM_3, CA_pred_BE_GBM_4, CA_pred_BE_GBM_5, CA_pred_BE_GBM_6)

## ----- French -----

NC_pred_FR_GBM_1 <- function(object, newdata){
  
  fold <- 1
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), NC_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_FR_GBM_2 <- function(object, newdata){
  
  fold <- 2
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), NC_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_FR_GBM_3 <- function(object, newdata){
  
  fold <- 3
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), NC_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_FR_GBM_4 <- function(object, newdata){
  
  fold <- 4
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), NC_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_FR_GBM_5 <- function(object, newdata){
  
  fold <- 5
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), NC_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_FR_GBM_6 <- function(object, newdata){
  
  fold <- 6
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), NC_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_freq_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_FR_GBM <- list(NC_pred_FR_GBM_1, NC_pred_FR_GBM_2, NC_pred_FR_GBM_3, NC_pred_FR_GBM_4, NC_pred_FR_GBM_5, NC_pred_FR_GBM_6)

CA_pred_FR_GBM_1 <- function(object, newdata){
  
  fold <- 1
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), CA_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_FR_GBM_2 <- function(object, newdata){
  
  fold <- 2
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), CA_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_FR_GBM_3 <- function(object, newdata){
  
  fold <- 3
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), CA_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_FR_GBM_4 <- function(object, newdata){
  
  fold <- 4
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), CA_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_FR_GBM_5 <- function(object, newdata){
  
  fold <- 5
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), CA_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_FR_GBM_6 <- function(object, newdata){
  
  fold <- 6
  
  cat_vars <- c("Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge")
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c("BonusMalus", "Density")) %>% 
    scale_withPar(c("BonusMalus", "Density"), CA_scaleinfo_FR %>% filter(Testfold == fold))
  
  new_prediction <- oos_sev_GBM_FR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_FR_GBM <- list(CA_pred_FR_GBM_1, CA_pred_FR_GBM_2, CA_pred_FR_GBM_3, CA_pred_FR_GBM_4, CA_pred_FR_GBM_5, CA_pred_FR_GBM_6)

## ----- Norwegian -----

NC_pred_NOR_GBM_1 <- function(object, newdata){
  
  fold <- 1
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c())
  
  new_prediction <- oos_freq_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_NOR_GBM_2 <- function(object, newdata){
  
  fold <- 2
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c()) 
  
  new_prediction <- oos_freq_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_NOR_GBM_3 <- function(object, newdata){
  
  fold <- 3
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c()) 
  
  new_prediction <- oos_freq_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_NOR_GBM_4 <- function(object, newdata){
  
  fold <- 4
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c())
  
  new_prediction <- oos_freq_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_NOR_GBM_5 <- function(object, newdata){
  
  fold <- 5
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c())
  
  new_prediction <- oos_freq_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_NOR_GBM_6 <- function(object, newdata){
  
  fold <- 6
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c()) 
  
  new_prediction <- oos_freq_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata) * newdata$expo
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
NC_pred_NOR_GBM <- list(NC_pred_NOR_GBM_1, NC_pred_NOR_GBM_2, NC_pred_NOR_GBM_3, NC_pred_NOR_GBM_4, NC_pred_NOR_GBM_5, NC_pred_NOR_GBM_6)

CA_pred_NOR_GBM_1 <- function(object, newdata){
  
  fold <- 1
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c()) 
  
  new_prediction <- oos_sev_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_NOR_GBM_2 <- function(object, newdata){
  
  fold <- 2
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c()) 
  
  new_prediction <- oos_sev_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_NOR_GBM_3 <- function(object, newdata){
  
  fold <- 3
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c())
  
  new_prediction <- oos_sev_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_NOR_GBM_4 <- function(object, newdata){
  
  fold <- 4
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c()) 
  
  new_prediction <- oos_sev_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_NOR_GBM_5 <- function(object, newdata){
  
  fold <- 5
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c()) 
  
  new_prediction <- oos_sev_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_NOR_GBM_6 <- function(object, newdata){
  
  fold <- 6
  
  cat_vars <- c('Male', 'Young', 'DistLimit', 'GeoRegion')
  
  train_cat_data <- lapply(cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  cont_data <- newdata %>% 
    dplyr::select(c())
  
  new_prediction <- oos_sev_GBM_NOR[[fold]][[3]] %>% predict_model(newdata = newdata)
  
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>%  data.matrix())
  
  return(object %>% predict(train_mat, type = "response") %>% mean)
}
CA_pred_NOR_GBM <- list(CA_pred_NOR_GBM_1, CA_pred_NOR_GBM_2, CA_pred_NOR_GBM_3, CA_pred_NOR_GBM_4, CA_pred_NOR_GBM_5, CA_pred_NOR_GBM_6)


# ----
# ----- INTERPRETATION CALCULATION -----

# Here we get the wanted interpretation results and surrogates

## ----- Interpretation functions -----

# Function to calculate variable importance
VI_calculation <- function(data, variables, model, pred_fun){
  
  # Predictions on the data set
  reg_prediction <- pred_fun(model, data)
  
  # For each variable, permutate and predict
  lapply(variables, function(var){
    
    # Permutate the variable
    permutated_data <- data %>% mutate(!!var := (slice_sample(., n=nrow(.)) %>% pull(var)))
    
    mut_prediction <- pred_fun(model, permutated_data)
    
    # Calculate VI 
    tibble(Variable = var, VI =  sum(abs(reg_prediction - mut_prediction)))
    
  }) %>% do.call(rbind,.) %>% mutate(scaled_VI = VI / sum(VI))
}

## ----- Make data samples to calculate interpretations -----

# Australian
NC_data_slice_AUS <- data_AUS %>% slice_sample(n=10000)
CA_data_slice_AUS <- data_AUS %>% filter(nclaims>0, !is.na(average)) %>% slice_sample(n=4000)
save(NC_data_slice_AUS, CA_data_slice_AUS, file = "data_slices_AUS")

# Belgian
NC_data_slice_BE <- data_BE_PC %>% slice_sample(n=10000)
CA_data_slice_BE <- data_BE_PC %>% filter(nclaims>0, !is.na(average)) %>% slice_sample(n=10000)
save(NC_data_slice_BE, CA_data_slice_BE, file = "data_slices_BE")

# French
NC_data_slice_FR <- data_FR %>% slice_sample(n=10000)
CA_data_slice_FR <- data_FR %>% filter(nclaims>0, !is.na(average)) %>% slice_sample(n=10000)
save(NC_data_slice_FR, CA_data_slice_FR, file = "data_slices_FR")

# Norwegian
NC_data_slice_NOR <- data_NOR %>% slice_sample(n=10000)
CA_data_slice_NOR <- data_NOR %>% filter(nclaims>0, !is.na(average)) %>% slice_sample(n=4000)
save(NC_data_slice_NOR, CA_data_slice_NOR, file = "data_slices_NOR")

## ----- Read in data samples -----

load('data_slices_AUS')
load('data_slices_BE')
load('data_slices_FR')
load('data_slices_NOR')

# ----
## ----- Australia -----

### ----- Surrogate -----

# Take a sample of train data
NC_data_train_slice_AUS <- data_AUS %>% filter(fold_nr != 1) %>% slice_sample(n=10000)

# Tune the surrogate technique and determine best split
tuned_surrogate <- maidrr::autotune(NC_opt_AUS[[1]]$model,
                                    data = NC_data_train_slice_AUS,
                                    vars = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'),
                                    target = 'nclaims',
                                    hcut = 0.75,
                                    pred_fun = NC_pred_AUS_GBM[[1]],
                                    lambdas = as.vector(outer(seq(1, 10, 2), 10^(-6:-2))),
                                    max_ngrps = 15,
                                    nfolds = 5,
                                    strat_vars = c('nclaims', 'expo'),
                                    glm_par = alist(family = poisson(link = 'log'),
                                                    offset = log(expo)),
                                    err_fun = maidrr::poi_dev,
                                    out_pds = TRUE,
                                    ncores = 1)
# Poisson deviance of the GLM on segmented data
seg_data <- tuned_surrogate$best_surr$data %>% select(c(ends_with("_"),'expo', 'nclaims'))
dev <- dev_poiss_2(seg_data %>% pull(nclaims), 
                   predict(tuned_surrogate$best_surr, seg_data, type = 'response')
)

# Combina the surrogate results and in-sample deviance
NC_Surrogate_CANN_GBM_flex_AUS <- append(tuned_surrogate,list(deviance = dev))

save(NC_Surrogate_CANN_GBM_flex_AUS, file = 'NC_Surrogate_CANN_GBM_flex_AUS')

# Take a sample of train data
NC_data_test_slice_AUS <- data_AUS %>% filter(fold_nr == 1) %>% slice_sample(n=10000)

# Construct data segmentation on test data
data_test_segm <- maidrr::segmentation(fx_vars = NC_Surrogate_CANN_GBM_flex_AUS$pd_fx[names(NC_Surrogate_CANN_GBM_flex_AUS$slct_feat)], 
                                       data = data_AUS %>% filter(fold_nr == 1) %>% slice_sample(n=10000) , 
                                       type = 'ngroups', 
                                       values = NC_Surrogate_CANN_GBM_flex_AUS$slct_feat)

# Calculate out-of-sample deviance of the surrogate model
NC_Surrogate_CANN_GBM_flex_AUS_OOS <- dev_poiss_2(data_test_segm %>% pull(nclaims), 
                   predict(NC_Surrogate_CANN_GBM_flex_AUS$best_surr, 
                           data_test_segm %>% select(c(ends_with("_"),'expo', 'nclaims')), 
                           type = 'response')
)

save(NC_data_train_slice_AUS, NC_data_test_slice_AUS, NC_Surrogate_CANN_GBM_flex_AUS_OOS, file = 'NC_Surrogate_CANN_GBM_flex_AUS_withOOS')

### ----- Partial Dependence -----

NC_PDP_CANN_GBM_flex_DrivAge_AUS <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = NC_opt_AUS[[fold]]$model,
    var = 'DrivAge',
    grid = data.frame(DrivAge = data_AUS %>% pull(DrivAge) %>% unique),
    data = NC_data_slice_AUS, 
    fun = NC_pred_AUS_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_PDP_CANN_GBM_flex_DrivAge_AUS, file = 'NC_PDP_CANN_GBM_flex_DrivAge_AUS')

NC_PDP_CANN_GBM_flex_VehValue_AUS <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = NC_opt_AUS[[fold]]$model,
    var = 'VehValue',
    grid = data.frame(VehValue = data_AUS %>% pull(VehValue) %>% unique),
    data = NC_data_slice_AUS, 
    fun = NC_pred_AUS_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_PDP_CANN_GBM_flex_VehValue_AUS, file = 'NC_PDP_CANN_GBM_flex_VehValue_AUS')

NC_PDP_CANN_GBM_flex_VehBody_AUS <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = NC_opt_AUS[[fold]]$model,
    var = 'VehBody',
    grid = data.frame(VehBody = data_AUS %>% pull(VehBody) %>% unique),
    data = NC_data_slice_AUS, 
    fun = NC_pred_AUS_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_PDP_CANN_GBM_flex_VehBody_AUS, file = 'NC_PDP_CANN_GBM_flex_VehBody_AUS')

CA_PDP_CANN_GBM_flex_VehValue_AUS <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = CA_opt_AUS[[fold]]$model,
    var = 'VehValue',
    grid = data.frame(VehValue = data_AUS %>% pull(VehValue) %>% unique),
    data = CA_data_slice_AUS, 
    fun = CA_pred_AUS_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_PDP_CANN_GBM_flex_VehValue_AUS, file = 'CA_PDP_CANN_GBM_flex_VehValue_AUS')

### ----- Variable Importance -----

NC_VI_CANN_GBM_flex_AUS <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_AUS,
    variables = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'), 
    model = NC_opt_AUS[[fold]]$model,
    pred_fun = NC_pred_AUS_GBM[[fold]]
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_CANN_GBM_flex_AUS, file = 'NC_VI_CANN_GBM_flex_AUS')

CA_VI_CANN_GBM_flex_AUS <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_AUS,
    variables = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'),  
    model = CA_opt_AUS[[fold]]$model,
    pred_fun = CA_pred_AUS_GBM[[fold]]
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_CANN_GBM_flex_AUS, file = 'CA_VI_CANN_GBM_flex_AUS')

## ----- Belgium -----

### ----- Surrogate -----

# Take a sample of train data
NC_data_train_slice_BE <- data_BE_PC %>% filter(fold_nr != 1) %>% slice_sample(n=10000)

# Tune the surrogate technique and determine best split
tuned_surrogate <- maidrr::autotune(NC_opt_BE[[1]]$model,
                                    data = NC_data_train_slice_BE,
                                    vars = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet', 'postcode'), 
                                    target = 'nclaims',
                                    hcut = 0.75,
                                    pred_fun = NC_pred_BE_GBM[[1]],
                                    lambdas = as.vector(outer(seq(1, 10, 2), 10^(-6:-2))),
                                    max_ngrps = 15,
                                    nfolds = 5,
                                    strat_vars = c('nclaims', 'expo'),
                                    glm_par = alist(family = poisson(link = 'log'),
                                                    offset = log(expo)),
                                    err_fun = maidrr::poi_dev,
                                    out_pds = TRUE,
                                    ncores = 1)
# Poisson deviance of the GLM on segmented data
seg_data <- tuned_surrogate$best_surr$data %>% select(c(ends_with("_"),'expo', 'nclaims'))
dev <- dev_poiss_2(seg_data %>% pull(nclaims), 
                   predict(tuned_surrogate$best_surr, seg_data, type = 'response')
)

# Combina the surrogate results and in-sample deviance
NC_Surrogate_CANN_GBM_flex_BE <- append(tuned_surrogate,list(deviance = dev))

save(NC_Surrogate_CANN_GBM_flex_BE, file = 'NC_Surrogate_CANN_GBM_flex_BE')

# Take a sample of train data
NC_data_test_slice_BE <- data_BE_PC %>% filter(fold_nr == 1) %>% slice_sample(n=10000)

# Construct data segmentation on test data
data_test_segm <- maidrr::segmentation(fx_vars = NC_Surrogate_CANN_GBM_flex_BE$pd_fx[names(NC_Surrogate_CANN_GBM_flex_BE$slct_feat)], 
                                       data = NC_data_test_slice_BE , 
                                       type = 'ngroups', 
                                       values = NC_Surrogate_CANN_GBM_flex_BE$slct_feat)

# Calculate out-of-sample deviance of the surrogate model
NC_Surrogate_CANN_GBM_flex_BE_OOS <- dev_poiss_2(data_test_segm %>% pull(nclaims), 
                                                  predict(NC_Surrogate_CANN_GBM_flex_BE$best_surr, 
                                                          data_test_segm %>% select(c(ends_with("_"),'expo', 'nclaims')), 
                                                          type = 'response')
)

save(NC_data_train_slice_BE, NC_data_test_slice_BE, NC_Surrogate_CANN_GBM_flex_BE_OOS, file = 'NC_Surrogate_CANN_GBM_flex_BE_withOOS')

### ----- Partial dependency -----

# Frequency 
NC_PDP_CANN_GBM_flex_AGEPH_BE <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = NC_opt_BE[[fold]]$model,
    var = 'ageph',
    grid = data.frame(ageph = c(18:95)),
    data = NC_data_slice_BE, 
    fun = NC_pred_BE_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_PDP_CANN_GBM_flex_AGEPH_BE, file = 'NC_PDP_CANN_GBM_flex_AGEPH_BE')

NC_PDP_CANN_GBM_flex_BM_BE <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = NC_opt_BE[[fold]]$model,
    var = 'bm',
    grid = data.frame(bm = c(0:22)),
    data = NC_data_slice_BE, 
    fun = NC_pred_BE_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_PDP_CANN_GBM_flex_BM_BE, file = 'NC_PDP_CANN_GBM_flex_BM_BE')

# We also want PDP effects for postalcodes not present in the data
belgium_shape_sf <- st_read('./shape file Belgie postcodes/npc96_region_Project1.shp', quiet = TRUE)
all_postalcodes <- belgium_shape_sf %>% pull(POSTCODE) %>% unique

NC_PDP_CANN_GBM_flex_POSTALCODE_BE <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = NC_opt_BE[[fold]]$model,
    var = 'postcode',
    grid = data.frame(postcode = all_postalcodes),
    data = NC_data_slice_BE, 
    fun = NC_pred_BE_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_PDP_CANN_GBM_flex_POSTALCODE_BE, file = 'NC_PDP_CANN_GBM_flex_POSTALCODE_BE')

# The MAIDRR function get_pd does not work for values not present in the input data. 
# We want partial dependency effect for all possible postal codes so we do it manually

NC_PDP_CANN_GBM_flex_POSTALCODE_BE_all <- sapply(all_postalcodes, function(pcode){
  NC_pred_BE_GBM[[1]](object = NC_opt_BE[[1]]$model, newdata = NC_data_slice_BE %>% mutate(postcode = pcode))
})

NC_PDP_CANN_GBM_flex_POSTALCODE_BE_all_cb <- bind_cols(x = all_postalcodes, y = NC_PDP_CANN_GBM_flex_POSTALCODE_BE_all)
save(NC_PDP_CANN_GBM_flex_POSTALCODE_BE_all_cb, file = 'NC_PDP_CANN_GBM_flex_POSTALCODE_BE_all_cb')

# Severity 
CA_PDP_CANN_GBM_flex_AGEPH_BE <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = CA_opt_BE[[fold]]$model,
    var = 'ageph',
    grid = data.frame(ageph = c(18:95)),
    data = CA_data_slice_BE, 
    fun = CA_pred_BE_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_PDP_CANN_GBM_flex_AGEPH_BE, file = 'CA_PDP_CANN_GBM_flex_AGEPH_BE')

CA_PDP_CANN_GBM_flex_BM_BE <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = CA_opt_BE[[fold]]$model,
    var = 'bm',
    grid = data.frame(bm = c(0:22)),
    data = CA_data_slice_BE, 
    fun = CA_pred_BE_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_PDP_CANN_GBM_flex_BM_BE, file = 'CA_PDP_CANN_GBM_flex_BM_BE')

CA_PDP_CANN_GBM_flex_POSTALCODE_BE <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = CA_opt_BE[[fold]]$model,
    var = 'postcode',
    grid = data.frame(postcode = all_postalcodes),
    data = CA_data_slice_BE, 
    fun = CA_pred_BE_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_PDP_CANN_GBM_flex_POSTALCODE_BE, file = 'CA_PDP_CANN_GBM_flex_POSTALCODE_BE')

CA_PDP_CANN_GBM_flex_POSTALCODE_BE_all <- sapply(all_postalcodes, function(pcode){
  CA_pred_BE_GBM[[1]](object = CA_opt_BE[[1]]$model, newdata = CA_data_slice_BE %>% mutate(postcode = pcode))
})
CA_PDP_CANN_GBM_flex_POSTALCODE_BE_all_cb <- bind_cols(x = all_postalcodes, y = CA_PDP_CANN_GBM_flex_POSTALCODE_BE_all)
save(CA_PDP_CANN_GBM_flex_POSTALCODE_BE_all_cb, file = 'CA_PDP_CANN_GBM_flex_POSTALCODE_BE_all_cb')

### ----- Variable importance -----

NC_VI_CANN_GBM_flex_BE <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_BE,
    variables = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet', 'postcode'), 
    model = NC_opt_BE[[fold]]$model,
    pred_fun = NC_pred_BE_GBM[[fold]]
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_CANN_GBM_flex_BE, file = 'NC_VI_CANN_GBM_flex_BE')

CA_VI_CANN_GBM_flex_BE <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_BE,
    variables = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet', 'postcode'), 
    model = CA_opt_BE[[fold]]$model,
    pred_fun = CA_pred_BE_GBM[[fold]]
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_CANN_GBM_flex_BE, file = 'CA_VI_CANN_GBM_flex_BE')

## ----- French -----

### ----- Surrogate -----

# Take a sample of train data
NC_data_train_slice_FR <- data_FR %>% filter(fold_nr != 1) %>% slice_sample(n=10000)

# Tune the surrogate technique and determine best split
tuned_surrogate <- maidrr::autotune(NC_opt_FR[[1]]$model,
                                    data = NC_data_train_slice_FR,
                                    vars = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"), 
                                    target = 'nclaims',
                                    hcut = 0.75,
                                    pred_fun = NC_pred_FR_GBM[[1]],
                                    lambdas = as.vector(outer(seq(1, 10, 2), 10^(-6:-2))),
                                    max_ngrps = 15,
                                    nfolds = 5,
                                    strat_vars = c('nclaims', 'expo'),
                                    glm_par = alist(family = poisson(link = 'log'),
                                                    offset = log(expo)),
                                    err_fun = maidrr::poi_dev,
                                    out_pds = TRUE,
                                    ncores = 1)
# Poisson deviance of the GLM on segmented data
seg_data <- tuned_surrogate$best_surr$data %>% select(c(ends_with("_"),'expo', 'nclaims'))
dev <- dev_poiss_2(seg_data %>% pull(nclaims), 
                   predict(tuned_surrogate$best_surr, seg_data, type = 'response')
)

# Combina the surrogate results and in-sample deviance
NC_Surrogate_CANN_GBM_flex_FR <- append(tuned_surrogate,list(deviance = dev))

save(NC_Surrogate_CANN_GBM_flex_FR, file = 'NC_Surrogate_CANN_GBM_flex_FR')

# Take a sample of train data
NC_data_test_slice_FR <- data_FR %>% filter(fold_nr == 1) %>% slice_sample(n=10000)

# Construct data segmentation on test data
data_test_segm <- maidrr::segmentation(fx_vars = NC_Surrogate_CANN_GBM_flex_FR$pd_fx[names(NC_Surrogate_CANN_GBM_flex_FR$slct_feat)], 
                                       data = NC_data_test_slice_FR , 
                                       type = 'ngroups', 
                                       values = NC_Surrogate_CANN_GBM_flex_FR$slct_feat)

# Calculate out-of-sample deviance of the surrogate model
NC_Surrogate_CANN_GBM_flex_FR_OOS <- dev_poiss_2(data_test_segm %>% pull(nclaims), 
                                                 predict(NC_Surrogate_CANN_GBM_flex_FR$best_surr, 
                                                         data_test_segm %>% select(c(ends_with("_"),'expo', 'nclaims')), 
                                                         type = 'response')
)

save(NC_data_train_slice_FR, NC_data_test_slice_FR, NC_Surrogate_CANN_GBM_flex_FR_OOS, file = 'NC_Surrogate_CANN_GBM_flex_FR_withOOS')

### ----- Partial Dependence -----

NC_PDP_CANN_GBM_flex_DrivAge_FR <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = NC_opt_FR[[fold]]$model,
    var = 'DrivAge',
    grid = data.frame(DrivAge = c(1:7)),
    data = NC_data_slice_FR, 
    fun = NC_pred_FR_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_PDP_CANN_GBM_flex_DrivAge_FR, file = 'NC_PDP_CANN_GBM_flex_DrivAge_FR')

NC_PDP_CANN_GBM_flex_BonusMalus_FR <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = NC_opt_FR[[fold]]$model,
    var = 'BonusMalus',
    grid = data.frame(BonusMalus = c(50:230)),
    data = NC_data_slice_FR, 
    fun = NC_pred_FR_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_PDP_CANN_GBM_flex_BonusMalus_FR, file = 'NC_PDP_CANN_GBM_flex_BonusMalus_FR')

### ----- Variable Importance -----

NC_VI_CANN_GBM_flex_FR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_FR,
    variables = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"), 
    model = NC_opt_FR[[fold]]$model,
    pred_fun = NC_pred_FR_GBM[[fold]]
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_CANN_GBM_flex_FR, file = 'NC_VI_CANN_GBM_flex_FR')

CA_VI_CANN_GBM_flex_FR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_FR,
    variables = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"), 
    model = CA_opt_FR[[fold]]$model,
    pred_fun = CA_pred_FR_GBM[[fold]]
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_CANN_GBM_flex_FR, file = 'CA_VI_CANN_GBM_flex_FR')

## ----- Norwegian -----

### ----- Surrogate -----

NC_Surrogate_CANN_GBM_flex_NOR <- lapply(1:6, function(fold){
  
  # Tune the surrogate technique and determine best split
  tuned_surrogate <- maidrr::autotune(NC_opt_NOR[[fold]]$model,
                                      data = NC_data_slice_NOR,
                                      vars = c('Male', 'Young', 'DistLimit', 'GeoRegion'),
                                      target = 'nclaims',
                                      hcut = 0.75,
                                      pred_fun = NC_pred_NOR_GBM[[fold]],
                                      lambdas = as.vector(outer(seq(1, 10, 2), 10^(-6:-2))),
                                      max_ngrps = 15,
                                      nfolds = 5,
                                      strat_vars = c('nclaims', 'expo'),
                                      glm_par = alist(family = poisson(link = 'log'),
                                                      offset = log(expo)),
                                      err_fun = maidrr::poi_dev,
                                      out_pds = TRUE,
                                      ncores = 1)
  # Poisson deviance of the GLM on segmented data
  seg_data <- tuned_surrogate$best_surr$data %>% select(c(ends_with("_"),'expo', 'nclaims'))
  dev <- dev_poiss_2(seg_data %>% pull(nclaims), 
                     predict(tuned_surrogate$best_surr, seg_data, type = 'response')
  )
  
  return(append(tuned_surrogate,list(deviance = dev)))
  
})
save(NC_Surrogate_CANN_GBM_flex_NOR, file = 'NC_Surrogate_CANN_GBM_flex_NOR')

CA_Surrogate_CANN_GBM_flex_NOR <- lapply(1:1, function(fold){
  
  # Tune the surrogate technique and determine best split
  tuned_surrogate <- maidrr::autotune(CA_opt_NOR[[fold]]$model,
                                      data = CA_data_slice_NOR,
                                      vars = c('Male', 'Young', 'DistLimit', 'GeoRegion'),
                                      target = 'average',
                                      hcut = 0.75,
                                      pred_fun = CA_pred_NOR_GBM[[fold]],
                                      lambdas = as.vector(outer(seq(1, 10, 2), 10^(-6:-2))),
                                      max_ngrps = 15,
                                      nfolds = 5,
                                      strat_vars = c('average', 'nclaims'),
                                      glm_par = alist(family = Gamma(link = "log"),
                                                      weights = nclaims),
                                      err_fun = maidrr::poi_dev,
                                      out_pds = TRUE,
                                      ncores = 1)
  # Poisson deviance of the GLM on segmented data
  seg_data <- tuned_surrogate$best_surr$data %>% select(c(ends_with("_"),'average', 'nclaims'))
  dev <- dev_gamma(seg_data %>% pull(average), 
                   predict(tuned_surrogate$best_surr, seg_data, type = 'response'),
                   weight = seg_data %>% pull(nclaims)
  )
  
  return(append(tuned_surrogate,list(deviance = dev)))
  
})
save(CA_Surrogate_CANN_GBM_flex_NOR, file = 'CA_Surrogate_CANN_GBM_flex_NOR')

### ----- Partial Dependence -----

NC_PDP_CANN_GBM_flex_Young_NOR <- lapply(1:6, function(fold){
  tibble(maidrr::get_pd(
    mfit = NC_opt_NOR[[fold]]$model,
    var = 'Young',
    grid = data.frame(Young = data_NOR %>% pull(Young) %>% unique),
    data = NC_data_slice_NOR, 
    fun = NC_pred_NOR_GBM[[fold]],
    ncores = 1
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_PDP_CANN_GBM_flex_Young_NOR, file = 'NC_PDP_CANN_GBM_flex_Young_NOR')

### ----- Variable Importance -----

NC_VI_CANN_GBM_flex_NOR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_NOR,
    variables =  c('Male', 'Young', 'DistLimit', 'GeoRegion'),
    model = NC_opt_NOR[[fold]]$model,
    pred_fun = NC_pred_NOR_GBM[[fold]]
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_CANN_GBM_flex_NOR, file = 'NC_VI_CANN_GBM_flex_NOR')

CA_VI_CANN_GBM_flex_NOR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_NOR,
    variables = c('Male', 'Young', 'DistLimit', 'GeoRegion'), 
    model = CA_opt_NOR[[fold]]$model,
    pred_fun = CA_pred_NOR_GBM[[fold]]
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_CANN_GBM_flex_NOR, file = 'CA_VI_CANN_GBM_flex_NOR')

# -----
# ----- PLOT MAKING -----

# Here we read in the results from the interpretation calculation
# Plots are created for each wanted item in the paper

## ----- VIP plot -----

load('NC_VI_CANN_GBM_flex_AUS')
load('NC_VI_CANN_GBM_flex_BE')
load('NC_VI_CANN_GBM_flex_FR')
load('NC_VI_CANN_GBM_flex_NOR')

load('CA_VI_CANN_GBM_flex_AUS')
load('CA_VI_CANN_GBM_flex_BE')
load('CA_VI_CANN_GBM_flex_FR')
load('CA_VI_CANN_GBM_flex_NOR')


NC_VIP_AUS <- NC_VI_CANN_GBM_flex_AUS %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(y = reorder(Variable, scaled_VI, mean))) +  
  geom_col(aes(x = scaled_VI, color = Testfold, fill = Testfold), position="dodge", alpha = 0.6) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot")
NC_VIP_BE <- NC_VI_CANN_GBM_flex_BE %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(y = reorder(Variable, scaled_VI, mean))) +  
  geom_col(aes(x = scaled_VI, color = Testfold, fill = Testfold), position="dodge", alpha = 0.6) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot")
NC_VIP_FR <- NC_VI_CANN_GBM_flex_FR %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(y = reorder(Variable, scaled_VI, mean))) +  
  geom_col(aes(x = scaled_VI, color = Testfold, fill = Testfold), position="dodge", alpha = 0.6) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot")
NC_VIP_NOR <- NC_VI_CANN_GBM_flex_NOR %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(y = reorder(Variable, scaled_VI, mean))) +  
  geom_col(aes(x = scaled_VI, color = Testfold, fill = Testfold), position="dodge", alpha = 0.6) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot")

CA_VIP_AUS <- CA_VI_CANN_GBM_flex_AUS %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(y = reorder(Variable, scaled_VI, mean))) +  
  geom_col(aes(x = scaled_VI, color = Testfold, fill = Testfold), position="dodge", alpha = 0.6) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot")
CA_VIP_BE <- CA_VI_CANN_GBM_flex_BE %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(y = reorder(Variable, scaled_VI, mean))) +  
  geom_col(aes(x = scaled_VI, color = Testfold, fill = Testfold), position="dodge", alpha = 0.6) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot")
CA_VIP_FR <- CA_VI_CANN_GBM_flex_FR %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(y = reorder(Variable, scaled_VI, mean))) +  
  geom_col(aes(x = scaled_VI, color = Testfold, fill = Testfold), position="dodge", alpha = 0.6) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot")
CA_VIP_NOR <- CA_VI_CANN_GBM_flex_NOR %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(y = reorder(Variable, scaled_VI, mean))) +  
  geom_col(aes(x = scaled_VI, color = Testfold, fill = Testfold), position="dodge", alpha = 0.6) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot")

# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(NC_VIP_AUS + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Australia") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.28)), 
                            NC_VIP_BE + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Belgium") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.28)),
                            NC_VIP_FR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "France") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.28)), 
                            NC_VIP_NOR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Norway") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.28)),  
                            CA_VIP_AUS + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL) + 
                              theme(plot.margin = unit(margin_set, "cm")), 
                            CA_VIP_BE + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL) + 
                              theme(plot.margin = unit(margin_set, "cm")),
                            CA_VIP_FR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL) + 
                              theme(plot.margin = unit(margin_set, "cm")),
                            CA_VIP_NOR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL) +  
                              theme(plot.margin = unit(margin_set, "cm")), 
                            align = "hv")

# Make a grid of all plots, with country names
allplotsgrid <- plot_grid(#ggdraw() + draw_label(''),ggdraw() + draw_label("Out-of-sample Poisson Deviance", size = 12),ggdraw() + draw_label("Out-of-sample gamma Deviance", size = 12),
  ggdraw() + draw_label("Frequency", angle = 90, size = 12),allplotslist[[1]], allplotslist[[2]], allplotslist[[3]], allplotslist[[4]],
  ggdraw() + draw_label("Severity", angle = 90, size = 12), allplotslist[[5]], allplotslist[[6]], allplotslist[[7]], allplotslist[[8]],
  ncol = 5, rel_widths = c(0.02,0.23, 0.23, 0.23, 0.23)#, rel_heights = c(0.05,0.24,0.24,0.24,0.24)
)

# Final plot grid, with legend
final_VIP_plot <- plot_grid(
  allplotsgrid,
  get_legend(NC_VIP_AUS),
  ncol = 1, rel_heights = c(0.9,0.1)
)

final_VIP_plot

# Save plot as PDF
ggsave("final_VIP_plot.pdf",
       final_VIP_plot, 
       device = cairo_pdf,
       width = 32,
       height = 15,
       scale = 1.2,
       units = "cm")

## ----- PDP AgePh -----

load('NC_PDP_CANN_GBM_flex_AGEPH_BE')
load('NC_PDP_CANN_GBM_flex_DrivAge_AUS')
load('NC_PDP_CANN_GBM_flex_DrivAge_FR')
load('NC_PDP_CANN_GBM_flex_Young_NOR')

NC_PDP_age_AUS <- NC_PDP_CANN_GBM_flex_DrivAge_AUS %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(x = x)) + geom_col(aes(y = y, group = Testfold, color = Testfold, fill = Testfold), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.5)) + 
  theme_bw() + 
  guides(group = guide_legend(nrow = 2, byrow = TRUE)) + 
  theme(legend.position="bottom", 
        legend.direction="horizontal") + 
  xlab("Policyholder age") + 
  ylab("Partial dependency effect")

NC_PDP_age_BE <- NC_PDP_CANN_GBM_flex_AGEPH_BE %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(x = x)) + geom_line(aes(y = y, group = Testfold, color = Testfold), size = 0.8) + 
  theme_bw() + 
  guides(group = guide_legend(nrow = 2, byrow = TRUE)) + 
  theme(legend.position="bottom", 
        legend.direction="horizontal") + 
  xlab("Policyholder age") + 
  ylab("Partial dependency effect")

NC_PDP_age_FR <- NC_PDP_CANN_GBM_flex_DrivAge_FR %>% 
  mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(x = x)) + geom_col(aes(y = y, group = Testfold, color = Testfold, fill = Testfold), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.5)) + 
  theme_bw() + 
  guides(group = guide_legend(nrow = 2, byrow = TRUE)) + 
  theme(legend.position="bottom", 
        legend.direction="horizontal") + 
  xlab("Age bucket") + 
  ylab("Partial dependency effect") + 
  scale_x_discrete(labels = c('< 21','[21,26[','[26,30[','[30,40[','[40,50[','[50,70[','\u2265 70'))

NC_PDP_age_NOR <- NC_PDP_CANN_GBM_flex_Young_NOR %>% mutate(Testfold = factor(Testfold)) %>% mutate(x = factor(x, levels = c("Yes", "No"))) %>% 
  ggplot(aes(x = x)) + geom_col(aes(y = y, group = Testfold, color = Testfold, fill = Testfold), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.5)) + 
  theme_bw() + 
  guides(group = guide_legend(nrow = 2, byrow = TRUE)) + 
  theme(legend.position="bottom", 
        legend.direction="horizontal") + 
  xlab("Young driver") + 
  ylab("Partial dependency effect")

# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(NC_PDP_age_AUS + theme(legend.position = "none") + 
                              labs(x = NULL, subtitle = "Australia") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0),
                                    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)), 
                            NC_PDP_age_BE + theme(legend.position = "none") + 
                              labs(y = NULL, subtitle = "Belgium") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0),
                                    axis.title.x = element_text(margin = margin(t = -60))),
                            NC_PDP_age_FR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "France") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0),
                                    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)), 
                            NC_PDP_age_NOR + theme(legend.position = "none") + 
                              labs(y = NULL, subtitle = "Norway") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0),
                                    axis.title.x = element_text(margin = margin(t = -60))),
                            align = "hv")

# Make a grid of all plots, with country names
allplotsgrid <- plot_grid(#ggdraw() + draw_label(''),ggdraw() + draw_label("Out-of-sample Poisson Deviance", size = 12),ggdraw() + draw_label("Out-of-sample gamma Deviance", size = 12),
  allplotslist[[1]], allplotslist[[2]], allplotslist[[3]], allplotslist[[4]],
  ncol = 4, rel_widths = c(0.25,0.25,0.25,0.25)#, rel_heights = c(0.05,0.24,0.24,0.24,0.24)
)

# Final plot grid, with legend
final_PDP_ageph_plot <- plot_grid(
  allplotsgrid,
  get_legend(NC_VIP_AUS),
  ncol = 1, rel_heights = c(0.9,0.1)
)

final_PDP_ageph_plot

# Save plot as PDF
ggsave("final_PDP_ageph_plot.pdf",
       final_PDP_ageph_plot, 
       device = cairo_pdf,
       width = 32,
       height = 10,
       scale = 1.2,
       units = "cm")


## ----- PDP Spatial -----

# Read in shape file of Belgium and transfor

sf_use_s2(FALSE) # Due to update to sf package, we need to switch of geometry check
belgium_shape_sf <- st_read('./shape file Belgie postcodes/npc96_region_Project1.shp', quiet = TRUE)
belgium_shape_sf <- st_transform(belgium_shape_sf, CRS("+proj=longlat +datum=WGS84"))

# First PDP calculation try

load('NC_PDP_CANN_GBM_flex_POSTALCODE_BE')
load('CA_PDP_CANN_GBM_flex_POSTALCODE_BE')

# Join pdp info with the Belgian Shape file
NC_PDP_spatial_BE <- left_join(belgium_shape_sf, NC_PDP_CANN_GBM_flex_POSTALCODE_BE %>% filter(Testfold == 1) %>% select(POSTCODE = x, NC = y), by = "POSTCODE") %>% 
  left_join(., CA_PDP_CANN_GBM_flex_POSTALCODE_BE %>% filter(Testfold == 1) %>% select(POSTCODE = x, CA = y), by = "POSTCODE")

# Plot the PDP effect over map of Belgium
NC_PDP_spatial_BE_plot <- tm_shape(NC_PDP_spatial_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "NC", title = "Average number of claims", 
          textNA = "No Policyholders", 
          style = "cont", 
          palette = "Blues", colorNA = "white") +
  tm_layout(frame = FALSE, main.title = "Frequency")

# Plot the PDP effect over map of Belgium
CA_PDP_spatial_BE_plot <- tm_shape(NC_PDP_spatial_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "CA", title = "Average claim amount", 
          textNA = "No Policyholders", 
          style = "cont", 
          palette = "Blues", colorNA = "white") +
  tm_layout(frame = FALSE, main.title = "Severity")

PDP_spatial_plotTogether <- tmap_arrange(NC_PDP_spatial_BE_plot, CA_PDP_spatial_BE_plot, nrow=1)

pdf(file = "PDP_spatial_plotTogether.pdf", height = 7, width = 14)
print(PDP_spatial_plotTogether)
dev.off()

# Redo - With all possible postalcodes

load('NC_PDP_CANN_GBM_flex_POSTALCODE_BE_all_cb')
load('CA_PDP_CANN_GBM_flex_POSTALCODE_BE_all_cb')

# Join pdp info with the Belgian Shape file
NC_PDP_spatial_BE <- left_join(belgium_shape_sf, NC_PDP_CANN_GBM_flex_POSTALCODE_BE_all_cb %>% select(POSTCODE = x, NC = y), by = "POSTCODE") %>% 
  left_join(., CA_PDP_CANN_GBM_flex_POSTALCODE_BE_all_cb %>% select(POSTCODE = x, CA = y), by = "POSTCODE")

# Plot the PDP effect over map of Belgium
NC_PDP_spatial_BE_plot <- tm_shape(NC_PDP_spatial_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "NC", title = "Average number of claims", 
          textNA = "No Policyholders", 
          style = "cont", 
          palette = "Blues", colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 1,
            legend.title.size=1.5,)

# Plot the PDP effect over map of Belgium
CA_PDP_spatial_BE_plot <- tm_shape(NC_PDP_spatial_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "CA", title = "Average claim amount", 
          textNA = "No Policyholders", 
          style = "cont", 
          palette = "Blues", colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 1,
            legend.title.size=1.5,)

PDP_spatial_plotTogether <- tmap_arrange(NC_PDP_spatial_BE_plot, CA_PDP_spatial_BE_plot, nrow=1)

PDP_spatial_plotTogether

pdf(file = "PDP_spatial_plotTogether.pdf", height = 6, width = 14)
print(PDP_spatial_plotTogether)
dev.off()

## ----- PDP Commercial -----

load('NC_PDP_CANN_GBM_flex_BM_BE')
load('CA_PDP_CANN_GBM_flex_VehValue_AUS')

NC_PDP_BM_BE <- NC_PDP_CANN_GBM_flex_BM_BE %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(x = x)) + geom_line(aes(y = y, group = Testfold, color = Testfold), size = 0.8) + 
  theme_bw() + 
  guides(group = guide_legend(nrow = 2, byrow = TRUE)) + 
  theme(legend.position="bottom", 
        legend.direction="horizontal") + 
  xlab("Bonus-malus score") + 
  ylab("Partial dependency effect")

CA_PDP_VehValue_AUS <- CA_PDP_CANN_GBM_flex_VehValue_AUS %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(x = x)) + geom_line(aes(y = y, group = Testfold, color = Testfold), size = 0.8) + 
  theme_bw() + 
  guides(group = guide_legend(nrow = 2, byrow = TRUE)) + 
  theme(legend.position="bottom", 
        legend.direction="horizontal") + 
  xlab("Vehicle value (x1000 AUD)") + 
  ylab("Partial dependency effect")

# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(NC_PDP_BM_BE + theme(legend.position = "none") + 
                              labs(subtitle = "Belgian bonus-malus effect (frequency)") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0)),
                            CA_PDP_VehValue_AUS + theme(legend.position = "none") + 
                              labs(y = NULL, subtitle = "Australian Vehicle value effect (severity)") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0)), 
                            align = "hv")

# Make a grid of all plots, with country names
allplotsgrid <- plot_grid(#ggdraw() + draw_label(''),ggdraw() + draw_label("Out-of-sample Poisson Deviance", size = 12),ggdraw() + draw_label("Out-of-sample gamma Deviance", size = 12),
  allplotslist[[1]], allplotslist[[2]], 
  ncol = 2, rel_widths = c(0.5,0.5)#, rel_heights = c(0.05,0.24,0.24,0.24,0.24)
)

# Final plot grid, with legend
final_PDP_BM_VV_plot <- plot_grid(
  allplotsgrid,
  get_legend(NC_VIP_AUS),
  ncol = 1, rel_heights = c(0.9,0.1)
)

final_PDP_BM_VV_plot

# Save plot as PDF
ggsave("final_PDP_BM_VV_plot.pdf",
       final_PDP_BM_VV_plot, 
       device = cairo_pdf,
       width = 18,
       height = 10,
       scale = 1.2,
       units = "cm")

## ----- Surrogate plots -----

### ----- Australian example -----

load('NC_Surrogate_CANN_GBM_flex_AUS')
load('NC_PDP_CANN_GBM_flex_VehValue_AUS')
load('NC_PDP_CANN_GBM_flex_VehBody_AUS')

NC_PDP_VehBody_AUS <- NC_PDP_CANN_GBM_flex_VehBody_AUS %>% mutate(Testfold = factor(Testfold)) %>%
  mutate(x = factor(x, levels = c('Bus', 'Coupe', 'Motorized caravan', 'Roadster', 
                                  'Convertible', 'Minibus', 'Utility', 'Hardtop', 
                                  'Hatchback','Panel van', 'Sedan', 'Station wagon', 'Truck'))) %>% 
  ggplot(aes(x = x)) + geom_col(aes(y = y, group = Testfold, color = Testfold, fill = Testfold), size = 0.6, alpha = 0.7, position = position_dodge(width = 0.5)) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  theme(legend.position="bottom", 
        legend.direction="horizontal") + 
  xlab("Vehicle body type") + 
  ylab("Partial dependency effect")

NC_PDP_VehBody_AUS <- NC_PDP_CANN_GBM_flex_VehBody_AUS %>% filter(Testfold == 1) %>%
  mutate(x = factor(x, levels = c('Bus', 'Coupe', 'Motorized caravan', 'Roadster', 
                                  'Convertible', 'Minibus', 'Utility', 'Hardtop', 
                                  'Hatchback','Panel van', 'Sedan', 'Station wagon', 'Truck'))) %>% 
  ggplot(aes(x = x)) + geom_col(aes(y = y), color = "#116E8A", fill = "#116E8A", width = 0.5, alpha = 0.7) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  theme(legend.position="bottom", 
        legend.direction="horizontal") + 
  xlab("Vehicle body type") + 
  ylab("Partial dependency effect")

NC_PDP_VehBody_AUS_SURR <- NC_PDP_VehBody_AUS + geom_vline(xintercept = c(4.5, 7.5), size = 1.5, color = "black") +
  scale_x_discrete(labels=c("Bin 1","","","","Bin 2","","","Bin 3","","","","",""), name = "Surrogate bins")

ggsave("NC_PDP_VehBody_AUS.pdf",
       NC_PDP_VehBody_AUS, 
       device = 'pdf',
       width = 4,
       height = 4,
       scale = 3,
       units = "cm")

ggsave("NC_PDP_VehBody_AUS_SURR.pdf",
       NC_PDP_VehBody_AUS_SURR, 
       device = 'pdf',
       width = 4,
       height = 4,
       scale = 3,
       units = "cm")

NC_PDP_VehValue_AUS <- NC_PDP_CANN_GBM_flex_VehValue_AUS %>% mutate(Testfold = factor(Testfold)) %>% 
  ggplot(aes(x = x)) + geom_line(aes(y = y, group = Testfold, color = Testfold), size = 0.8) + 
  theme_bw() + 
  guides(group = guide_legend(nrow = 2, byrow = TRUE)) + 
  theme(legend.position="bottom", 
        legend.direction="horizontal") + 
  xlab("Vehicle value (x1000 AUD)") + 
  ylab("Partial dependency effect")

NC_PDP_VehValue_AUS_SURR <- NC_PDP_VehValue_AUS + geom_vline(xintercept = c(0.45,1.96,20.75), size = 1.5, color = "black") +
  scale_x_continuous(breaks=c(0,(1.96-0.45)/2+0.45,(20.75-1.96)/2+1.96,(34.56 -20.75)/2+20.75), 
                     labels=c("Bin 1","Bin 2", "Bin 3", "Bin 4"),name = "Surrogate bins")

### ----- French example -----

load('NC_Surrogate_CANN_GBM_flex_FR')

# Extract the four wanted PD effects from the surrogate model on French data
FR_PDP_BonusMalus <- NC_Surrogate_CANN_GBM_flex_FR$pd_fx$BonusMalus
FR_PDP_DrivAge <- NC_Surrogate_CANN_GBM_flex_FR$pd_fx$DrivAge
FR_PDP_Region <- NC_Surrogate_CANN_GBM_flex_FR$pd_fx$Region
FR_PDP_VehPower <- NC_Surrogate_CANN_GBM_flex_FR$pd_fx$VehPower

# French spatial plot

# Names in data and shape file are slightly different. 
# We also add the grouping of the surrogate here
french_regions <- tribble(
  ~old_region, ~new_region, ~group,
  "Rhone-Alpes", "Rh\xf4ne-Alpes", 'Group_1',
  "Picardie", "Picardie", 'Group_1',                
  "Aquitaine", "Aquitaine", 'Group_2',                
  "Nord-Pas-de-Calais", "Nord-Pas-de-Calais", 'Group_2',      
  "Languedoc-Roussillon", "Languedoc-Roussillon", 'Group_3',     
  "Pays-de-la-Loire", "Pays de la Loire", 'Group_4',           
  "Provence-Alpes-Cotes-D'Azur", "Provence-Alpes-C\xf4te d'Azur", 'Group_2',
  "Ile-de-France",  "\xcele-de-France", 'Group_1',            
  "Centre", "Centre", 'Group_1',
  "Corse",  "Corse", 'Group_5',                   
  "Auvergne", "Auvergne", 'Group_3', 
  "Poitou-Charentes", "Poitou-Charentes", 'Group_2',         
  "Bourgogne", "Bourgogne", 'Group_4',                  
  "Bretagne", "Bretagne", 'Group_1',               
  "Midi-Pyrenees", "Midi-Pyr\xe9n\xe9es", 'Group_3',             
  "Alsace", "Alsace", 'Group_2',                     
  "Basse-Normandie", "Basse-Normandie", 'Group_2',           
  "Champagne-Ardenne", "Champagne-Ardenne", 'Group_5',        
  "Franche-Comte", "Franche-Comt\xe9", 'Group_2',             
  "Limousin", "Limousin", 'Group_4',                  
  "Haute-Normandie", "Haute-Normandie",'Group_2'
)

# Read in French shape file
french_shape_sf <- st_read('./shape file French regions/ym781wr7170.shp', quiet = TRUE)
french_shape_sf <- st_transform(french_shape_sf, CRS("+proj=longlat +datum=WGS84"))
french_shape_sf <- st_simplify(french_shape_sf, dTolerance = 0.00001)

# Add surrogate groups per French region
french_shape_sf <- french_shape_sf %>% bind_cols(Group = c(2,2,3,1,2,4,1,1,5,5,2,2,3,4,NA,3,2,4,1,2,2,1)) %>% mutate(Group = paste0("Group_",Group))

# Remove Lorraine, not in data
french_shape_sf <- french_shape_sf %>% filter(name_1 != "Lorraine")

# Average PD per group
FR_Region_PDP_perGroup <- FR_PDP_Region %>% 
  left_join(french_regions, by=c("x"="old_region")) %>% 
  group_by(group) %>% 
  summarise(y = mean(y)) %>% mutate(y = round(y, digits = 4)) %>% mutate(y = factor(y))

# Join pdp info with the French Shape file
NC_PDP_spatial_FR <- left_join(french_shape_sf, FR_Region_PDP_perGroup, by = c("Group"="group"))

# Plot of the spatial effect on France map
NC_PDP_spatial_FR_plot <- tm_shape(NC_PDP_spatial_FR) + 
  tm_borders(col = "black") + 
  tm_fill(col = "y", title = "Spatial partial \ndependency effect \nper group", 
          textNA = "No Policyholders", 
          #style = "cont", 
          palette = "RdYlBu", 
          colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 0.6,
            legend.title.size = 0.8)

NC_PDP_spatial_FR_GrobPlot <- tmap_grob(NC_PDP_spatial_FR_plot)

# French bonus-malus plot

breaks <- tibble(lower = c(0,54.5,57.5,61.5,62.5,63.5,64.5,69.5,71.5,71.5,72.5,75.5,76.5,78.5,95.5,100.5),
                 upper = c(54.5,57.5,61.5,62.5,63.5,64.5,69.5,71.5,71.5,72.5,75.5,76.5,78.5,95.5,100.5,173.5),
                 gr = c(1:16))

FR_PDP_BonusMalus_withGroups <- fuzzy_join(FR_PDP_BonusMalus, breaks, by = c('x'='lower', 'x'='upper'), match_fun = list(`>`,`<=`))

NC_PDP_BonusMalus_FR_Surr <- FR_PDP_BonusMalus_withGroups %>% mutate(Group = paste0("Group_",gr)) %>% 
  ggplot(aes(x = x)) + geom_line(aes(y = y, color = Group), size = 1) + geom_point(aes(y = y, color = Group), size = 0.8) +
  theme_bw() + 
  scale_colour_discrete("RdYlBu") +
  theme(legend.position = 'none') + 
  xlab("Bonus-malus score") + 
  ylab("Partial dependency effect") + 
  geom_vline(xintercept = c(54.5,57.5,61.5,62.5,63.5,64.5,69.5,71.5,71.5,72.5,75.5,76.5,78.5,95.5,100.5))
  
# French Drivers age

NC_PDP_DrivAge_FR_Surr <- FR_PDP_DrivAge %>% bind_cols(Group = c("Group_1","Group_2","Group_3","Group_2","Group_4","Group_4","Group_1")) %>% 
  ggplot(aes(x = x)) + geom_col(aes(y = y, color = Group, fill = Group), size = 0.8, alpha = 0.7) + 
  theme_bw() + 
  scale_colour_discrete("RdYlBu") +
  theme(legend.position = 'none') + 
  xlab("Age bucket") + 
  ylab("Partial dependency effect") + 
  scale_x_discrete(labels = c('< 21','[21,26[','[26,30[','[30,40[','[40,50[','[50,70[','\u2265 70'))

# French Vehicle Power

NC_PDP_VehPower_FR_Surr <- FR_PDP_VehPower %>% 
  arrange(x) %>% 
  bind_cols(Group = c("Group_1","Group_2","Group_3","Group_4","Group_5","Group_6","Group_7","Group_7","Group_8","Group_8","Group_8","Group_8")) %>% 
  ggplot(aes(x = x)) + geom_col(aes(y = y, color = Group, fill = Group), size = 0.8, alpha = 0.7) + 
  theme_bw() + 
  scale_colour_discrete("RdYlBu") +
  theme(legend.position = 'none') + 
  xlab("Vehicle power") + 
  ylab("Partial dependency effect")


# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(NC_PDP_BonusMalus_FR_Surr + theme(legend.position = "none") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0)), 
                            NC_PDP_DrivAge_FR_Surr + theme(legend.position = "none") + 
                              labs(y = NULL) + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0),
                                    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)), 
                            NC_PDP_VehPower_FR_Surr + theme(legend.position = "none") + 
                              labs(y = NULL) + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0)), 
                            align = "hv")

# Make a grid of all plots, with country names
allplotsgrid <- plot_grid(#ggdraw() + draw_label(''),ggdraw() + draw_label("Out-of-sample Poisson Deviance", size = 12),ggdraw() + draw_label("Out-of-sample gamma Deviance", size = 12),
  allplotslist[[1]], allplotslist[[2]], allplotslist[[3]], 
  ncol = 3, rel_widths = c(0.3,0.3,0.3)#, rel_heights = c(0.05,0.24,0.24,0.24,0.24)
)

Surr_plot_French_comp <- plot_grid(NC_PDP_spatial_FR_GrobPlot, allplotsgrid, ncol = 2, rel_widths = c(0.3,0.7))

# Save plot as PDF
ggsave("Surr_plot_French_comp.pdf",
       Surr_plot_French_comp, 
       device = cairo_pdf,
       width = 32,
       height = 8,
       scale = 1.2,
       units = "cm")

# -----
# ----- THE END -----
# -----
