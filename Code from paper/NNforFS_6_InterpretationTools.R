# ----- SETUP R -----

# All setup is done in this section
# Installing and loading all packages, setting up tensorflow and keras
# Reading in data and small data prep
# Define metrics for later use

## ----- Install packages needed -----

used_packages <- c("sp", "vip","ggplot2",
                   "pdp","cplm","mltools",
                   "data.table", "keras", "tensorflow",
                   "reticulate", "tidyverse",
                   "gtools", "beepr", "gbm",
                   "gridExtra", "cowplot", "RColorBrewer",
                   "fuzzyjoin", "colorspace", "sf",
                   "tmap", "rgdal","egg", 
                   "tcltk", "xtable","progress",
                   "doParallel", "maidrr", "iml")
suppressMessages(packages <- lapply(used_packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x)
    library(x, character.only = TRUE)
  }
}))

#install.packages('devtools')
#devtools::install_github('henckr/maidrr')
#library("maidrr")

## ----- Setup Keras and Tensorflow -----

# Laptop enviroment
# use_python("C:/Users/u0086713/AppData/Local/r-miniconda/python.exe")
# use_condaenv("C:/Users/u0086713/AppData/Local/r-miniconda")
# install_tensorflow(method = "conda", conda = "C:/Users/u0086713/AppData/Local/r-miniconda/Scripts/conda.exe")

# Desktop enviroment
 use_python("C:/Users/Frynn/AppData/Local/r-miniconda/python.exe")
 use_condaenv("C:/Users/Frynn/AppData/Local/r-miniconda")
# install_tensorflow(method = "conda", conda = "C:/Users/Frynn/AppData/Local/r-miniconda/Scripts/conda.exe")

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
#location_datasets <- "/home/lynn/Dropbox/MTPL Data Sets"
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
  .GlobalEnv$poisson_metric(y_true, y_pred)
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
                     trainable_output = TRUE,
                     output_modelinfo = TRUE)
})

CA_opt_AUS <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = CA_data_AUS_GBM[[fold]], 
                     flags_list = AUS_CA_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                     cat_vars = cat_AUS,
                     trainable_output = TRUE,
                     output_modelinfo = TRUE)
})

# Belgian
NC_opt_BE <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = NC_data_BE_GBM[[fold]], 
                     flags_list = BE_NC_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_BE[[fold]],
                     cat_vars = cat_BE,
                     trainable_output = TRUE,
                     output_modelinfo = TRUE)
})

CA_opt_BE <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = CA_data_BE_GBM[[fold]], 
                     flags_list = BE_CA_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_BE[[fold]],
                     cat_vars = cat_BE,
                     trainable_output = TRUE,
                     output_modelinfo = TRUE)
})

# French
NC_opt_FR <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = NC_data_FR_GBM[[fold]],
                     flags_list = FR_NC_CANN_GBM_flex[[fold]],
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_FR[[fold]],
                     cat_vars = cat_FR,
                     trainable_output = TRUE,
                     output_modelinfo = TRUE)
})

CA_opt_FR <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = CA_data_FR_GBM[[fold]],
                     flags_list = FR_CA_CANN_GBM_flex[[fold]],
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_FR[[fold]],
                     cat_vars = cat_FR,
                     trainable_output = TRUE,
                     output_modelinfo = TRUE)
})

# Norwegian
NC_opt_NOR <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = NC_data_NOR_GBM[[fold]], 
                     flags_list = NOR_NC_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                     cat_vars = cat_NOR,
                     trainable_output = TRUE,
                     output_modelinfo = TRUE)
})

CA_opt_NOR <- lapply(1:6, function(fold){
  single_CANN_run_AE(fold_data = CA_data_NOR_GBM[[fold]], 
                     flags_list = NOR_CA_CANN_GBM_flex[[fold]], 
                     random_val_split = 0.2,
                     autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                     cat_vars = cat_NOR,
                     trainable_output = TRUE,
                     output_modelinfo = TRUE)
})


sapply(c(1:6), function(x){NC_opt_FR[[x]]$results})
sapply(c(1:6), function(x){CA_opt_FR[[x]]$results})

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

## ----- New gamma loss function for surrogates ----

gamma_loss <- function(y_true, y_pred, w_case){
  sum(2*w_case*(((y_true-y_pred)/y_pred)-log(y_true/y_pred)))/length(y_true)
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


# We define object that combine all info needed to make predictions with out CANN models
# This allows us to use more parallelization for the interpretation tools

## ----- Define objects for prediction -----

# List of objects for AUS predictions
NC_object_AUS <- lapply(1:6, function(fold){
  list(NN_model = NC_opt_AUS[[fold]],
       GBM_model = oos_freq_GBM_AUS[[fold]][[3]],
       scale_info = NC_scaleinfo_AUS %>% filter(Testfold == fold),
       latlong_conversion = NULL,
       problem = 'Frequency')
})
CA_object_AUS <- lapply(1:6, function(fold){
  list(NN_model = CA_opt_AUS[[fold]],
       GBM_model = oos_sev_GBM_AUS[[fold]][[3]],
       scale_info = CA_scaleinfo_AUS %>% filter(Testfold == fold),
       latlong_conversion = NULL,
       problem = 'Severity')
})

# List of objects for BE predictions
NC_object_BE <- lapply(1:6, function(fold){
  list(NN_model = NC_opt_BE[[fold]],
       GBM_model = gbm_fits[[fold]],
       scale_info = NC_scaleinfo_BE %>% filter(Testfold == fold),
       latlong_conversion = latlong_per_postalcode,
       problem = 'Frequency')
})
CA_object_BE <- lapply(1:6, function(fold){
  list(NN_model = CA_opt_BE[[fold]],
       GBM_model = gbm_fits[[6+fold]],
       scale_info = CA_scaleinfo_BE %>% filter(Testfold == fold),
       latlong_conversion = latlong_per_postalcode,
       problem = 'Severity')
})

# List of objects for FR predictions
NC_object_FR <- lapply(1:6, function(fold){
  list(NN_model = NC_opt_FR[[fold]],
       GBM_model = oos_freq_GBM_FR[[fold]][[3]],
       scale_info = NC_scaleinfo_FR %>% filter(Testfold == fold),
       latlong_conversion = NULL,
       problem = 'Frequency')
})
CA_object_FR <- lapply(1:6, function(fold){
  list(NN_model = CA_opt_FR[[fold]],
       GBM_model = oos_sev_GBM_FR[[fold]][[3]],
       scale_info = CA_scaleinfo_FR %>% filter(Testfold == fold),
       latlong_conversion = NULL,
       problem = 'Severity')
})

# List of objects for NOR predictions
NC_object_NOR <- lapply(1:6, function(fold){
  list(NN_model = NC_opt_NOR[[fold]],
       GBM_model = oos_freq_GBM_NOR[[fold]][[3]],
       scale_info = NULL,
       latlong_conversion = NULL,
       problem = 'Frequency')
})
CA_object_NOR <- lapply(1:6, function(fold){
  list(NN_model = CA_opt_NOR[[fold]],
       GBM_model = oos_sev_GBM_NOR[[fold]][[3]],
       scale_info = NULL,
       latlong_conversion = NULL,
       problem = 'Severity')
})

## ---- Custom prediction functions for the CANN models -----

# Function which makes CANN prediction, average prediction per set returned
CANN_model_predictions <- function(object, newdata){
  
  # One-hot encode all categorical variables
  train_cat_data <- lapply(object$NN_model$cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  # Bind the one-hot variables into a matrix
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  # If a latlong conversion is supplied, apply it
  if(!is.null(object$latlong_conversion)){
    cont_data_LL <- newdata %>% 
      left_join(object$latlong_conversion,by=c("postcode"))
  } else {
    cont_data_LL <- newdata
  }
  
  # Scale the continuous variables
  if(!is.null(object$scale_info)){
    cont_data <- cont_data_LL %>%
      dplyr::select(object$NN_model$other_vars) %>%
      .GlobalEnv$scale_withPar(object$NN_model$other_vars, 
                               object$scale_info %>% 
                                 arrange(factor(Variable, levels = object$NN_model$other_vars)))
  } else {
    cont_data <- cont_data_LL %>% dplyr::select(c())
  }
  
  # Add the GBM predictions, offset based on whether it is frequency or severity model
  if(object$problem == 'Frequency'){
    new_prediction <- .GlobalEnv$NC_gbm_prediction_perpoint(object = object$GBM_model, newdata = cont_data_LL)
  } else {
    new_prediction <- .GlobalEnv$CA_gbm_prediction_perpoint(object = object$GBM_model, newdata = cont_data_LL)
  }
  
  # Bind all data into a list of matrices
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>% data.matrix())

  # Make predictions on the data with the supplied CANN model
  return(object$NN_model$model %>% predict(train_mat, type = "response", verbose = 0) %>% mean)
}

# Function which makes CANN prediction, prediction per point returned
CANN_model_predictions_perpoint <- function(object, newdata){
  
  # One-hot encode all categorical variables
  train_cat_data <- lapply(object$NN_model$cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  # Bind the one-hot variables into a matrix
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  # If a latlong conversion is supplied, apply it
  if(!is.null(object$latlong_conversion)){
    cont_data_LL <- newdata %>% 
      left_join(object$latlong_conversion,by=c("postcode"))
  } else {
    cont_data_LL <- newdata
  }
  
  # Scale the continuous variables
  if(!is.null(object$scale_info)){
    cont_data <- cont_data_LL %>%
      dplyr::select(object$NN_model$other_vars) %>%
      .GlobalEnv$scale_withPar(object$NN_model$other_vars, 
                               object$scale_info %>% 
                                 arrange(factor(Variable, levels = object$NN_model$other_vars)))
  } else {
    cont_data <- cont_data_LL %>% dplyr::select(c())
  }
  
  # Add the GBM predictions, offset based on whether it is frequency or severity model
  if(object$problem == 'Frequency'){
    new_prediction <- .GlobalEnv$NC_gbm_prediction_perpoint(object = object$GBM_model, newdata = cont_data_LL)
  } else {
    new_prediction <- .GlobalEnv$CA_gbm_prediction_perpoint(object = object$GBM_model, newdata = cont_data_LL)
  }
  
  # Bind all data into a list of matrices
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat,
                    new_prediction %>% log %>% data.matrix())
  
  # Make predictions on the data with the supplied CANN model
  return(object$NN_model$model %>% predict(train_mat, type = "response", verbose = 0))
}

# -----
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

# We made a small adjustment to the Maidrr package, to fit our smaller data sets.
source('autotune_FH_print.R')
source('insights_FH_print.R')

# GBM prediction functions
NC_gbm_prediction_perpoint <- function(object, newdata){
  if(!is.null(object$latlong_conversion)){
    newdata %>% left_join(object$latlong_conversion,by=c("postcode")) %>% 
      predict_model.gbm(object$GBM_model, .) * newdata$expo
  } else {
    predict_model.gbm(object, newdata) * newdata$expo
  }
}

NC_gbm_prediction <- function(object, newdata){
  NC_gbm_prediction_perpoint(object, newdata) %>% mean
}

CA_gbm_prediction_perpoint <- function(object, newdata){
  if(!is.null(object$latlong_conversion)){
    newdata %>% left_join(object$latlong_conversion,by=c("postcode")) %>% 
      predict_model.gbm(object$GBM_model, .) %>% exp
  } else {
    predict_model.gbm(object, newdata) %>% exp
  }
}

CA_gbm_prediction <- function(object, newdata){
  CA_gbm_prediction_perpoint(object, newdata) %>% mean
}

## ----- Make data samples to calculate interpretations -----

# Australian
NC_data_slice_AUS <- data_AUS %>% slice_sample(n=10000)
CA_data_slice_AUS <- data_AUS %>% filter(nclaims>0, !is.na(average)) %>% slice_sample(n=4000)
#save(NC_data_slice_AUS, CA_data_slice_AUS, file = "data_slices_AUS")

NC_data_slice_testfold_AUS <- lapply(1:6, function(fold) data_AUS %>%  filter(fold_nr != fold) %>% sample_n(size = 10000) )
CA_data_slice_testfold_AUS <- lapply(1:6, function(fold) data_AUS %>% filter(nclaims>0, !is.na(average)) %>% filter(fold_nr != fold) %>% sample_n(size = 3000) )
#save(NC_data_slice_testfold_AUS, CA_data_slice_testfold_AUS, file = "data_slice_testfold_AUS")

# Belgian
NC_data_slice_BE <- data_BE_PC %>% slice_sample(n=10000)
CA_data_slice_BE <- data_BE_PC %>% filter(nclaims>0, !is.na(average)) %>% slice_sample(n=10000)
#save(NC_data_slice_BE, CA_data_slice_BE, file = "data_slices_BE")

NC_data_slice_testfold_BE <- lapply(1:6, function(fold) data_BE_PC %>%  filter(fold_nr != fold) %>% sample_n(size = 10000) )
CA_data_slice_testfold_BE <- lapply(1:6, function(fold) data_BE_PC %>% filter(nclaims>0, !is.na(average)) %>% filter(fold_nr != fold) %>% sample_n(size = 10000) )
#save(NC_data_slice_testfold_BE, CA_data_slice_testfold_BE, file = "data_slice_testfold_BE")

# French
NC_data_slice_FR <- data_FR %>% slice_sample(n=10000)
CA_data_slice_FR <- data_FR %>% filter(nclaims>0, !is.na(average)) %>% slice_sample(n=10000)
#save(NC_data_slice_FR, CA_data_slice_FR, file = "data_slices_FR")

NC_data_slice_testfold_FR <- lapply(1:6, function(fold) data_FR %>%  filter(fold_nr != fold) %>% sample_n(size = 10000) )
CA_data_slice_testfold_FR <- lapply(1:6, function(fold) data_FR %>% filter(nclaims>0, !is.na(average)) %>% filter(fold_nr != fold) %>% sample_n(size = 10000) )
#save(NC_data_slice_testfold_FR, CA_data_slice_testfold_FR, file = "data_slice_testfold_FR")

# Norwegian
NC_data_slice_NOR <- data_NOR %>% slice_sample(n=10000)
CA_data_slice_NOR <- data_NOR %>% filter(nclaims>0, !is.na(average)) %>% slice_sample(n=4000)
#save(NC_data_slice_NOR, CA_data_slice_NOR, file = "data_slices_NOR")

## ----- Read in data samples -----

load('data_slices_AUS')
load('data_slice_testfold_AUS')
load('data_slices_BE')
load('data_slice_testfold_BE')
load('data_slices_FR')
load('data_slice_testfold_FR')
load('data_slices_NOR')

# ----
## ----- Australia -----
### ----- Surrogate -----

# We round values with high level of unique obvservations
data_AUS_ord <- data_AUS %>% mutate(VehValue = plyr::round_any(VehValue, 0.1))

NC_CANN_GBM_flex_SURR_allFolds_AUS <- lapply(1:6, function(fold){
  # Tune the surrogate technique and determine best split
  autotune_FH_print(NC_object_AUS[[fold]],
                   data = data_AUS_ord %>% filter(fold_nr != fold),
                   vars = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'),
                   target = 'nclaims',
                   hcut = 0.75,
                   pred_fun = CANN_model_predictions,
                   lambdas = as.vector(outer(seq(1, 10, 2), 10^(-6:-2))),
                   max_ngrps = 15,
                   nfolds = 5,
                   strat_vars = c('nclaims', 'expo'),
                   glm_par = alist(family = poisson(link = 'log'),
                                   offset = log(expo)),
                   err_fun = maidrr::poi_dev,
                   out_pds = TRUE,
                   ncores=1,
                   full_data = data_AUS_ord)
})
save(NC_CANN_GBM_flex_SURR_allFolds_AUS, file = 'NC_CANN_GBM_flex_SURR_allFolds_AUS')

CA_CANN_GBM_flex_SURR_allFolds_AUS <- lapply(1:6, function(fold){
  # Tune the surrogate technique and determine best split
  autotune_FH(CA_object_AUS[[fold]],
              data = data_AUS_ord %>% filter(fold_nr != fold) %>% filter(nclaims>0 & !is.na(average)),
              vars = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'),
              target = 'average',
              hcut = 0.75,
              pred_fun = CANN_model_predictions,
              lambdas = as.vector(outer(seq(1, 10, 1), 10^(-5:1))),
              max_ngrps = 15,
              nfolds = 5,
              strat_vars = c('average', 'nclaims'),
              glm_par = alist(family = Gamma(link = "log"),
                              weights = nclaims),
              err_fun = gamma_loss,
              out_pds = TRUE,
              ncores=1,
              full_data = data_AUS_ord)
})
save(CA_CANN_GBM_flex_SURR_allFolds_AUS, file = 'CA_CANN_GBM_flex_SURR_allFolds_AUS')

### ----- Partial Dependence -----

# All variables ranges for PDP calculation
all_var_ranges_AUS <- list('VehValue' = seq(0, 35, by = 0.2), 
                          'VehAge' = data_AUS %>% pull(VehAge) %>% unique, 
                          'VehBody' = data_AUS %>% pull(VehBody) %>% unique,  
                          'Gender' = data_AUS %>% pull(Gender) %>% unique, 
                          'DrivAge' = data_AUS %>% pull(DrivAge) %>% unique)

# To make the functions a bit more readable, we apply this shortening of notation
sn <- function(vector){
  setNames(vector,vector)
}
# This makes sure that the lapply function returns named objects

NC_PDP_GBM_AllVARS_AUS <- lapply(sn(c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_object_AUS[[fold]]$GBM_model,
      var = var,
      grid = data.frame(all_var_ranges_AUS[[var]]) %>% setNames(var),
      data = NC_data_slice_AUS, 
      fun = NC_gbm_prediction,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_GBM_AllVARS_AUS, file = 'NC_PDP_GBM_AllVARS_AUS')

CA_PDP_GBM_AllVARS_AUS <- lapply(sn(c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_object_AUS[[fold]]$GBM_model,
      var = var,
      grid = data.frame(all_var_ranges_AUS[[var]]) %>% setNames(var),
      data = CA_data_slice_AUS, 
      fun = CA_gbm_prediction,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_GBM_AllVARS_AUS, file = 'CA_PDP_GBM_AllVARS_AUS')

NC_PDP_CANN_GBM_flex_AllVARS_AUS <- lapply(sn(c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_object_AUS[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_AUS[[var]]) %>% setNames(var),
      data = NC_data_slice_AUS, 
      fun = CANN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_CANN_GBM_flex_AllVARS_AUS, file = 'NC_PDP_CANN_GBM_flex_AllVARS_AUS')

CA_PDP_CANN_GBM_flex_AllVARS_AUS <- lapply(sn(c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_object_AUS[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_AUS[[var]]) %>% setNames(var),
      data = CA_data_slice_AUS, 
      fun = CANN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_CANN_GBM_flex_AllVARS_AUS, file = 'CA_PDP_CANN_GBM_flex_AllVARS_AUS')


### ----- Variable Importance -----

# Frequency - Initial Model
NC_VI_GBM_AUS <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_AUS,
    variables = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'), 
    model = NC_object_AUS[[fold]]$GBM_model,
    pred_fun = NC_gbm_prediction_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_GBM_AUS, file = 'NC_VI_GBM_AUS')

# Frequency - CANN Model
NC_VI_CANN_GBM_flex_AUS <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_AUS,
    variables = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'), 
    model = NC_object_AUS[[fold]],
    pred_fun = CANN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_CANN_GBM_flex_AUS, file = 'NC_VI_CANN_GBM_flex_AUS')

# Severity - Initial Model
CA_VI_GBM_AUS <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_AUS,
    variables = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'),  
    model = CA_object_AUS[[fold]]$GBM_model,
    pred_fun = CA_gbm_prediction_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_GBM_AUS, file = 'CA_VI_GBM_AUS')

# Severity - CANN Model
CA_VI_CANN_GBM_flex_AUS <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_AUS,
    variables = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'),  
    model = CA_object_AUS[[fold]],
    pred_fun = CANN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_CANN_GBM_flex_AUS, file = 'CA_VI_CANN_GBM_flex_AUS')

# -----
## ----- Belgium -----


### ----- Surrogate -----

data_BE_ord <- data_BE_PC %>% mutate(coverage = factor(coverage, ordered = T))

NC_CANN_GBM_flex_SURR_allFolds_BE <- lapply(1:6, function(fold){
  # Tune the surrogate technique and determine best split
  autotune_FH(NC_object_BE[[fold]],
              data = data_BE_ord %>% filter(fold_nr != fold),
              vars = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet', 'postcode'),
              target = 'nclaims',
              hcut = 0.75,
              pred_fun = CANN_model_predictions,
              lambdas = as.vector(outer(seq(1, 10, 2), 10^(-6:-2))),
              max_ngrps = 15,
              nfolds = 5,
              strat_vars = c('nclaims', 'expo'),
              glm_par = alist(family = poisson(link = 'log'),
                              offset = log(expo)),
              err_fun = maidrr::poi_dev,
              out_pds = TRUE,
              ncores=1,
              full_data = data_BE_ord)
})
save(NC_CANN_GBM_flex_SURR_allFolds_BE, file = 'NC_CANN_GBM_flex_SURR_allFolds_BE')

CA_CANN_GBM_flex_SURR_allFolds_BE <- lapply(1:6, function(fold){
  # Tune the surrogate technique and determine best split
  autotune_FH(CA_object_BE[[fold]],
              data = data_BE_ord %>% filter(fold_nr != fold) %>% filter(nclaims>0 & !is.na(average)),
              vars = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet', 'postcode'),
              target = 'average',
              hcut = 0.75,
              pred_fun = CANN_model_predictions,
              lambdas = as.vector(outer(seq(1, 10, 1), 10^(-5:1))),
              max_ngrps = 15,
              nfolds = 5,
              strat_vars = c('average', 'nclaims'),
              glm_par = alist(family = Gamma(link = "log"),
                              weights = nclaims),
              err_fun = gamma_loss,
              out_pds = TRUE,
              ncores=1,
              full_data = data_BE_ord)
})
save(CA_CANN_GBM_flex_SURR_allFolds_BE, file = 'CA_CANN_GBM_flex_SURR_allFolds_BE')

### ----- Partial dependency -----

# Table of all postalcodes in Belgium
belgium_shape_sf <- st_read('./shape file Belgie postcodes/npc96_region_Project1.shp', quiet = TRUE)
all_postalcodes <- belgium_shape_sf %>% pull(POSTCODE) %>% unique

# All variables ranges for PDP calculation
all_var_ranges_BE <- list('coverage' = data_BE %>% pull(coverage) %>% unique, 
                          'ageph' = c(18:95), 
                          'sex' = data_BE %>% pull(sex) %>% unique,  
                          'bm' = c(0:22), 
                          'power' = c(10:243), 
                          'agec' = c(0:48),  
                          'fuel' = data_BE %>% pull(fuel) %>% unique,  
                          'use' = data_BE %>% pull(use) %>% unique,  
                          'fleet' = data_BE %>% pull(fleet) %>% unique,  
                          'postcode' = all_postalcodes)

# To make the functions a bit more readable, we apply this shortening of notation
sn <- function(vector){
  setNames(vector,vector)
}
# This makes sure that the lapply function returns named objects

NC_PDP_GBM_AllVARS_BE <- lapply(sn(c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet','postcode')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = list(GBM_model = NC_object_BE[[fold]]$GBM_model, latlong_conversion = latlong_per_postalcode),
      var = var,
      grid = data.frame(all_var_ranges_BE[[var]]) %>% setNames(var),
      data = NC_data_slice_BE, 
      fun = NC_gbm_prediction,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_GBM_AllVARS_BE, file = 'NC_PDP_GBM_AllVARS_BE')

CA_PDP_GBM_AllVARS_BE <- lapply(sn(c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet','postcode')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = list(GBM_model = CA_object_BE[[fold]]$GBM_model, latlong_conversion = latlong_per_postalcode),
      var = var,
      grid = data.frame(all_var_ranges_BE[[var]]) %>% setNames(var),
      data = CA_data_slice_BE, 
      fun = CA_gbm_prediction,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_GBM_AllVARS_BE, file = 'CA_PDP_GBM_AllVARS_BE')

NC_PDP_CANN_GBM_flex_AllVARS_BE <- lapply(sn(c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet','postcode')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_object_BE[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_BE[[var]]) %>% setNames(var),
      data = NC_data_slice_BE, 
      fun = CANN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_CANN_GBM_flex_AllVARS_BE, file = 'NC_PDP_CANN_GBM_flex_AllVARS_BE')

CA_PDP_CANN_GBM_flex_AllVARS_BE <- lapply(sn(c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet','postcode')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_object_BE[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_BE[[var]]) %>% setNames(var),
      data = CA_data_slice_BE, 
      fun = CANN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_CANN_GBM_flex_AllVARS_BE, file = 'CA_PDP_CANN_GBM_flex_AllVARS_BE')

### ----- Variable importance -----

# Frequency - Initial Model
NC_VI_GBM_BE <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_BE,
    variables = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet', 'postcode'), 
    model = list(GBM_model = NC_object_BE[[fold]]$GBM_model, latlong_conversion = latlong_per_postalcode),
    pred_fun = NC_gbm_prediction_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_GBM_BE, file = 'NC_VI_GBM_BE')

# Frequency - CANN Model
NC_VI_CANN_GBM_flex_BE <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_BE,
    variables = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet', 'postcode'), 
    model = NC_object_BE[[fold]],
    pred_fun = CANN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_CANN_GBM_flex_BE, file = 'NC_VI_CANN_GBM_flex_BE')

# Severity - Initial Model
CA_VI_GBM_BE <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_BE,
    variables = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet', 'postcode'), 
    model = list(GBM_model = CA_object_BE[[fold]]$GBM_model, latlong_conversion = latlong_per_postalcode),
    pred_fun = CA_gbm_prediction_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_GBM_BE, file = 'CA_VI_GBM_BE')

# Severity - CANN Model
CA_VI_CANN_GBM_flex_BE <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_BE,
    variables = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet', 'postcode'), 
    model = CA_object_BE[[fold]],
    pred_fun = CANN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_CANN_GBM_flex_BE, file = 'CA_VI_CANN_GBM_flex_BE')

# -----
## ----- French -----

### ----- Surrogate -----

data_FR_ord <- data_FR %>% mutate(across(all_of(c('VehAge','DrivAge')), ~ 
                                           factor(., ordered = TRUE)))  %>% 
  mutate(Density = plyr::round_any(Density, 0.5)) %>% 
  mutate(BonusMalus = plyr::round_any(BonusMalus, 2))


NC_CANN_GBM_flex_SURR_allFolds_FR <- lapply(1:6, function(fold){
  # Tune the surrogate technique and determine best split
  autotune_FH(NC_object_FR[[fold]],
              data = data_FR_ord %>% filter(fold_nr != fold),
              vars = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"), 
              target = 'nclaims',
              hcut = 0.75,
              pred_fun = CANN_model_predictions,
              lambdas = as.vector(outer(seq(1, 10, 2), 10^(-6:-2))),
              max_ngrps = 15,
              nfolds = 5,
              strat_vars = c('nclaims', 'expo'),
              glm_par = alist(family = poisson(link = 'log'),
                              offset = log(expo)),
              err_fun = maidrr::poi_dev,
              out_pds = TRUE,
              ncores=1,
              full_data = data_FR_ord)
})
save(NC_CANN_GBM_flex_SURR_allFolds_FR, file = 'NC_CANN_GBM_flex_SURR_allFolds_FR')

CA_CANN_GBM_flex_SURR_allFolds_FR <- lapply(1:6, function(fold){
  # Tune the surrogate technique and determine best split
  autotune_FH(CA_object_FR[[fold]],
              data = data_FR_ord %>% filter(fold_nr != fold) %>% filter(nclaims>0 & !is.na(average)),
              vars = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"), 
              target = 'average',
              hcut = 0.75,
              pred_fun = CANN_model_predictions,
              lambdas = as.vector(outer(seq(1, 10, 1), 10^(-5:1))),
              max_ngrps = 15,
              nfolds = 5,
              strat_vars = c('average', 'nclaims'),  
              glm_par = alist(family = Gamma(link = "log"),
                              weights = nclaims),
              
              err_fun = gamma_loss,
              out_pds = TRUE, 
              ncores=1,
              full_data = data_FR_ord)
})
save(CA_CANN_GBM_flex_SURR_allFolds_FR, file = 'CA_CANN_GBM_flex_SURR_allFolds_FR')

### ----- Partial Dependence -----

# All variables ranges for PDP calculation
all_var_ranges_FR <- list('VehPower' = data_FR %>% pull(VehPower) %>% unique, 
                          'VehAge' = data_FR %>% pull(VehAge) %>% unique, 
                          'DrivAge' = data_FR %>% pull(DrivAge) %>% unique,  
                          'BonusMalus' = c(50:230), 
                          'VehBrand' = data_FR %>% pull(VehBrand) %>% unique, 
                          'VehGas' = data_FR %>% pull(VehGas) %>% unique,  
                          'Area' = data_FR %>% pull(Area) %>% unique,  
                          'Density' = seq(0,10.3,by = 0.05),  
                          'Region' = data_FR %>% pull(Region) %>% unique)

# To make the functions a bit more readable, we apply this shortening of notation
sn <- function(vector){
  setNames(vector,vector)
}

# This makes sure that the lapply function returns named objects

NC_PDP_GBM_AllVARS_FR <- lapply(sn(c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_object_FR[[fold]]$GBM_model,
      var = var,
      grid = data.frame(all_var_ranges_FR[[var]]) %>% setNames(var),
      data = NC_data_slice_FR, 
      fun = NC_gbm_prediction,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_GBM_AllVARS_FR, file = 'NC_PDP_GBM_AllVARS_FR')

CA_PDP_GBM_AllVARS_FR <- lapply(sn(c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_object_FR[[fold]]$GBM_model,
      var = var,
      grid = data.frame(all_var_ranges_FR[[var]]) %>% setNames(var),
      data = CA_data_slice_FR, 
      fun = CA_gbm_prediction,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_GBM_AllVARS_FR, file = 'CA_PDP_GBM_AllVARS_FR')

NC_PDP_CANN_GBM_flex_AllVARS_FR <- lapply(sn(c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_object_FR[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_FR[[var]]) %>% setNames(var),
      data = NC_data_slice_FR, 
      fun = CANN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_CANN_GBM_flex_AllVARS_FR, file = 'NC_PDP_CANN_GBM_flex_AllVARS_FR')

CA_PDP_CANN_GBM_flex_AllVARS_FR <- lapply(sn(c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_object_FR[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_FR[[var]]) %>% setNames(var),
      data = CA_data_slice_FR, 
      fun = CANN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_CANN_GBM_flex_AllVARS_FR, file = 'CA_PDP_CANN_GBM_flex_AllVARS_FR')

### ----- Variable Importance -----

# Frequency - Initial Model
NC_VI_GBM_FR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_FR,
    variables = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"), 
    model = NC_object_FR[[fold]]$GBM_model,
    pred_fun = NC_gbm_prediction_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_GBM_FR, file = 'NC_VI_GBM_FR')

# Frequency - CANN Model
NC_VI_CANN_GBM_flex_FR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_FR,
    variables = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"), 
    model = NC_object_FR[[fold]],
    pred_fun = CANN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_CANN_GBM_flex_FR, file = 'NC_VI_CANN_GBM_flex_FR')

# Severity - Initial Model
CA_VI_GBM_FR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_FR,
    variables = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"), 
    model = CA_object_FR[[fold]]$GBM_model,
    pred_fun = CA_gbm_prediction_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_GBM_FR, file = 'CA_VI_GBM_FR')

# Severity - CANN Model
CA_VI_CANN_GBM_flex_FR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_FR,
    variables = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"), 
    model = CA_object_FR[[fold]],
    pred_fun = CANN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_CANN_GBM_flex_FR, file = 'CA_VI_CANN_GBM_flex_FR')

# -----
## ----- Norwegian -----

### ----- Surrogate -----

### ----- Partial Dependence -----

# All variables ranges for PDP calculation
all_var_ranges_NOR <- list('Male' = data_NOR %>% pull(Male) %>% unique, 
                           'Young' = data_NOR %>% pull(Young) %>% unique, 
                           'DistLimit' = data_NOR %>% pull(DistLimit) %>% unique,  
                           'GeoRegion' = data_NOR %>% pull(GeoRegion) %>% unique)

# To make the functions a bit more readable, we apply this shortening of notation
sn <- function(vector){
  setNames(vector,vector)
}
# This makes sure that the lapply function returns named objects

NC_PDP_GBM_AllVARS_NOR <- lapply(sn(c('Male', 'Young', 'DistLimit', 'GeoRegion')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_object_NOR[[fold]]$GBM_model,
      var = var,
      grid = data.frame(all_var_ranges_NOR[[var]]) %>% setNames(var),
      data = NC_data_slice_NOR, 
      fun = NC_gbm_prediction,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_GBM_AllVARS_NOR, file = 'NC_PDP_GBM_AllVARS_NOR')

CA_PDP_GBM_AllVARS_NOR <- lapply(sn(c('Male', 'Young', 'DistLimit', 'GeoRegion')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_object_NOR[[fold]]$GBM_model,
      var = var,
      grid = data.frame(all_var_ranges_NOR[[var]]) %>% setNames(var),
      data = CA_data_slice_NOR, 
      fun = CA_gbm_prediction,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_GBM_AllVARS_NOR, file = 'CA_PDP_GBM_AllVARS_NOR')

NC_PDP_CANN_GBM_flex_AllVARS_NOR <- lapply(sn(c('Male', 'Young', 'DistLimit', 'GeoRegion')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_object_NOR[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_NOR[[var]]) %>% setNames(var),
      data = NC_data_slice_NOR, 
      fun = CANN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_CANN_GBM_flex_AllVARS_NOR, file = 'NC_PDP_CANN_GBM_flex_AllVARS_NOR')

CA_PDP_CANN_GBM_flex_AllVARS_NOR <- lapply(sn(c('Male', 'Young', 'DistLimit', 'GeoRegion')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_object_NOR[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_NOR[[var]]) %>% setNames(var),
      data = CA_data_slice_NOR, 
      fun = CANN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_CANN_GBM_flex_AllVARS_NOR, file = 'CA_PDP_CANN_GBM_flex_AllVARS_NOR')

### ----- Variable Importance -----

# Frequency - Initial Model
NC_VI_GBM_NOR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_NOR,
    variables =  c('Male', 'Young', 'DistLimit', 'GeoRegion'),
    model = NC_object_NOR[[fold]]$GBM_model,
    pred_fun = NC_gbm_prediction_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_GBM_NOR, file = 'NC_VI_GBM_NOR')

# Frequency - CANN Model
NC_VI_CANN_GBM_flex_NOR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_NOR,
    variables =  c('Male', 'Young', 'DistLimit', 'GeoRegion'),
    model = NC_object_NOR[[fold]],
    pred_fun = CANN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_CANN_GBM_flex_NOR, file = 'NC_VI_CANN_GBM_flex_NOR')

# Severity - Initial Model
CA_VI_GBM_NOR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_NOR,
    variables = c('Male', 'Young', 'DistLimit', 'GeoRegion'), 
    model = CA_object_NOR[[fold]]$GBM_model,
    pred_fun = CA_gbm_prediction_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_GBM_NOR, file = 'CA_VI_GBM_NOR')

# Severity - CANN Model
CA_VI_CANN_GBM_flex_NOR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_NOR,
    variables = c('Male', 'Young', 'DistLimit', 'GeoRegion'), 
    model = CA_object_NOR[[fold]],
    pred_fun = CANN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_CANN_GBM_flex_NOR, file = 'CA_VI_CANN_GBM_flex_NOR')

# -----
# ----- COMPARISON MODELS -----

# We fit extra models for comparisons.

## ----- Comparison of PDP with FFNN -----

# Prediction function for FFNN with autoencoder embedding
FFNN_model_predictions <- function(object, newdata){
  
  # One-hot encode all categorical variables
  train_cat_data <- lapply(object$NN_model$cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  # Bind the one-hot variables into a matrix
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  # If a latlong conversion is supplied, apply it
  if(!is.null(object$latlong_conversion)){
    cont_data_LL <- newdata %>% 
      left_join(object$latlong_conversion,by=c("postcode"))
  } else {
    cont_data_LL <- newdata
  }
  
  # Scale the continuous variables
  if(!is.null(object$scale_info)){
    cont_data <- cont_data_LL %>%
      dplyr::select(object$NN_model$other_vars) %>%
      .GlobalEnv$scale_withPar(object$NN_model$other_vars[!object$NN_model$other_vars=='expo'], 
                               object$scale_info %>% 
                                 arrange(factor(Variable, 
                                                levels = object$NN_model$other_vars[!object$NN_model$other_vars=='expo'])))
  } else {
    cont_data <- cont_data_LL %>% dplyr::select(object$NN_model$other_vars)
  }
  
  # Bind all data into a list of matrices
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat)
  
  # Make predictions on the data with the supplied CANN model
  return(object$NN_model$model %>% predict(train_mat, type = "response", verbose = 0) %>% mean)
}

FFNN_model_predictions_perpoint <- function(object, newdata){
  
  # One-hot encode all categorical variables
  train_cat_data <- lapply(object$NN_model$cat_vars,function(var_FH){
    newdata %>% 
      dplyr::pull(var_FH) %>% 
      data.table::as.data.table() %>% 
      mltools::one_hot(cols=".") %>% 
      data.matrix()
  })
  
  # Bind the one-hot variables into a matrix
  train_cat_data_concat <- do.call('cbind',train_cat_data)
  
  # If a latlong conversion is supplied, apply it
  if(!is.null(object$latlong_conversion)){
    cont_data_LL <- newdata %>% 
      left_join(object$latlong_conversion,by=c("postcode"))
  } else {
    cont_data_LL <- newdata
  }
  
  # Scale the continuous variables
  if(!is.null(object$scale_info)){
    cont_data <- cont_data_LL %>%
      dplyr::select(object$NN_model$other_vars) %>%
      .GlobalEnv$scale_withPar(object$NN_model$other_vars[!object$NN_model$other_vars=='expo'], 
                               object$scale_info %>% 
                                 arrange(factor(Variable, 
                                                levels = object$NN_model$other_vars[!object$NN_model$other_vars=='expo'])))
    } else {
    cont_data <- cont_data_LL %>% dplyr::select(object$NN_model$other_vars)
  }
  
  # Bind all data into a list of matrices
  train_mat <- list(cont_data %>% data.matrix(),
                    train_cat_data_concat)
  
  # Make predictions on the data with the supplied CANN model
  return(object$NN_model$model %>% predict(train_mat, type = "response", verbose = 0))
}

### ----- Fitting models -----

# Fit FFNN with optimal tuning parameters
NC_opt_FFNN_AUS <- lapply(1:6, function(fold){
  single_run_AE(fold_data = NC_data_AUS[[fold]], 
                flags_list = AUS_NC_NN[[fold]], 
                random_val_split = 0.2,
                autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                cat_vars = cat_AUS,
                output_modelinfo = TRUE)
})
CA_opt_FFNN_AUS <- lapply(1:6, function(fold){
  single_run_AE(fold_data = CA_data_AUS[[fold]], 
                flags_list = AUS_CA_NN[[fold]], 
                random_val_split = 0.2,
                autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                cat_vars = cat_AUS,
                output_modelinfo = TRUE)
})

NC_opt_FFNN_BE <- lapply(1:6, function(fold){
  single_run_AE(fold_data = NC_data_BE[[fold]], 
                flags_list = BE_NC_NN[[fold]], 
                random_val_split = 0.2,
                autoencoder_trained = AE_weights_scaled_BE[[fold]],
                cat_vars = cat_BE,
                output_modelinfo = TRUE)
})
CA_opt_FFNN_BE <- lapply(1:6, function(fold){
  single_run_AE(fold_data = CA_data_BE[[fold]], 
                flags_list = BE_CA_NN[[fold]], 
                random_val_split = 0.2,
                autoencoder_trained = AE_weights_scaled_BE[[fold]],
                cat_vars = cat_BE,
                output_modelinfo = TRUE)
})

NC_opt_FFNN_FR <- lapply(1:6, function(fold){
  single_run_AE(fold_data = NC_data_FR[[fold]], 
                flags_list = FR_NC_NN[[fold]], 
                random_val_split = 0.2,
                autoencoder_trained = AE_weights_scaled_FR[[fold]],
                cat_vars = cat_FR,
                output_modelinfo = TRUE)
})
CA_opt_FFNN_FR <- lapply(1:6, function(fold){
  single_run_AE(fold_data = CA_data_FR[[fold]], 
                flags_list = FR_CA_NN[[fold]], 
                random_val_split = 0.2,
                autoencoder_trained = AE_weights_scaled_FR[[fold]],
                cat_vars = cat_FR,
                output_modelinfo = TRUE)
})

NC_opt_FFNN_NOR <- lapply(1:6, function(fold){
  single_run_AE(fold_data = NC_data_NOR[[fold]], 
                flags_list = NOR_NC_NN[[fold]], 
                random_val_split = 0.2,
                autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                cat_vars = cat_NOR,
                output_modelinfo = TRUE)
})
CA_opt_FFNN_NOR <- lapply(1:6, function(fold){
  single_run_AE(fold_data = CA_data_NOR[[fold]], 
                flags_list = NOR_CA_NN[[fold]], 
                random_val_split = 0.2,
                autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                cat_vars = cat_NOR,
                output_modelinfo = TRUE)
})

# Make FFNN model objects for custom prediction function
NC_FFNNobject_AUS <- lapply(1:6, function(fold){
  list(NN_model = NC_opt_FFNN_AUS[[fold]],
       scale_info = NC_scaleinfo_AUS %>% filter(Testfold == fold),
       problem = 'Frequency')
})
CA_FFNNobject_AUS <- lapply(1:6, function(fold){
  list(NN_model = CA_opt_FFNN_AUS[[fold]],
       scale_info = CA_scaleinfo_AUS %>% filter(Testfold == fold),
       problem = 'Frequency')
})

NC_FFNNobject_BE <- lapply(1:6, function(fold){
  list(NN_model = NC_opt_FFNN_BE[[fold]],
       scale_info = NC_scaleinfo_BE %>% filter(Testfold == fold),
       latlong_conversion = latlong_per_postalcode,
       problem = 'Frequency')
})
CA_FFNNobject_BE <- lapply(1:6, function(fold){
  list(NN_model = CA_opt_FFNN_BE[[fold]],
       scale_info = CA_scaleinfo_BE %>% filter(Testfold == fold),
       latlong_conversion = latlong_per_postalcode,
       problem = 'Severity')
})

NC_FFNNobject_FR <- lapply(1:6, function(fold){
  list(NN_model = NC_opt_FFNN_FR[[fold]],
       scale_info = NC_scaleinfo_FR %>% filter(Testfold == fold),
       problem = 'Frequency')
})
CA_FFNNobject_FR <- lapply(1:6, function(fold){
  list(NN_model = CA_opt_FFNN_FR[[fold]],
       scale_info = CA_scaleinfo_FR %>% filter(Testfold == fold),
       problem = 'Frequency')
})

NC_FFNNobject_NOR <- lapply(1:6, function(fold){
  list(NN_model = NC_opt_FFNN_NOR[[fold]],
       scale_info = NULL,
       problem = 'Frequency')
})
CA_FFNNobject_NOR <- lapply(1:6, function(fold){
  list(NN_model = CA_opt_FFNN_NOR[[fold]],
       scale_info = NULL,
       problem = 'Frequency')
})

# To make the functions a bit more readable, we apply this shortening of notation
sn <- function(vector){
  setNames(vector,vector)
}
# This makes sure that the lapply function returns named objects

### ----- VIP -----

NC_VI_FFNN_AUS <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_AUS,
    variables = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'),
    model = NC_FFNNobject_AUS[[fold]],
    pred_fun = FFNN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_FFNN_AUS, file = 'NC_VI_FFNN_AUS')

CA_VI_FFNN_AUS <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_AUS,
    variables = c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge'),
    model = CA_FFNNobject_AUS[[fold]],
    pred_fun = FFNN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_FFNN_AUS, file = 'CA_VI_FFNN_AUS')

NC_VI_FFNN_BE <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_BE,
    variables = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet','postcode'),
    model = NC_FFNNobject_BE[[fold]],
    pred_fun = FFNN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_FFNN_BE, file = 'NC_VI_FFNN_BE')

CA_VI_FFNN_BE <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_BE,
    variables = c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet','postcode'),
    model = CA_FFNNobject_BE[[fold]],
    pred_fun = FFNN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_FFNN_BE, file = 'CA_VI_FFNN_BE')

NC_VI_FFNN_FR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_FR,
    variables = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"),
    model = NC_FFNNobject_FR[[fold]],
    pred_fun = FFNN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_FFNN_FR, file = 'NC_VI_FFNN_FR')

CA_VI_FFNN_FR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_FR,
    variables = c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"),
    model = CA_FFNNobject_FR[[fold]],
    pred_fun = FFNN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_FFNN_FR, file = 'CA_VI_FFNN_FR')

NC_VI_FFNN_NOR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = NC_data_slice_NOR,
    variables = c('Male', 'Young', 'DistLimit', 'GeoRegion'),
    model = NC_FFNNobject_NOR[[fold]],
    pred_fun = FFNN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(NC_VI_FFNN_NOR, file = 'NC_VI_FFNN_NOR')

CA_VI_FFNN_NOR <- lapply(1:6, function(fold){
  tibble(VI_calculation(
    data = CA_data_slice_NOR,
    variables = c('Male', 'Young', 'DistLimit', 'GeoRegion'),
    model = CA_FFNNobject_NOR[[fold]],
    pred_fun = FFNN_model_predictions_perpoint
  ), Testfold = fold)
}) %>% do.call(rbind,.)
save(CA_VI_FFNN_NOR, file = 'CA_VI_FFNN_NOR')

### ----- PDP -----

NC_PDP_FFNN_AllVARS_AUS <- lapply(sn(c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_FFNNobject_AUS[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_AUS[[var]]) %>% setNames(var),
      data = NC_data_slice_AUS, 
      fun = FFNN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_FFNN_AllVARS_AUS, file = 'NC_PDP_FFNN_AllVARS_AUS')

CA_PDP_FFNN_AllVARS_AUS <- lapply(sn(c('VehValue', 'VehAge', 'VehBody', 'Gender', 'DrivAge')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_FFNNobject_AUS[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_AUS[[var]]) %>% setNames(var),
      data = CA_data_slice_AUS, 
      fun = FFNN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_FFNN_AllVARS_AUS, file = 'CA_PDP_FFNN_AllVARS_AUS')

NC_PDP_FFNN_AllVARS_BE <- lapply(sn(c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet','postcode')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_FFNNobject_BE[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_BE[[var]]) %>% setNames(var),
      data = NC_data_slice_BE, 
      fun = FFNN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_FFNN_AllVARS_BE, file = 'NC_PDP_FFNN_AllVARS_BE')

CA_PDP_FFNN_AllVARS_BE <- lapply(sn(c('coverage', 'ageph', 'sex', 'bm', 'power', 'agec', 'fuel', 'use', 'fleet','postcode')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_FFNNobject_BE[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_BE[[var]]) %>% setNames(var),
      data = CA_data_slice_BE, 
      fun = FFNN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_FFNN_AllVARS_BE, file = 'CA_PDP_FFNN_AllVARS_BE')

NC_PDP_FFNN_AllVARS_FR <- lapply(sn(c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_FFNNobject_FR[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_FR[[var]]) %>% setNames(var),
      data = NC_data_slice_FR, 
      fun = FFNN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_FFNN_AllVARS_FR, file = 'NC_PDP_FFNN_AllVARS_FR')

CA_PDP_FFNN_AllVARS_FR <- lapply(sn(c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_FFNNobject_FR[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_FR[[var]]) %>% setNames(var),
      data = CA_data_slice_FR, 
      fun = FFNN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_FFNN_AllVARS_FR, file = 'CA_PDP_FFNN_AllVARS_FR')

NC_PDP_FFNN_AllVARS_NOR <- lapply(sn(c('Male', 'Young', 'DistLimit', 'GeoRegion')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = NC_FFNNobject_NOR[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_NOR[[var]]) %>% setNames(var),
      data = NC_data_slice_NOR, 
      fun = FFNN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(NC_PDP_FFNN_AllVARS_NOR, file = 'NC_PDP_FFNN_AllVARS_NOR')

CA_PDP_FFNN_AllVARS_NOR <- lapply(sn(c('Male', 'Young', 'DistLimit', 'GeoRegion')), function(var){
  print(paste("Now calculating the PDP for variable",var))
  lapply(1:6, function(fold){
    tibble(maidrr::get_pd(
      mfit = CA_FFNNobject_NOR[[fold]],
      var = var,
      grid = data.frame(all_var_ranges_NOR[[var]]) %>% setNames(var),
      data = CA_data_slice_NOR, 
      fun = FFNN_model_predictions,
    ), Testfold = fold, Variable = var)
  }) %>% do.call(rbind,.)
})
save(CA_PDP_FFNN_AllVARS_NOR, file = 'CA_PDP_FFNN_AllVARS_NOR')

# -----
## ----- Comparison OneHot vs Autoencoder -----

# We choose a set of tuning parameters for both frequency and severity

NC_tuning_set <- tibble(
  optimizer = 'adam',
  batch = 15000,
  activation_h = 'sigmoid',
  dropout = 0.1,
  activation_out = 'exponential',
  epochs = 500,
  loss = 'poisson'
) %>% 
  mutate(hiddennodes = list(rep(18,2))) %>% 
  as.data.table

CA_tuning_set <- tibble(
  optimizer = 'adam',
  batch = 200,
  activation_h = 'sigmoid',
  dropout = 0.001,
  activation_out = 'exponential',
  epochs = 500,
  loss = 'gamma'
) %>% 
  mutate(hiddennodes = list(rep(20,1))) %>% 
  as.data.table

### ----- AUS data -----

# Data prep for onehot encoding
NC_data_OH_AUS <- NC_data_AUS
for(i in 1:6){
  NC_data_OH_AUS[[i]]$trainset$data <- NC_data_OH_AUS[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_AUS) %>% as_tibble()
  NC_data_OH_AUS[[i]]$testset$data <- NC_data_OH_AUS[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_AUS) %>% as_tibble()
}

NC_data_OH_AUS_GBM <- NC_data_AUS_GBM
for(i in 1:6){
  NC_data_OH_AUS_GBM[[i]]$trainset$data <- NC_data_OH_AUS_GBM[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_AUS) %>% as_tibble()
  NC_data_OH_AUS_GBM[[i]]$testset$data <- NC_data_OH_AUS_GBM[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_AUS) %>% as_tibble()
}

CA_data_OH_AUS <- CA_data_AUS
for(i in 1:6){
  CA_data_OH_AUS[[i]]$trainset$data <- CA_data_OH_AUS[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_AUS) %>% as_tibble()
  CA_data_OH_AUS[[i]]$testset$data <- CA_data_OH_AUS[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_AUS) %>% as_tibble()
}

CA_data_OH_AUS_GBM <- CA_data_AUS_GBM
for(i in 1:6){
  CA_data_OH_AUS_GBM[[i]]$trainset$data <- CA_data_OH_AUS_GBM[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_AUS) %>% as_tibble()
  CA_data_OH_AUS_GBM[[i]]$testset$data <- CA_data_OH_AUS_GBM[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_AUS) %>% as_tibble()
}

# FFNN with AE
NC_FFNN_AEcomp_fold1_AUS <- single_run_AE(fold_data = NC_data_AUS[[1]], 
                                          flags_list = NC_tuning_set, 
                                          random_val_split = 0.2,
                                          autoencoder_trained = AE_weights_scaled_AUS[[1]],
                                          cat_vars = cat_AUS)

CA_FFNN_AEcomp_fold1_AUS <- single_run_AE(fold_data = CA_data_AUS[[1]], 
                                          flags_list = CA_tuning_set, 
                                          random_val_split = 0.2,
                                          autoencoder_trained = AE_weights_scaled_AUS[[1]],
                                          cat_vars = cat_AUS)


# CANN model with AE
NC_CANN_GBM_flex_AEcomp_fold1_AUS <- single_CANN_run_AE(fold_data = NC_data_AUS_GBM[[1]], 
                                                        flags_list = NC_tuning_set, 
                                                        random_val_split = 0.2,
                                                        autoencoder_trained = AE_weights_scaled_AUS[[1]],
                                                        cat_vars = cat_AUS,
                                                        trainable_output = TRUE)

CA_CANN_GBM_flex_AEcomp_fold1_AUS <- single_CANN_run_AE(fold_data = CA_data_AUS_GBM[[1]], 
                                                        flags_list = CA_tuning_set, 
                                                        random_val_split = 0.2,
                                                        autoencoder_trained = AE_weights_scaled_AUS[[1]],
                                                        cat_vars = cat_AUS,
                                                        trainable_output = TRUE)

# FFNN with OH
NC_FFNN_OHcomp_fold1_AUS <- single_run(fold_data = NC_data_OH_AUS[[1]], 
                                       flags_list = NC_tuning_set, 
                                       random_val_split = 0.2)

CA_FFNN_OHcomp_fold1_AUS <- single_run(fold_data = CA_data_OH_AUS[[1]], 
                                       flags_list = CA_tuning_set, 
                                       random_val_split = 0.2)

# CANN model with AE
NC_CANN_GBM_flex_OHcomp_fold1_AUS <- single_CANN_run(fold_data = NC_data_OH_AUS_GBM[[1]], 
                                                     flags_list = NC_tuning_set, 
                                                     random_val_split = 0.2,
                                                     trainable_output = TRUE)

CA_CANN_GBM_flex_OHcomp_fold1_AUS <- single_CANN_run(fold_data = CA_data_OH_AUS_GBM[[1]], 
                                                     flags_list = CA_tuning_set, 
                                                     random_val_split = 0.2,
                                                     trainable_output = TRUE)

OHvsAE_comp_AUS <- tibble(Model = c('FFNN', 'CANN GBM flex', 'FFNN', 'CANN GBM flex'), 
                          Embedding = c('One-Hot', 'One-Hot', 'Autoencoder', 'Autoencoder'),
                          Freq_OOS = c(NC_FFNN_OHcomp_fold1_AUS$val_loss, 
                                       NC_CANN_GBM_flex_OHcomp_fold1_AUS$val_loss, 
                                       NC_FFNN_AEcomp_fold1_AUS$val_loss, 
                                       NC_CANN_GBM_flex_AEcomp_fold1_AUS$val_loss),
                          Sev_OOS = c(CA_FFNN_OHcomp_fold1_AUS$val_loss, 
                                      CA_CANN_GBM_flex_OHcomp_fold1_AUS$val_loss, 
                                      CA_FFNN_AEcomp_fold1_AUS$val_loss, 
                                      CA_CANN_GBM_flex_AEcomp_fold1_AUS$val_loss)) %>% 
  mutate(Dataset = 'AUS')

OHvsAE_comp_AUS

save(NC_FFNN_AEcomp_fold1_AUS, CA_FFNN_AEcomp_fold1_AUS, 
     NC_CANN_GBM_flex_AEcomp_fold1_AUS, CA_CANN_GBM_flex_AEcomp_fold1_AUS, 
     NC_FFNN_OHcomp_fold1_AUS, CA_FFNN_OHcomp_fold1_AUS, 
     NC_CANN_GBM_flex_OHcomp_fold1_AUS, CA_CANN_GBM_flex_OHcomp_fold1_AUS, 
     file = 'OHvsAE_compmodels_AUS')

### ----- BE data -----

# Data prep for onehot encoding
NC_data_OH_BE <- NC_data_BE
for(i in 1:6){
  NC_data_OH_BE[[i]]$trainset$data <- NC_data_OH_BE[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_BE) %>% as_tibble()
  NC_data_OH_BE[[i]]$testset$data <- NC_data_OH_BE[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_BE) %>% as_tibble()
}

NC_data_OH_BE_GBM <- NC_data_BE_GBM
for(i in 1:6){
  NC_data_OH_BE_GBM[[i]]$trainset$data <- NC_data_OH_BE_GBM[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_BE) %>% as_tibble()
  NC_data_OH_BE_GBM[[i]]$testset$data <- NC_data_OH_BE_GBM[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_BE) %>% as_tibble()
}

CA_data_OH_BE <- CA_data_BE
for(i in 1:6){
  CA_data_OH_BE[[i]]$trainset$data <- CA_data_OH_BE[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_BE) %>% as_tibble()
  CA_data_OH_BE[[i]]$testset$data <- CA_data_OH_BE[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_BE) %>% as_tibble()
}

CA_data_OH_BE_GBM <- CA_data_BE_GBM
for(i in 1:6){
  CA_data_OH_BE_GBM[[i]]$trainset$data <- CA_data_OH_BE_GBM[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_BE) %>% as_tibble()
  CA_data_OH_BE_GBM[[i]]$testset$data <- CA_data_OH_BE_GBM[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_BE) %>% as_tibble()
}

# FFNN with AE
NC_FFNN_AEcomp_fold1_BE <- single_run_AE(fold_data = NC_data_BE[[1]], 
                                         flags_list = NC_tuning_set, 
                                         random_val_split = 0.2,
                                         autoencoder_trained = AE_weights_scaled_BE[[1]],
                                         cat_vars = cat_BE)

CA_FFNN_AEcomp_fold1_BE <- single_run_AE(fold_data = CA_data_BE[[1]], 
                                         flags_list = CA_tuning_set, 
                                         random_val_split = 0.2,
                                         autoencoder_trained = AE_weights_scaled_BE[[1]],
                                         cat_vars = cat_BE)

# CANN model with AE
NC_CANN_GBM_flex_AEcomp_fold1_BE <- single_CANN_run_AE(fold_data = NC_data_BE_GBM[[1]], 
                                                       flags_list = NC_tuning_set, 
                                                       random_val_split = 0.2,
                                                       autoencoder_trained = AE_weights_scaled_BE[[1]],
                                                       cat_vars = cat_BE,
                                                       trainable_output = TRUE)
NC_CANN_GBM_flex_AEcomp_fold1_BE$val_loss

CA_CANN_GBM_flex_AEcomp_fold1_BE <- single_CANN_run_AE(fold_data = CA_data_BE_GBM[[1]], 
                                                       flags_list = CA_tuning_set, 
                                                       random_val_split = 0.2,
                                                       autoencoder_trained = AE_weights_scaled_BE[[1]],
                                                       cat_vars = cat_BE,
                                                       trainable_output = TRUE)

# FFNN with OH
NC_FFNN_OHcomp_fold1_BE <- single_run(fold_data = NC_data_OH_BE[[1]], 
                                      flags_list = NC_tuning_set, 
                                      random_val_split = 0.2)

CA_FFNN_OHcomp_fold1_BE <- single_run(fold_data = CA_data_OH_BE[[1]], 
                                      flags_list = CA_tuning_set, 
                                      random_val_split = 0.2)

# CANN model with OH
NC_CANN_GBM_flex_OHcomp_fold1_BE <- single_CANN_run(fold_data = NC_data_OH_BE_GBM[[1]], 
                                                    flags_list = NC_tuning_set, 
                                                    random_val_split = 0.2,
                                                    trainable_output = TRUE)
NC_CANN_GBM_flex_OHcomp_fold1_BE$val_loss

CA_CANN_GBM_flex_OHcomp_fold1_BE <- single_CANN_run(fold_data = CA_data_OH_BE_GBM[[1]], 
                                                    flags_list = CA_tuning_set, 
                                                    random_val_split = 0.2,
                                                    trainable_output = TRUE)

OHvsAE_comp_BE <- tibble(Model = c('FFNN', 'CANN GBM flex', 'FFNN', 'CANN GBM flex'), 
                         Embedding = c('One-Hot', 'One-Hot', 'Autoencoder', 'Autoencoder'),
                         Freq_OOS = c(NC_FFNN_OHcomp_fold1_BE$val_loss, 
                                      NC_CANN_GBM_flex_OHcomp_fold1_BE$val_loss, 
                                      NC_FFNN_AEcomp_fold1_BE$val_loss, 
                                      NC_CANN_GBM_flex_AEcomp_fold1_BE$val_loss),
                         Sev_OOS = c(CA_FFNN_OHcomp_fold1_BE$val_loss, 
                                     CA_CANN_GBM_flex_OHcomp_fold1_BE$val_loss, 
                                     CA_FFNN_AEcomp_fold1_BE$val_loss, 
                                     CA_CANN_GBM_flex_AEcomp_fold1_BE$val_loss)) %>% 
  mutate(Dataset = 'BE')

OHvsAE_comp_BE

save(NC_FFNN_AEcomp_fold1_BE, CA_FFNN_AEcomp_fold1_BE, 
     NC_CANN_GBM_flex_AEcomp_fold1_BE, CA_CANN_GBM_flex_AEcomp_fold1_BE, 
     NC_FFNN_OHcomp_fold1_BE, CA_FFNN_OHcomp_fold1_BE, 
     NC_CANN_GBM_flex_OHcomp_fold1_BE, CA_CANN_GBM_flex_OHcomp_fold1_BE, 
     file = 'OHvsAE_compmodels_BE')

### ----- FR data -----

# Data prep for onehot encoding
NC_data_OH_FR <- NC_data_FR
for(i in 1:6){
  NC_data_OH_FR[[i]]$trainset$data <- NC_data_OH_FR[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_FR) %>% as_tibble()
  NC_data_OH_FR[[i]]$testset$data <- NC_data_OH_FR[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_FR) %>% as_tibble()
}

NC_data_OH_FR_GBM <- NC_data_FR_GBM
for(i in 1:6){
  NC_data_OH_FR_GBM[[i]]$trainset$data <- NC_data_OH_FR_GBM[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_FR) %>% as_tibble()
  NC_data_OH_FR_GBM[[i]]$testset$data <- NC_data_OH_FR_GBM[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_FR) %>% as_tibble()
}

CA_data_OH_FR <- CA_data_FR
for(i in 1:6){
  CA_data_OH_FR[[i]]$trainset$data <- CA_data_OH_FR[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_FR) %>% as_tibble()
  CA_data_OH_FR[[i]]$testset$data <- CA_data_OH_FR[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_FR) %>% as_tibble()
}

CA_data_OH_FR_GBM <- CA_data_FR_GBM
for(i in 1:6){
  CA_data_OH_FR_GBM[[i]]$trainset$data <- CA_data_OH_FR_GBM[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_FR) %>% as_tibble()
  CA_data_OH_FR_GBM[[i]]$testset$data <- CA_data_OH_FR_GBM[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_FR) %>% as_tibble()
}

# FFNN with AE
NC_FFNN_AEcomp_fold1_FR <- single_run_AE(fold_data = NC_data_FR[[1]], 
                                         flags_list = NC_tuning_set, 
                                         random_val_split = 0.2,
                                         autoencoder_trained = AE_weights_scaled_FR[[1]],
                                         cat_vars = cat_FR)

CA_FFNN_AEcomp_fold1_FR <- single_run_AE(fold_data = CA_data_FR[[1]], 
                                         flags_list = CA_tuning_set, 
                                         random_val_split = 0.2,
                                         autoencoder_trained = AE_weights_scaled_FR[[1]],
                                         cat_vars = cat_FR)


# CANN model with AE
NC_CANN_GBM_flex_AEcomp_fold1_FR <- single_CANN_run_AE(fold_data = NC_data_FR_GBM[[1]], 
                                                       flags_list = NC_tuning_set, 
                                                       random_val_split = 0.2,
                                                       autoencoder_trained = AE_weights_scaled_FR[[1]],
                                                       cat_vars = cat_FR,
                                                       trainable_output = TRUE)

CA_CANN_GBM_flex_AEcomp_fold1_FR <- single_CANN_run_AE(fold_data = CA_data_FR_GBM[[1]], 
                                                       flags_list = CA_tuning_set, 
                                                       random_val_split = 0.2,
                                                       autoencoder_trained = AE_weights_scaled_FR[[1]],
                                                       cat_vars = cat_FR,
                                                       trainable_output = TRUE)

# FFNN with OH
NC_FFNN_OHcomp_fold1_FR <- single_run(fold_data = NC_data_OH_FR[[1]], 
                                      flags_list = NC_tuning_set, 
                                      random_val_split = 0.2)

CA_FFNN_OHcomp_fold1_FR <- single_run(fold_data = CA_data_OH_FR[[1]], 
                                      flags_list = CA_tuning_set, 
                                      random_val_split = 0.2)

# CANN model with AE
NC_CANN_GBM_flex_OHcomp_fold1_FR <- single_CANN_run(fold_data = NC_data_OH_FR_GBM[[1]], 
                                                    flags_list = NC_tuning_set, 
                                                    random_val_split = 0.2,
                                                    trainable_output = TRUE)

CA_CANN_GBM_flex_OHcomp_fold1_FR <- single_CANN_run(fold_data = CA_data_OH_FR_GBM[[1]], 
                                                    flags_list = CA_tuning_set, 
                                                    random_val_split = 0.2,
                                                    trainable_output = TRUE)

OHvsAE_comp_FR <- tibble(Model = c('FFNN', 'CANN GBM flex', 'FFNN', 'CANN GBM flex'), 
                         Embedding = c('One-Hot', 'One-Hot', 'Autoencoder', 'Autoencoder'),
                         Freq_OOS = c(NC_FFNN_OHcomp_fold1_FR$val_loss, 
                                      NC_CANN_GBM_flex_OHcomp_fold1_FR$val_loss, 
                                      NC_FFNN_AEcomp_fold1_FR$val_loss, 
                                      NC_CANN_GBM_flex_AEcomp_fold1_FR$val_loss),
                         Sev_OOS = c(CA_FFNN_OHcomp_fold1_FR$val_loss, 
                                     CA_CANN_GBM_flex_OHcomp_fold1_FR$val_loss, 
                                     CA_FFNN_AEcomp_fold1_FR$val_loss, 
                                     CA_CANN_GBM_flex_AEcomp_fold1_FR$val_loss)) %>% 
  mutate(Dataset = 'FR')

OHvsAE_comp_FR

save(NC_FFNN_AEcomp_fold1_FR, CA_FFNN_AEcomp_fold1_FR, 
     NC_CANN_GBM_flex_AEcomp_fold1_FR, CA_CANN_GBM_flex_AEcomp_fold1_FR, 
     NC_FFNN_OHcomp_fold1_FR, CA_FFNN_OHcomp_fold1_FR, 
     NC_CANN_GBM_flex_OHcomp_fold1_FR, CA_CANN_GBM_flex_OHcomp_fold1_FR, 
     file = 'OHvsAE_compmodels_FR')

### ----- NOR data -----

# Data prep for onehot encoding
NC_data_OH_NOR <- NC_data_NOR
for(i in 1:6){
  NC_data_OH_NOR[[i]]$trainset$data <- NC_data_OH_NOR[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_NOR) %>% as_tibble()
  NC_data_OH_NOR[[i]]$testset$data <- NC_data_OH_NOR[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_NOR) %>% as_tibble()
}

NC_data_OH_NOR_GBM <- NC_data_NOR_GBM
for(i in 1:6){
  NC_data_OH_NOR_GBM[[i]]$trainset$data <- NC_data_OH_NOR_GBM[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_NOR) %>% as_tibble()
  NC_data_OH_NOR_GBM[[i]]$testset$data <- NC_data_OH_NOR_GBM[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_NOR) %>% as_tibble()
}

CA_data_OH_NOR <- CA_data_NOR
for(i in 1:6){
  CA_data_OH_NOR[[i]]$trainset$data <- CA_data_OH_NOR[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_NOR) %>% as_tibble()
  CA_data_OH_NOR[[i]]$testset$data <- CA_data_OH_NOR[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_NOR) %>% as_tibble()
}

CA_data_OH_NOR_GBM <- CA_data_NOR_GBM
for(i in 1:6){
  CA_data_OH_NOR_GBM[[i]]$trainset$data <- CA_data_OH_NOR_GBM[[i]]$trainset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_NOR) %>% as_tibble()
  CA_data_OH_NOR_GBM[[i]]$testset$data <- CA_data_OH_NOR_GBM[[i]]$testset$data %>% 
    data.table::as.data.table() %>% 
    mltools::one_hot(cols=cat_NOR) %>% as_tibble()
}

# FFNN with AE
NC_FFNN_AEcomp_fold1_NOR <- single_run_AE(fold_data = NC_data_NOR[[1]], 
                                          flags_list = NC_tuning_set, 
                                          random_val_split = 0.2,
                                          autoencoder_trained = AE_weights_scaled_NOR[[1]],
                                          cat_vars = cat_NOR)

CA_FFNN_AEcomp_fold1_NOR <- single_run_AE(fold_data = CA_data_NOR[[1]], 
                                          flags_list = CA_tuning_set, 
                                          random_val_split = 0.2,
                                          autoencoder_trained = AE_weights_scaled_NOR[[1]],
                                          cat_vars = cat_NOR)


# CANN model with AE
NC_CANN_GBM_flex_AEcomp_fold1_NOR <- single_CANN_run_AE(fold_data = NC_data_NOR_GBM[[1]], 
                                                        flags_list = NC_tuning_set, 
                                                        random_val_split = 0.2,
                                                        autoencoder_trained = AE_weights_scaled_NOR[[1]],
                                                        cat_vars = cat_NOR,
                                                        trainable_output = TRUE)

CA_CANN_GBM_flex_AEcomp_fold1_NOR <- single_CANN_run_AE(fold_data = CA_data_NOR_GBM[[1]], 
                                                        flags_list = CA_tuning_set, 
                                                        random_val_split = 0.2,
                                                        autoencoder_trained = AE_weights_scaled_NOR[[1]],
                                                        cat_vars = cat_NOR,
                                                        trainable_output = TRUE)

# FFNN with OH
NC_FFNN_OHcomp_fold1_NOR <- single_run(fold_data = NC_data_OH_NOR[[1]], 
                                       flags_list = NC_tuning_set, 
                                       random_val_split = 0.2)

CA_FFNN_OHcomp_fold1_NOR <- single_run(fold_data = CA_data_OH_NOR[[1]], 
                                       flags_list = CA_tuning_set, 
                                       random_val_split = 0.2)

# CANN model with AE
NC_CANN_GBM_flex_OHcomp_fold1_NOR <- single_CANN_run(fold_data = NC_data_OH_NOR_GBM[[1]], 
                                                     flags_list = NC_tuning_set, 
                                                     random_val_split = 0.2,
                                                     trainable_output = TRUE)

CA_CANN_GBM_flex_OHcomp_fold1_NOR <- single_CANN_run(fold_data = CA_data_OH_NOR_GBM[[1]], 
                                                     flags_list = CA_tuning_set, 
                                                     random_val_split = 0.2,
                                                     trainable_output = TRUE)

OHvsAE_comp_NOR <- tibble(Model = c('FFNN', 'CANN GBM flex', 'FFNN', 'CANN GBM flex'), 
                          Embedding = c('One-Hot', 'One-Hot', 'Autoencoder', 'Autoencoder'),
                          Freq_OOS = c(NC_FFNN_OHcomp_fold1_NOR$val_loss, 
                                       NC_CANN_GBM_flex_OHcomp_fold1_NOR$val_loss, 
                                       NC_FFNN_AEcomp_fold1_NOR$val_loss, 
                                       NC_CANN_GBM_flex_AEcomp_fold1_NOR$val_loss),
                          Sev_OOS = c(CA_FFNN_OHcomp_fold1_NOR$val_loss, 
                                      CA_CANN_GBM_flex_OHcomp_fold1_NOR$val_loss, 
                                      CA_FFNN_AEcomp_fold1_NOR$val_loss, 
                                      CA_CANN_GBM_flex_AEcomp_fold1_NOR$val_loss)) %>% 
  mutate(Dataset = 'NOR')

OHvsAE_comp_NOR

save(NC_FFNN_AEcomp_fold1_NOR, CA_FFNN_AEcomp_fold1_NOR, 
     NC_CANN_GBM_flex_AEcomp_fold1_NOR, CA_CANN_GBM_flex_AEcomp_fold1_NOR, 
     NC_FFNN_OHcomp_fold1_NOR, CA_FFNN_OHcomp_fold1_NOR, 
     NC_CANN_GBM_flex_OHcomp_fold1_NOR, CA_CANN_GBM_flex_OHcomp_fold1_NOR, 
     file = 'OHvsAE_compmodels_NOR')

### ----- Combine Onehot versus autoencoder results -----

OHvsAE_comp_allDataSets <- rbind(
  OHvsAE_comp_AUS, OHvsAE_comp_BE, OHvsAE_comp_FR, OHvsAE_comp_NOR
)

save(OHvsAE_comp_allDataSets, file = 'OHvsAE_comp_allDataSets')

# -----
# ----- PLOT MAKING -----

# Here we read in the results from the interpretation calculation
# Plots are created for each wanted item in the paper

## ----- Variable importance plot -----

### ----- Read in VIP data -----

load('NC_VI_CANN_GBM_flex_AUS')
load('NC_VI_CANN_GBM_flex_BE')
load('NC_VI_CANN_GBM_flex_FR')
load('NC_VI_CANN_GBM_flex_NOR')

load('NC_VI_GBM_AUS')
load('NC_VI_GBM_BE')
load('NC_VI_GBM_FR')
load('NC_VI_GBM_NOR')

load('NC_VI_FFNN_AUS')
load('NC_VI_FFNN_BE')
load('NC_VI_FFNN_FR')
load('NC_VI_FFNN_NOR')

load('CA_VI_CANN_GBM_flex_AUS')
load('CA_VI_CANN_GBM_flex_BE')
load('CA_VI_CANN_GBM_flex_FR')
load('CA_VI_CANN_GBM_flex_NOR')

load('CA_VI_GBM_AUS')
load('CA_VI_GBM_BE')
load('CA_VI_GBM_FR')
load('CA_VI_GBM_NOR')

load('CA_VI_FFNN_AUS')
load('CA_VI_FFNN_BE')
load('CA_VI_FFNN_FR')
load('CA_VI_FFNN_NOR')

### ----- Comparison GBM with CANN GBM flex plots ----
#### ----- AUS -----

AUS_var_order <- NC_VI_CANN_GBM_flex_AUS %>% filter(Testfold == 1) %>% arrange(scaled_VI) %>% pull(Variable)
AUS_var_order_x <- AUS_var_order %>% as_tibble_col(column_name = 'Variable') %>% left_join(vars_with_label_AUS %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'))

data_VIP_CANNvsGBM_AUS <- bind_rows(
  NC_VI_CANN_GBM_flex_AUS %>% filter(Testfold == 1) %>% mutate(Model = "CANN GBM flex", Problem = 'Freq'),
  NC_VI_GBM_AUS %>% filter(Testfold == 1) %>% mutate(Model = "GBM", Problem = 'Freq'),
  NC_VI_FFNN_AUS  %>% filter(Testfold == 1) %>% mutate(Model = "FFNN", Problem = 'Freq'),
  CA_VI_CANN_GBM_flex_AUS %>% filter(Testfold == 1) %>% mutate(Model = "CANN GBM flex", Problem = 'Sev'),
  CA_VI_GBM_AUS %>% filter(Testfold == 1) %>% mutate(Model = "GBM", Problem = 'Sev'),
  CA_VI_FFNN_AUS  %>% filter(Testfold == 1) %>% mutate(Model = "FFNN", Problem = 'Sev')
) %>% 
  mutate(Variable = fct_relevel(Variable, AUS_var_order_x$Variable)) %>% 
  left_join(AUS_var_order_x, by = 'Variable') %>% 
  mutate(xlabels = fct_relevel(xlabels, AUS_var_order_x$xlabels)) %>% 
  mutate(Model = fct_relevel(Model, c("CANN GBM flex", "FFNN", "GBM")))

NC_VIP_CANNvsGBM_AUS <- data_VIP_CANNvsGBM_AUS %>% #filter(Model != 'FFNN') %>% 
  #left_join(vars_with_label_AUS %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'), by = 'Variable') %>% 
  filter(Problem == 'Freq') %>% 
  ggplot(aes(y = xlabels)) +  
  geom_col(aes(x = scaled_VI, fill = Model), position="dodge", alpha = 0.8) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Frequency modelling')

CA_VIP_CANNvsGBM_AUS <- data_VIP_CANNvsGBM_AUS %>% #filter(Model != 'FFNN') %>% 
  #left_join(vars_with_label_AUS %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'), by = 'Variable') %>% 
  filter(Problem == 'Sev') %>% 
  ggplot(aes(y = xlabels)) +  
  geom_col(aes(x = scaled_VI, fill = Model), position="dodge", alpha = 0.8) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Severity modelling')

#### ----- BE -----

BE_var_order <- NC_VI_CANN_GBM_flex_BE %>% filter(Testfold == 1) %>% arrange(scaled_VI) %>% pull(Variable)
BE_var_order_x <- BE_var_order %>% as_tibble_col(column_name = 'Variable') %>% left_join(vars_with_label_BE %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'))

data_VIP_CANNvsGBM_BE <- bind_rows(
  NC_VI_CANN_GBM_flex_BE %>% filter(Testfold == 1) %>% mutate(Model = "CANN GBM flex", Problem = 'Freq'),
  NC_VI_GBM_BE %>% filter(Testfold == 1) %>% mutate(Model = "GBM", Problem = 'Freq'),
  NC_VI_FFNN_BE  %>% filter(Testfold == 1) %>% mutate(Model = "FFNN", Problem = 'Freq'),
  CA_VI_CANN_GBM_flex_BE %>% filter(Testfold == 1) %>% mutate(Model = "CANN GBM flex", Problem = 'Sev'),
  CA_VI_GBM_BE %>% filter(Testfold == 1) %>% mutate(Model = "GBM", Problem = 'Sev'),
  CA_VI_FFNN_BE  %>% filter(Testfold == 1) %>% mutate(Model = "FFNN", Problem = 'Sev')
) %>% 
  mutate(Variable = fct_relevel(Variable, BE_var_order_x$Variable)) %>% 
  left_join(BE_var_order_x, by = 'Variable') %>% 
  mutate(xlabels = fct_relevel(xlabels, BE_var_order_x$xlabels)) %>% 
  mutate(Model = fct_relevel(Model, c("CANN GBM flex", "FFNN", "GBM")))

NC_VIP_CANNvsGBM_BE <- data_VIP_CANNvsGBM_BE %>% #filter(Model != 'FFNN') %>% 
  #left_join(vars_with_label_BE %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'), by = 'Variable') %>% 
  filter(Problem == 'Freq') %>% 
  ggplot(aes(y = xlabels)) +  
  geom_col(aes(x = scaled_VI, fill = Model), position="dodge", alpha = 0.8) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Frequency modelling')

CA_VIP_CANNvsGBM_BE <- data_VIP_CANNvsGBM_BE %>% #filter(Model != 'FFNN') %>% 
  #left_join(vars_with_label_BE %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'), by = 'Variable') %>% 
  filter(Problem == 'Sev') %>% 
  ggplot(aes(y = xlabels)) +  
  geom_col(aes(x = scaled_VI, fill = Model), position="dodge", alpha = 0.8) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Severity modelling')


#### ----- FR -----

FR_var_order <- NC_VI_CANN_GBM_flex_FR %>% filter(Testfold == 1) %>% arrange(scaled_VI) %>% pull(Variable)
FR_var_order_x <- FR_var_order %>% as_tibble_col(column_name = 'Variable') %>% left_join(vars_with_label_FR %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'))

data_VIP_CANNvsGBM_FR <- bind_rows(
  NC_VI_CANN_GBM_flex_FR %>% filter(Testfold == 1) %>% mutate(Model = "CANN GBM flex", Problem = 'Freq'),
  NC_VI_GBM_FR %>% filter(Testfold == 1) %>% mutate(Model = "GBM", Problem = 'Freq'),
  NC_VI_FFNN_FR  %>% filter(Testfold == 1) %>% mutate(Model = "FFNN", Problem = 'Freq'),
  CA_VI_CANN_GBM_flex_FR %>% filter(Testfold == 1) %>% mutate(Model = "CANN GBM flex", Problem = 'Sev'),
  CA_VI_GBM_FR %>% filter(Testfold == 1) %>% mutate(Model = "GBM", Problem = 'Sev'),
  CA_VI_FFNN_FR  %>% filter(Testfold == 1) %>% mutate(Model = "FFNN", Problem = 'Sev')
) %>% 
  mutate(Variable = fct_relevel(Variable, FR_var_order_x$Variable)) %>% 
  left_join(FR_var_order_x, by = 'Variable') %>% 
  mutate(xlabels = fct_relevel(xlabels, FR_var_order_x$xlabels)) %>% 
  mutate(Model = fct_relevel(Model, c("CANN GBM flex", "FFNN", "GBM")))

NC_VIP_CANNvsGBM_FR <- data_VIP_CANNvsGBM_FR %>% #filter(Model != 'FFNN') %>% 
  filter(Problem == 'Freq') %>% 
  ggplot(aes(y = xlabels)) +  
  geom_col(aes(x = scaled_VI, fill = Model), position="dodge", alpha = 0.8) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Frequency modelling')

CA_VIP_CANNvsGBM_FR <- data_VIP_CANNvsGBM_FR %>% #filter(Model != 'FFNN') %>% 
  filter(Problem == 'Sev') %>% 
  ggplot(aes(y = xlabels)) +  
  geom_col(aes(x = scaled_VI, fill = Model), position="dodge", alpha = 0.8) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Severity modelling')

#### ----- NOR -----

NOR_var_order <- NC_VI_CANN_GBM_flex_NOR %>% filter(Testfold == 1) %>% arrange(scaled_VI) %>% pull(Variable)
NOR_var_order_x <- NOR_var_order %>% as_tibble_col(column_name = 'Variable') %>% left_join(vars_with_label_NOR %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'))

data_VIP_CANNvsGBM_NOR <- bind_rows(
  NC_VI_CANN_GBM_flex_NOR %>% filter(Testfold == 1) %>% mutate(Model = "CANN GBM flex", Problem = 'Freq'),
  NC_VI_GBM_NOR %>% filter(Testfold == 1) %>% mutate(Model = "GBM", Problem = 'Freq'),
  NC_VI_FFNN_NOR  %>% filter(Testfold == 1) %>% mutate(Model = "FFNN", Problem = 'Freq'),
  CA_VI_CANN_GBM_flex_NOR %>% filter(Testfold == 1) %>% mutate(Model = "CANN GBM flex", Problem = 'Sev'),
  CA_VI_GBM_NOR %>% filter(Testfold == 1) %>% mutate(Model = "GBM", Problem = 'Sev'),
  CA_VI_FFNN_NOR  %>% filter(Testfold == 1) %>% mutate(Model = "FFNN", Problem = 'Sev')
) %>% 
  mutate(Variable = fct_relevel(Variable, NOR_var_order_x$Variable)) %>% 
  left_join(NOR_var_order_x, by = 'Variable') %>% 
  mutate(xlabels = fct_relevel(xlabels, NOR_var_order_x$xlabels)) %>% 
  mutate(Model = fct_relevel(Model, c("CANN GBM flex", "FFNN", "GBM")))

NC_VIP_CANNvsGBM_NOR <- data_VIP_CANNvsGBM_NOR %>% #filter(Model != 'FFNN') %>% 
  filter(Problem == 'Freq') %>% 
  ggplot(aes(y = xlabels)) +  
  geom_col(aes(x = scaled_VI, fill = Model), position="dodge", alpha = 0.8) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Frequency modelling')

CA_VIP_CANNvsGBM_NOR <- data_VIP_CANNvsGBM_NOR %>% #filter(Model != 'FFNN') %>% 
  filter(Problem == 'Sev') %>% 
  ggplot(aes(y = xlabels)) +  
  geom_col(aes(x = scaled_VI, fill = Model), position="dodge", alpha = 0.8) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Severity modelling')


#### ----- Model comparison VIP -----

# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(NC_VIP_CANNvsGBM_AUS + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Australia") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.46)), 
                            NC_VIP_CANNvsGBM_BE + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Belgium") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.46)),
                            NC_VIP_CANNvsGBM_FR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "France") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.45)), 
                            NC_VIP_CANNvsGBM_NOR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Norway") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.45)),  
                            CA_VIP_CANNvsGBM_AUS + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "") + 
                              theme(plot.margin = unit(margin_set, "cm")), 
                            CA_VIP_CANNvsGBM_BE + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "") + 
                              theme(plot.margin = unit(margin_set, "cm")),
                            CA_VIP_CANNvsGBM_FR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "") + 
                              theme(plot.margin = unit(margin_set, "cm")),
                            CA_VIP_CANNvsGBM_NOR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "") +  
                              theme(plot.margin = unit(margin_set, "cm")), 
                            align = "hv")

# Make a grid of all plots, with country names
allplotsgrid <- plot_grid(
  ggdraw() + draw_label("Frequency", angle = 90, size = 14),
  allplotslist[[1]], allplotslist[[2]], allplotslist[[3]], allplotslist[[4]],
  ggdraw() + draw_label("Severity", angle = 90, size = 14), 
  allplotslist[[5]], allplotslist[[6]], allplotslist[[7]], allplotslist[[8]],
  ncol = 5, rel_widths = c(0.015,0.23, 0.23, 0.23, 0.23)
)

# Final plot grid, with legend
final_VIP_GBMvsCANN_plot <- plot_grid(
  allplotsgrid,
  get_legend(NC_VIP_CANNvsGBM_AUS),
  ncol = 1, rel_heights = c(0.9,0.1)
)

#final_VIP_GBMvsCANN_plot

# Save plot as PDF
ggsave("final_VIP_GBMvsCANN_plot.pdf",
       final_VIP_GBMvsCANN_plot, 
       device = cairo_pdf,
       width = 36,
       height = 18,
       scale = 1.2,
       units = "cm")

### ----- CANN VIP All data sets-----

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

# -----
## ----- Partial dependency plot -----
### ----- Read in PDP data -----

load('NC_PDP_GBM_AllVARS_AUS')
load('CA_PDP_GBM_AllVARS_AUS')
load('NC_PDP_FFNN_AllVARS_AUS')
load('CA_PDP_FFNN_AllVARS_AUS')
load('NC_PDP_CANN_GBM_flex_AllVARS_AUS')
load('CA_PDP_CANN_GBM_flex_AllVARS_AUS')

load('NC_PDP_GBM_AllVARS_BE')
load('CA_PDP_GBM_AllVARS_BE')
load('NC_PDP_FFNN_AllVARS_BE')
load('CA_PDP_FFNN_AllVARS_BE')
load('NC_PDP_CANN_GBM_flex_AllVARS_BE')
load('CA_PDP_CANN_GBM_flex_AllVARS_BE')

load('NC_PDP_GBM_AllVARS_FR')
load('CA_PDP_GBM_AllVARS_FR')
load('NC_PDP_FFNN_AllVARS_FR')
load('CA_PDP_FFNN_AllVARS_FR')
load('NC_PDP_CANN_GBM_flex_AllVARS_FR')
load('CA_PDP_CANN_GBM_flex_AllVARS_FR')

load('NC_PDP_GBM_AllVARS_NOR')
load('CA_PDP_GBM_AllVARS_NOR')
load('NC_PDP_FFNN_AllVARS_NOR')
load('CA_PDP_FFNN_AllVARS_NOR')
load('NC_PDP_CANN_GBM_flex_AllVARS_NOR')
load('CA_PDP_CANN_GBM_flex_AllVARS_NOR')

### ----- PDP AgePh Frequency-----

NC_PDP_plot_ageph_ModelComp_AUS <- rbind(
  NC_PDP_GBM_AllVARS_AUS$DrivAge %>% mutate(Model = 'GBM', Data = 'Australia'),
  NC_PDP_CANN_GBM_flex_AllVARS_AUS$DrivAge %>% mutate(Model = 'CANN GBM flex', Data = 'Australia'),
  NC_PDP_FFNN_AllVARS_AUS$DrivAge %>% mutate(Model = 'FFNN', Data = 'Australia')) %>% 
  filter(Testfold == 1) %>% 
  mutate(Model = fct_relevel(Model, c("GBM", "FFNN", "CANN GBM flex"))) %>% 
  ggplot(aes(x = x, y = y)) +  
  #geom_line(aes(color = Model), linewidth = 1.4) + 
  geom_col(aes(color = Model, fill = Model), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.8)) +  
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab(vars_with_label_AUS %>% filter(Variable == 'DrivAge') %>% pull(xlabels)) + ylab("Average predicted claim frequency") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=12), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM", "FFNN","CANN GBM flex"),
                     labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                     values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Frequency modelling')

NC_PDP_plot_ageph_ModelComp_BE <- rbind(
  NC_PDP_GBM_AllVARS_BE$ageph %>% mutate(Model = 'GBM', Data = 'Belgium'),
  NC_PDP_CANN_GBM_flex_AllVARS_BE$ageph %>% mutate(Model = 'CANN GBM flex', Data = 'Belgium'),
  NC_PDP_FFNN_AllVARS_BE$ageph %>% mutate(Model = 'FFNN', Data = 'Belgium')) %>% 
  filter(Testfold == 1) %>% 
  mutate(Model = fct_relevel(Model, c("GBM", "FFNN", "CANN GBM flex"))) %>% 
  ggplot(aes(x = x, y = y)) +  
  geom_line(aes(color = Model), linewidth = 1.4) + 
  #geom_col(aes(color = Model, fill = Model), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.8)) +  
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab(vars_with_label_BE %>% filter(Variable == 'ageph') %>% pull(xlabels)) + ylab("Average predicted claim frequency") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=12), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM", "FFNN","CANN GBM flex"),
                     labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                     values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Frequency modelling')

NC_PDP_plot_ageph_ModelComp_FR <- rbind(
  NC_PDP_GBM_AllVARS_FR$DrivAge %>% mutate(Model = 'GBM', Data = 'France'),
  NC_PDP_CANN_GBM_flex_AllVARS_FR$DrivAge %>% mutate(Model = 'CANN GBM flex', Data = 'France'),
  NC_PDP_FFNN_AllVARS_FR$DrivAge %>% mutate(Model = 'FFNN', Data = 'France')) %>% 
  filter(Testfold == 1) %>% 
  mutate(Model = fct_relevel(Model, c("GBM", "FFNN", "CANN GBM flex"))) %>% 
  ggplot(aes(x = x, y = y)) +  
  #geom_line(aes(color = Model), linewidth = 1.4) + 
  geom_col(aes(color = Model, fill = Model), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.8)) +  
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab(vars_with_label_FR %>% filter(Variable == 'DrivAge') %>% pull(xlabels)) + ylab("Average predicted claim frequency") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=12), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM", "FFNN","CANN GBM flex"),
                     labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                     values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_x_discrete(labels = c('< 21','[21,26[','[26,30[','[30,40[','[40,50[','[50,70[','\u2265 70')) +
  labs(subtitle = 'Frequency modelling')

NC_PDP_plot_ageph_ModelComp_NOR <- rbind(
  NC_PDP_GBM_AllVARS_NOR$Young %>% mutate(Model = 'GBM', Data = 'Norway'),
  NC_PDP_CANN_GBM_flex_AllVARS_NOR$Young %>% mutate(Model = 'CANN GBM flex', Data = 'Norway'),
  NC_PDP_FFNN_AllVARS_NOR$Young %>% mutate(Model = 'FFNN', Data = 'Norway')) %>% 
  filter(Testfold == 1) %>% 
  mutate(Model = fct_relevel(Model, c("GBM", "FFNN", "CANN GBM flex"))) %>% 
  mutate(x = fct_relevel(x,c('Yes','No'))) %>% 
  ggplot(aes(x = x, y = y)) +  
  #geom_line(aes(color = Model), linewidth = 1.4) + 
  geom_col(aes(color = Model, fill = Model), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.8)) +  
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab(vars_with_label_NOR %>% filter(Variable == 'Young') %>% pull(xlabels)) + ylab("Average predicted claim frequency") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=12), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM", "FFNN","CANN GBM flex"),
                     labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                     values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_x_discrete(labels = c('Young','Old'))
labs(subtitle = 'Frequency modelling')

# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(NC_PDP_plot_ageph_ModelComp_AUS + theme(legend.position = "none") + 
                              labs(x = NULL, subtitle = "Australia") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.20),
                                    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)), 
                            NC_PDP_plot_ageph_ModelComp_BE + theme(legend.position = "none") + 
                              labs(y = NULL, subtitle = "Belgium") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.20),
                                    axis.title.x = element_text(margin = margin(t = -60))),
                            NC_PDP_plot_ageph_ModelComp_FR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "France") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.19),
                                    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)), 
                            NC_PDP_plot_ageph_ModelComp_NOR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Norway") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.19),
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
  get_legend(NC_PDP_plot_ageph_ModelComp_AUS),
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


### ----- PDP AgePh Severity -----

CA_PDP_plot_ageph_ModelComp_AUS <- rbind(
  CA_PDP_GBM_AllVARS_AUS$DrivAge %>% mutate(Model = 'GBM', Data = 'Australia'),
  CA_PDP_CANN_GBM_flex_AllVARS_AUS$DrivAge %>% mutate(Model = 'CANN GBM flex', Data = 'Australia'),
  CA_PDP_FFNN_AllVARS_AUS$DrivAge %>% mutate(Model = 'FFNN', Data = 'Australia')) %>% 
  filter(Testfold == 1) %>% 
  mutate(Model = fct_relevel(Model, c("GBM", "FFNN", "CANN GBM flex"))) %>% 
  ggplot(aes(x = x, y = y)) +  
  #geom_line(aes(color = Model), linewidth = 1.4) + 
  geom_col(aes(color = Model, fill = Model), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.8)) +  
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab(vars_with_label_AUS %>% filter(Variable == 'DrivAge') %>% pull(xlabels)) + ylab("Average predicted claim severity") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=12), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM", "FFNN","CANN GBM flex"),
                     labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                     values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Severity modelling')

CA_PDP_plot_ageph_ModelComp_BE <- rbind(
  CA_PDP_GBM_AllVARS_BE$ageph %>% mutate(Model = 'GBM', Data = 'Belgium'),
  CA_PDP_CANN_GBM_flex_AllVARS_BE$ageph %>% mutate(Model = 'CANN GBM flex', Data = 'Belgium'),
  CA_PDP_FFNN_AllVARS_BE$ageph %>% mutate(Model = 'FFNN', Data = 'Belgium')) %>% 
  filter(Testfold == 1) %>% 
  mutate(Model = fct_relevel(Model, c("GBM", "FFNN", "CANN GBM flex"))) %>% 
  ggplot(aes(x = x, y = y)) +  
  geom_line(aes(color = Model), linewidth = 1.4) + 
  #geom_col(aes(color = Model, fill = Model), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.8)) +  
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab(vars_with_label_BE %>% filter(Variable == 'ageph') %>% pull(xlabels)) + ylab("Average predicted claim severity") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=12), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM", "FFNN","CANN GBM flex"),
                     labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                     values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Severity modelling')

CA_PDP_plot_ageph_ModelComp_FR <- rbind(
  CA_PDP_GBM_AllVARS_FR$DrivAge %>% mutate(Model = 'GBM', Data = 'France'),
  CA_PDP_CANN_GBM_flex_AllVARS_FR$DrivAge %>% mutate(Model = 'CANN GBM flex', Data = 'France'),
  CA_PDP_FFNN_AllVARS_FR$DrivAge %>% mutate(Model = 'FFNN', Data = 'France')) %>% 
  filter(Testfold == 1) %>% 
  mutate(Model = fct_relevel(Model, c("GBM", "FFNN", "CANN GBM flex"))) %>% 
  ggplot(aes(x = x, y = y)) +  
  #geom_line(aes(color = Model), linewidth = 1.4) + 
  geom_col(aes(color = Model, fill = Model), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.8)) +  
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab(vars_with_label_FR %>% filter(Variable == 'DrivAge') %>% pull(xlabels)) + ylab("Average predicted claim severity") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=12), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM", "FFNN","CANN GBM flex"),
                     labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                     values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_x_discrete(labels = c('< 21','[21,26[','[26,30[','[30,40[','[40,50[','[50,70[','\u2265 70')) +
  labs(subtitle = 'Severity modelling')

CA_PDP_plot_ageph_ModelComp_NOR <- rbind(
  CA_PDP_GBM_AllVARS_NOR$Young %>% mutate(Model = 'GBM', Data = 'Norway'),
  CA_PDP_CANN_GBM_flex_AllVARS_NOR$Young %>% mutate(Model = 'CANN GBM flex', Data = 'Norway'),
  CA_PDP_FFNN_AllVARS_NOR$Young %>% mutate(Model = 'FFNN', Data = 'Norway')) %>% 
  filter(Testfold == 1) %>% 
  mutate(Model = fct_relevel(Model, c("GBM", "FFNN", "CANN GBM flex"))) %>% 
  mutate(x = fct_relevel(x,c('Yes','No'))) %>% 
  ggplot(aes(x = x, y = y)) +  
  #geom_line(aes(color = Model), linewidth = 1.4) + 
  geom_col(aes(color = Model, fill = Model), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.8)) +  
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab(vars_with_label_NOR %>% filter(Variable == 'Young') %>% pull(xlabels)) + ylab("Average predicted claim severity") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=12), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM", "FFNN","CANN GBM flex"),
                     labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                     values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_x_discrete(labels = c('Young','Old'))
labs(subtitle = 'Severity modelling')

# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(CA_PDP_plot_ageph_ModelComp_AUS + theme(legend.position = "none") + 
                              labs(x = NULL, subtitle = "Australia") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.20),
                                    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)), 
                            CA_PDP_plot_ageph_ModelComp_BE + theme(legend.position = "none") + 
                              labs(y = NULL, subtitle = "Belgium") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.20),
                                    axis.title.x = element_text(margin = margin(t = -60))),
                            CA_PDP_plot_ageph_ModelComp_FR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "France") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.19),
                                    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)), 
                            CA_PDP_plot_ageph_ModelComp_NOR + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Norway") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.19),
                                    axis.title.x = element_text(margin = margin(t = -60))),
                            align = "hv")

# Make a grid of all plots, with country names
allplotsgrid <- plot_grid(#ggdraw() + draw_label(''),ggdraw() + draw_label("Out-of-sample Poisson Deviance", size = 12),ggdraw() + draw_label("Out-of-sample gamma Deviance", size = 12),
  allplotslist[[1]], allplotslist[[2]], allplotslist[[3]], allplotslist[[4]],
  ncol = 4, rel_widths = c(0.25,0.25,0.25,0.25)#, rel_heights = c(0.05,0.24,0.24,0.24,0.24)
)

# Final plot grid, with legend
final_PDP_ageph_SEV_plot <- plot_grid(
  allplotsgrid,
  get_legend(CA_PDP_plot_ageph_ModelComp_AUS),
  ncol = 1, rel_heights = c(0.9,0.1)
)

final_PDP_ageph_SEV_plot

# Save plot as PDF
ggsave("final_PDP_ageph_SEV_plot.pdf",
       final_PDP_ageph_SEV_plot, 
       device = cairo_pdf,
       width = 32,
       height = 10,
       scale = 1.2,
       units = "cm")

### ----- PDP BM  -----

NC_PDP_plot_BM_ModelComp_BE <- rbind(
  NC_PDP_GBM_AllVARS_BE$bm %>% mutate(Model = 'GBM', Data = 'Belgium'),
  NC_PDP_CANN_GBM_flex_AllVARS_BE$bm %>% mutate(Model = 'CANN GBM flex', Data = 'Belgium'),
  NC_PDP_FFNN_AllVARS_BE$bm %>% mutate(Model = 'FFNN', Data = 'Belgium')) %>% 
  filter(Testfold == 1) %>% 
  mutate(Model = fct_relevel(Model, c("GBM", "FFNN", "CANN GBM flex"))) %>% 
  ggplot(aes(x = x, y = y)) +  
  geom_line(aes(color = Model), linewidth = 1.4) + 
  #geom_col(aes(color = Model, fill = Model), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.8)) +  
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab(vars_with_label_BE %>% filter(Variable == 'bm') %>% pull(xlabels)) + ylab("Average predicted claim frequency") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=12), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM", "FFNN","CANN GBM flex"),
                     labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                     values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  labs(subtitle = 'Frequency modelling')

NC_PDP_plot_BM_ModelComp_FR <- rbind(
  NC_PDP_GBM_AllVARS_FR$BonusMalus %>% mutate(Model = 'GBM', Data = 'France'),
  NC_PDP_CANN_GBM_flex_AllVARS_FR$BonusMalus %>% mutate(Model = 'CANN GBM flex', Data = 'France'),
  NC_PDP_FFNN_AllVARS_FR$BonusMalus %>% mutate(Model = 'FFNN', Data = 'France')) %>% 
  filter(Testfold == 1) %>% 
  mutate(Model = fct_relevel(Model, c("GBM", "FFNN", "CANN GBM flex"))) %>% 
  ggplot(aes(x = x, y = y)) +  
  geom_line(aes(color = Model), linewidth = 1.4) + 
  #geom_col(aes(color = Model, fill = Model), size = 0.8, alpha = 0.7, position = position_dodge(width = 0.8)) +  
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab(vars_with_label_FR %>% filter(Variable == 'BonusMalus') %>% pull(xlabels)) + ylab("Average predicted claim frequency") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=12), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM", "FFNN","CANN GBM flex"),
                     labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                     values = c("#52BDEC","#116E8A", "#00407A")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM", "FFNN","CANN GBM flex"),
                    labels = c('GBM', "FFNN", 'CANN GBM flexible'), 
                    values = c("#52BDEC","#116E8A", "#00407A")) + 
  #scale_x_discrete(labels = c('< 21','[21,26[','[26,30[','[30,40[','[40,50[','[50,70[','\u2265 70')) +
  labs(subtitle = 'Frequency modelling')

# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(NC_PDP_plot_BM_ModelComp_BE + theme(legend.position = "none") + 
                              labs(subtitle = "Belgium") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.16)), 
                            NC_PDP_plot_BM_ModelComp_FR + theme(legend.position = "none") + 
                              labs(y = NULL, subtitle = "France") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.16)),
                            align = "hv")

# Make a grid of all plots, with country names
allplotsgrid <- plot_grid(#ggdraw() + draw_label(''),ggdraw() + draw_label("Out-of-sample Poisson Deviance", size = 12),ggdraw() + draw_label("Out-of-sample gamma Deviance", size = 12),
  allplotslist[[1]], allplotslist[[2]],
  ncol = 2, rel_widths = c(0.5,0.5)#, rel_heights = c(0.05,0.24,0.24,0.24,0.24)
)

# Final plot grid, with legend
final_PDP_bm_plot <- plot_grid(
  allplotsgrid,
  get_legend(NC_PDP_plot_ageph_ModelComp_AUS),
  ncol = 1, rel_heights = c(0.9,0.1)
)

final_PDP_bm_plot

# Save plot as PDF
ggsave("final_PDP_bm_plot.pdf",
       final_PDP_bm_plot, 
       device = cairo_pdf,
       width = 16,
       height = 10,
       scale = 1.2,
       units = "cm")


### ----- PDP Spatial -----

sf_use_s2(FALSE) # Due to update to sf package, we need to switch of geometry check
belgium_shape_sf <- st_read('./shape file Belgie postcodes/npc96_region_Project1.shp', quiet = TRUE)
belgium_shape_sf <- st_transform(belgium_shape_sf, CRS("+proj=longlat +datum=WGS84"))

# Join pdp info with the Belgian Shape file
NC_PDP_spatial_GBM_BE <- left_join(belgium_shape_sf, 
                                   NC_PDP_GBM_AllVARS_BE$postcode %>% filter(Testfold == 1) %>% select(POSTCODE = x, NC = y), 
                                   by = "POSTCODE")

NC_PDP_spatial_CANN_GBM_flex_BE <- left_join(belgium_shape_sf, 
                                             NC_PDP_CANN_GBM_flex_AllVARS_BE$postcode %>% filter(Testfold == 1) %>% select(POSTCODE = x, NC = y), 
                                             by = "POSTCODE")

NC_PDP_spatial_FFNN_BE <- left_join(belgium_shape_sf, 
                                    NC_PDP_FFNN_AllVARS_BE$postcode %>% filter(Testfold == 1) %>% select(POSTCODE = x, NC = y), 
                                    by = "POSTCODE")

EMP_NC <- data_BE_PC %>% select(POSTCODE = postcode, NC = nclaims) %>% 
  group_by(POSTCODE) %>% 
  summarise(NC = mean(NC))

# Join pdp info with the Belgian Shape file
NC_EMP_spatial_BE <- left_join(belgium_shape_sf, EMP_NC, by = "POSTCODE")

# Plot the PDP effect over map of Belgium
NC_PDP_plot_spatial_GBM_BE <- tm_shape(NC_PDP_spatial_GBM_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "NC", title = "Average number of claims", 
          textNA = "No Policyholders", 
          style = "cont", breaks = seq(0.05, 0.20, by = 0.05),
          palette = "Blues", colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 1,
            legend.title.size=1.5,
            title= 'GBM', 
            title.position = c('left', 'top'),
            title.size = 1.5)

NC_PDP_plot_spatial_CANN_GBM_flex_BE <- tm_shape(NC_PDP_spatial_CANN_GBM_flex_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "NC", title = "Average number of claims", 
          textNA = "No Policyholders", 
          style = "cont", breaks = seq(0.05, 0.20, by = 0.05),
          palette = "Blues", colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 1,
            legend.title.size=1.5,
            title= 'CANN GBM flexible', 
            title.position = c('left', 'top'),
            title.size = 1.5)

NC_PDP_plot_spatial_FFNN_BE <- tm_shape(NC_PDP_spatial_FFNN_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "NC", title = "Average number of claims", 
          textNA = "No Policyholders", 
          style = "cont", breaks = seq(0.05, 0.20, by = 0.05),
          palette = "Blues", colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 1,
            legend.title.size=1.5,
            title= 'FFNN', 
            title.position = c('left', 'top'),
            title.size = 1.5)

# Plot the PDP effect over map of Belgium
NC_EMP_spatial_BE_plot <- tm_shape(NC_EMP_spatial_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "NC", title = "Emperical number of claims", 
          textNA = "No Policyholders", 
          style = "cont", breaks = seq(0.05, 0.20, by = 0.05),
          palette = "Blues", colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 1,
            legend.title.size=1.5,
            title= 'Emprical claim frequency', 
            title.position = c('left', 'top'),
            title.size = 1.5)

tmap_save(NC_PDP_plot_spatial_GBM_BE, 
          filename = "NC_PDP_plot_spatial_GBM_BE.pdf", 
          device = cairo_pdf, 
          height = 7, 
          width = 7)

tmap_save(NC_PDP_plot_spatial_CANN_GBM_flex_BE, 
          filename = "NC_PDP_plot_spatial_CANN_GBM_flex_BE.pdf", 
          device = cairo_pdf, 
          height = 7, 
          width = 7)

tmap_save(NC_PDP_plot_spatial_FFNN_BE, 
          filename = "NC_PDP_plot_spatial_FFNN_BE.pdf", 
          device = cairo_pdf, 
          height = 7, 
          width = 7)

tmap_save(NC_EMP_spatial_BE_plot, 
          filename = "NC_EMP_spatial_BE_plot.pdf", 
          device = cairo_pdf, 
          height = 7, 
          width = 7)

NC_pdp_spatial_grid <- tmap_arrange(NC_PDP_plot_spatial_GBM_BE, NC_PDP_plot_spatial_CANN_GBM_flex_BE, 
                                    NC_PDP_plot_spatial_FFNN_BE, NC_EMP_spatial_BE_plot,
                                    ncol = 2, widths = c(0.5, 0.5))

tmap_save(NC_pdp_spatial_grid, 
          filename = "NC_pdp_spatial_grid.pdf", 
          device = cairo_pdf, 
          height = 14, 
          width = 14)

# -----
## ----- Surrogate results -----

load('NC_CANN_GBM_flex_SURR_allFolds_AUS')
load('NC_CANN_GBM_flex_SURR_allFolds_BE')
load('NC_CANN_GBM_flex_SURR_allFolds_FR')

# These results come from the Managerial Insights code
load('tariff_plan_AUS')
load('tariff_plan_BE')
load('tariff_plan_FR')

OOS_surrogates <- lapply(c('AUS', 'BE', 'FR'), function(country){
  lapply(c(1:6), function(fold){
    tibble(Data = country,
           Problem = c('Frequency', 'Severity'),
           OOS = c(
             dev_poiss(get(paste0('tariff_plan_',country)) %>% filter(fold_nr == fold) %>%  pull(nclaims), 
                       get(paste0('tariff_plan_',country)) %>% filter(fold_nr == fold) %>%  pull(surr_pred_poiss)),
             dev_gamma(get(paste0('tariff_plan_',country)) %>% filter(fold_nr == fold) %>% filter(!is.na(average)) %>%  pull(average), 
                       get(paste0('tariff_plan_',country)) %>% filter(fold_nr == fold) %>% filter(!is.na(average)) %>%  pull(surr_pred_gamma),
                       get(paste0('tariff_plan_',country)) %>% filter(fold_nr == fold) %>% filter(!is.na(average)) %>%  pull(nclaims))
           ), 
           fold_nr = fold)
  }) %>% do.call(rbind,.)
}) %>% do.call(rbind,.) %>% arrange(Data,Problem,fold_nr)

# Results for in paper
OOS_surrogates %>% filter(fold_nr == 1, Problem == 'Frequency')

# The used variables in the surrogate GLM
NC_CANN_GBM_flex_SURR_allFolds_AUS[[1]]$slct_feat
NC_CANN_GBM_flex_SURR_allFolds_BE[[1]]$slct_feat
NC_CANN_GBM_flex_SURR_allFolds_FR[[1]]$slct_feat

### ----- Plot: Surrogate coefficients -----

NC_SURR_coefficients_FR <- summary(NC_CANN_GBM_flex_SURR_allFolds_FR[[1]]$best_surr)$coefficients %>% 
  as_tibble(rownames = NA) %>% 
  rownames_to_column() %>% 
  mutate(coef = Estimate, lower_sd = coef - `Std. Error`, upper_sd = coef + `Std. Error`) %>% 
  mutate_at(vars(coef,lower_sd,upper_sd), ~exp(.)) %>% 
  select(rowname,coef,lower_sd,upper_sd)

# Check the variables that lead to interaction effect
NC_CANN_GBM_flex_SURR_allFolds_FR[[1]]$best_surr$data %>% select(c('VehAge','VehBrand','VehAge_VehBrand_')) %>% unique() %>% filter(VehAge_VehBrand_ == 2)
# Only vehAge 1 and VehBrand B12 lead to interaction group 2, all others are group 1

NC_SURR_coefficients_FR_singleeffect <- NC_SURR_coefficients_FR %>% filter(!grepl('VehAge_VehBrand',rowname))
all_vars_singeleffect <- (NC_CANN_GBM_flex_SURR_allFolds_FR[[1]]$slct_feat %>% names)[!((NC_CANN_GBM_flex_SURR_allFolds_FR[[1]]$slct_feat %>% names) %in% 'VehAge_VehBrand')]

# Manually selected the groups for each policyholder
policyholder_selection <- tibble(
  VehPower_ = NC_SURR_coefficients_FR_singleeffect %>% filter(grepl('VehPower',rowname)) %>% arrange(coef) %>% 
    slice(3,5,9) %>% mutate(rowname = str_replace(rowname,'VehPower_','')) %>% pull(rowname),
  VehAge_ = NC_SURR_coefficients_FR_singleeffect %>% filter(grepl('VehAge',rowname)) %>% arrange(coef) %>% 
    slice(1,2,2) %>% mutate(rowname = str_replace(rowname,'VehAge_','')) %>% pull(rowname),
  DrivAge_ = NC_SURR_coefficients_FR_singleeffect %>% filter(grepl('DrivAge',rowname)) %>% arrange(coef) %>% 
    slice(1,3,5) %>% mutate(rowname = str_replace(rowname,'DrivAge_','')) %>% pull(rowname),
  BonusMalus_ = NC_SURR_coefficients_FR_singleeffect %>% filter(grepl('BonusMalus',rowname)) %>% arrange(coef) %>% 
    slice(1,5,10) %>% mutate(rowname = str_replace(rowname,'BonusMalus_','')) %>% pull(rowname),
  VehBrand_ = NC_SURR_coefficients_FR_singleeffect %>% filter(grepl('VehBrand',rowname)) %>% arrange(coef) %>% 
    slice(1,2,2) %>% mutate(rowname = str_replace(rowname,'VehBrand_','')) %>% pull(rowname),
  VehGas_ = NC_SURR_coefficients_FR_singleeffect %>% filter(grepl('VehGas',rowname)) %>% arrange(coef) %>% 
    slice(1,1,1) %>% mutate(rowname = str_replace(rowname,'VehGas_','')) %>% pull(rowname),
  Density_ = NC_SURR_coefficients_FR_singleeffect %>% filter(grepl('Density',rowname)) %>% arrange(coef) %>% 
    slice(1,4,6) %>% mutate(rowname = str_replace(rowname,'Density_','')) %>% pull(rowname),
  Region_ = NC_SURR_coefficients_FR_singleeffect %>% filter(grepl('Region',rowname)) %>% arrange(coef) %>% 
    slice(1,4,6) %>% mutate(rowname = str_replace(rowname,'Region_','')) %>% pull(rowname)
) %>% mutate(expo = 1, VehAge_VehBrand_ = '1')

# Manually change some to reference level
policyholder_selection[2,'VehGas_'] <- '{Regular}'
policyholder_selection[3,'VehGas_'] <- '{Regular}'
policyholder_selection[2,'VehAge_'] <- '[2, 2]'
policyholder_selection[1,'BonusMalus_'] <- '[50, 54]'

# The selected low, med and high policyholders with the prediction by the surrogate
policyholder_selection_pred <- bind_cols(policyholder_selection,
          pred = predict(NC_CANN_GBM_flex_SURR_allFolds_FR[[1]]$best_surr, 
                         policyholder_selection, 
                         type = 'response')
)

# Choose specific values and add them, to the policyholders for CANN predictions
policyholder_selection_pred_forCANN <- policyholder_selection_pred %>% 
  mutate(
    VehPower = c(4,7,9),
    VehAge = c(3,2,1),
    DrivAge = c(3,2,1),
    BonusMalus = c(50,76,99),
    VehBrand = c('B11','B10','B6'),
    VehGas = c('Diesel','Regular','Regular'),
    Density = c(1,4,10),
    Region = c('Auvergne','Basse-Normandie','Corse'),
    Area = c('A','B','C')
  ) %>% 
  mutate(across(all_of(c('VehPower')),
                ~ factor(., levels = levels(data_FR$.), ordered = TRUE)
  )) %>% 
  mutate(across(all_of(c('VehAge', 'DrivAge', 'VehBrand', 'VehGas', 'Area', 'Region')),
                ~ factor(., levels = levels(data_FR$.))
  )) %>% 
  mutate(BonusMalus = as.integer(BonusMalus))

# Use the Maidrr:explain to get the plotted feature contributions
coef_plots <- lapply(c(1:3), function(risk){
  NC_CANN_GBM_flex_SURR_allFolds_FR[[1]]$best_surr %>% 
    explain(instance = policyholder_selection_pred_forCANN[risk, ], plt = F) %>% 
    ggplot(aes(y = reorder(value, fit_resp), x = fit_resp)) + geom_point(size = 3) +
    geom_errorbar(aes(xmin = lwr_conf, xmax = upr_conf), width = 0.5) +
    geom_vline(xintercept = exp(0), alpha = 0.5, linewidth = 0.8, linetype = 'dashed') +
    labs(y = '', x = 'Feature contributions') + theme_bw()
})

### ----- Plot: Shapley values -----

#install.packages('iml')
library(iml)

# Data used in shapley calculating is the policyholders + a random sample of size 1000
X = bind_rows(policyholder_selection_pred_forCANN %>% 
            select(!contains("_")) %>% 
            select(!pred),
            data_FR %>% select(!c(id,fold_nr,average,nclaims)) %>% sample_n(size=1000))

# We make a predictor object for the IML package
mod <- Predictor$new(
  model = NC_object_FR[[1]],
  data = X,
  predict.function = CANN_model_predictions_perpoint)
  
# Then we apply the Shapley method:
shapley_per_policyholder <- lapply(c(1:3), function(n){
  sh <- iml::Shapley$new(mod, x.interest = X[n,])
  # We do not want the effect of expo shown in the Shapley plots
  sh$results <- sh$results %>% filter(feature != 'expo')
  return(sh)
})
# save(shapley_per_policyholder, file = 'shapley_per_policyholder')
# load('shapley_per_policyholder')

# Make the shapley plots and feature importance plots

# Set margins for plot combinations
margin_set <- c(0,0.4,0,-0.4)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(coef_plots[[1]] + labs(title="", x ="Feature contribution", y = "Surrogate GLM") +
                              theme(axis.title.y = element_text(vjust = -5)) +
                              theme(plot.margin = unit(margin_set + c(0,0,0,0), "cm"), plot.subtitle=element_text(size=12, hjust = 0.16)), 
                            plot(shapley_per_policyholder[[1]]) + theme_bw() + labs(title="Low risk", y ="Shapley value", x = "CANN GBM flexible") + 
                              theme(axis.title.y = element_text(vjust = -5)) +
                              theme(plot.margin = unit(margin_set + c(0,0,0,0), "cm"), plot.subtitle=element_text(size=12, hjust = 0.16)),
                            coef_plots[[2]] + labs(title="", x ="Feature contribution", y = "") +
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.16)), 
                            plot(shapley_per_policyholder[[2]]) + theme_bw() + labs(title="Medium risk", y ="Shapley value", x = "")+ 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.16)),
                            coef_plots[[3]] + labs(title="", x ="Feature contribution", y = "") +
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.16)), 
                            plot(shapley_per_policyholder[[3]]) + theme_bw() + labs(title="High risk", y ="Shapley value", x = "")+ 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.16)),
                            align = "hv")

# Make a grid of all plots, with country names
Shapley_vs_Coefficients_plot <- plot_grid(
  allplotslist[[2]], allplotslist[[4]], allplotslist[[6]],
  allplotslist[[1]], allplotslist[[3]], allplotslist[[5]],
  ncol = 3, rel_widths = c(0.33,0.33,0.33)
)

Shapley_vs_Coefficients_plot

ggsave("NC_FR_Shapley_vs_Coefficients_plot.pdf",
       Shapley_vs_Coefficients_plot, 
       device = cairo_pdf,
       width = 14,
       height = 7,
       scale = 2.4,
       units = "cm")

### ----- Plot: French surrogate -----

load('NC_CANN_GBM_flex_SURR_allFolds_FR')
load('NC_PDP_CANN_GBM_flex_AllVARS_FR')

# Extract the four wanted PD effects from the surrogate model on French data
FR_PDP_BonusMalus <- NC_PDP_CANN_GBM_flex_AllVARS_FR$BonusMalus %>% filter(Testfold == 1)
FR_PDP_Region <- NC_PDP_CANN_GBM_flex_AllVARS_FR$Region %>% filter(Testfold == 1)
FR_PDP_VehPower <- NC_PDP_CANN_GBM_flex_AllVARS_FR$VehPower %>% filter(Testfold == 1)
FR_PDP_DrivAge <- NC_PDP_CANN_GBM_flex_AllVARS_FR$DrivAge %>% filter(Testfold == 1)

# Extract splitting points from the surrogate, works only for continuous numerical variables
get_splits <- function(surrogate, var){
  surrogate$best_surr$R %>% 
    as.data.frame() %>% 
    rownames_to_column() %>% 
    pull(rowname) %>% as_tibble() %>% 
    filter(grepl(var, value)) %>% 
    mutate(value = gsub(var, '', value)) %>% 
    mutate(value = gsub("\\[", '', value)) %>% 
    mutate(value = gsub("\\]", '', value)) %>% 
    separate(value, c("start", "end"), ", ") %>% 
    mutate_at(c('start', 'end'), as.numeric) %>% 
    arrange(start)
}

# French spatial plot

# Names in data and shape file are slightly different. 
# We also add the grouping of the surrogate here
french_regions <- tribble(
  ~old_region, ~new_region, ~group,
  "Rhone-Alpes", "Rh\xf4ne-Alpes", 'Group_0',
  "Picardie", "Picardie", 'Group_0',                
  "Aquitaine", "Aquitaine", 'Group_2',                
  "Nord-Pas-de-Calais", "Nord-Pas-de-Calais", 'Group_2',      
  "Languedoc-Roussillon", "Languedoc-Roussillon", 'Group_6',     
  "Pays-de-la-Loire", "Pays de la Loire", 'Group_4',           
  "Provence-Alpes-Cotes-D'Azur", "Provence-Alpes-C\xf4te d'Azur", 'Group_1',
  "Ile-de-France",  "\xcele-de-France", 'Group_0',            
  "Centre", "Centre", 'Group_0',
  "Corse",  "Corse", 'Group_5',                   
  "Auvergne", "Auvergne", 'Group_3', 
  "Poitou-Charentes", "Poitou-Charentes", 'Group_1',         
  "Bourgogne", "Bourgogne", 'Group_4',                  
  "Bretagne", "Bretagne", 'Group_0',               
  "Midi-Pyrenees", "Midi-Pyr\xe9n\xe9es", 'Group_6',             
  "Alsace", "Alsace", 'Group_1',                     
  "Basse-Normandie", "Basse-Normandie", 'Group_1',           
  "Champagne-Ardenne", "Champagne-Ardenne", 'Group_5',        
  "Franche-Comte", "Franche-Comt\xe9", 'Group_6',             
  "Limousin", "Limousin", 'Group_4',                  
  "Haute-Normandie", "Haute-Normandie",'Group_1'
)

# Read in French shape file
french_shape_sf <- st_read('./shape file French regions/ym781wr7170.shp', quiet = TRUE)
french_shape_sf <- st_transform(french_shape_sf, CRS("+proj=longlat +datum=WGS84"))
french_shape_sf <- st_simplify(french_shape_sf, dTolerance = 0.00001)

# Remove Lorraine, not in data
french_shape_sf <- french_shape_sf %>% filter(name_1 != "Lorraine")

# Add surrogate groups per French region
french_shape_sf <- french_shape_sf %>% bind_cols(Group = c(1,2,3,0,1,4,0,0,5,5,6,1,6,4,6,2,4,0,1,1,0)) %>% mutate(Group = paste0("Group_",Group))

# Average PD per group
FR_Region_PDP_perGroup <- FR_PDP_Region %>% 
  left_join(french_regions, by=c("x"="old_region")) %>% 
  group_by(group) %>% 
  summarise(y = mean(y)) %>% mutate(y = format(round(y, digits=4), nsmall = 2) 
) %>% mutate(y = factor(y))

# Join pdp info with the French Shape file
NC_PDP_spatial_FR <- left_join(french_shape_sf, FR_Region_PDP_perGroup, by = c("Group"="group"))

# BBOX magic to put the legend in the map further to the left
bbox_new <- st_bbox(NC_PDP_spatial_FR) # current bounding box
xrange <- bbox_new$xmax - bbox_new$xmin # range of x values
yrange <- bbox_new$ymax - bbox_new$ymin # range of y values
bbox_new[1] <- bbox_new[1] - (0.15 * xrange) # xmin - left
#bbox_new[3] <- bbox_new[3] + (0.25 * xrange) # xmax - right
bbox_new[2] <- bbox_new[2] - (0.08 * yrange) # ymin - bottom
#bbox_new[4] <- bbox_new[4] + (0.2 * yrange) # ymax - top
bbox_new <- bbox_new %>%  # take the bounding box ...
  st_as_sfc() # ... and make it a sf polygon

# Plot of the spatial effect on France map
NC_PDP_spatial_FR_plot <- tm_shape(NC_PDP_spatial_FR, bbox = bbox_new) + 
  tm_borders(col = "black") + 
  tm_fill(col = "y", title = "Spatial partial \ndependency effect \nper group", 
          textNA = "No Policyholders", 
          #style = "cont", 
          palette = "Accent", 
          colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 1.4,
            legend.title.size = 1.6)

NC_PDP_spatial_FR_plot

tmap_save(NC_PDP_spatial_FR_plot, 
          filename = "NC_PDP_spatial_FR_plot.pdf", 
          device = cairo_pdf, 
          height = 7, 
          width = 7)

# French bonus-malus plot

breaks <- get_splits(NC_CANN_GBM_flex_SURR_allFolds_FR[[1]],'BonusMalus_') %>% 
  mutate(lower = start-0.5) %>% 
  mutate(upper = lead(lower)) %>% 
  add_row(lower = 50, upper = 55.5, .before=1) %>% 
  mutate(upper = replace(upper, end == 230, 230)) %>% 
  rownames_to_column(var = 'gr') %>% 
  select(lower,upper,gr)

FR_PDP_BonusMalus_withGroups <- fuzzy_join(FR_PDP_BonusMalus, breaks, by = c('x'='lower', 'x'='upper'), match_fun = list(`>`,`<=`))

NC_PDP_BonusMalus_FR_Surr <- FR_PDP_BonusMalus_withGroups %>% mutate(Group = paste0("Group_",gr)) %>% 
  ggplot(aes(x = x)) + geom_line(aes(y = y, color = Group), size = 1) + geom_point(aes(y = y, color = Group), size = 0.8) +
  theme_bw() + 
  scale_colour_discrete("Accent") +
  theme(legend.position = 'none') + 
  xlab("Bonus-malus score") + 
  ylab("Partial dependency effect") + 
  geom_vline(xintercept = breaks$lower[2:nrow(breaks)])

NC_PDP_BonusMalus_FR_Surr

ggsave("NC_PDP_BonusMalus_FR_Surr.pdf",
       NC_PDP_BonusMalus_FR_Surr, 
       device = cairo_pdf,
       width = 7,
       height = 7,
       scale = 1.2,
       units = "cm")

# French Vehicle Power ##NOT USED IN PAPER

get_splits(NC_CANN_GBM_flex_SURR_allFolds_FR[[1]],'VehPower_')

NC_PDP_VehPower_FR_Surr <- FR_PDP_VehPower %>% 
  arrange(x) %>% 
  bind_cols(Group = c("Group_1","Group_2","Group_3","Group_4","Group_5","Group_6","Group_7","Group_7","Group_8","Group_9","Group_9","Group_10")) %>% 
  ggplot(aes(x = x)) + geom_col(aes(y = y, color = Group, fill = Group), size = 0.8, alpha = 0.7) + 
  theme_bw() + 
  scale_colour_discrete("Accent") +
  theme(legend.position = 'none') + 
  xlab("Vehicle power") + 
  ylab("Partial dependency effect")

ggsave("NC_PDP_VehPower_FR_Surr.pdf",
       NC_PDP_VehPower_FR_Surr, 
       device = cairo_pdf,
       width = 7,
       height = 7,
       scale = 1.2,
       units = "cm")

# French Driv Age

get_splits(NC_CANN_GBM_flex_SURR_allFolds_FR[[1]],'DrivAge_')

NC_PDP_DrivAge_FR_Surr <- FR_PDP_DrivAge %>% 
  arrange(x) %>% 
  bind_cols(Group = c("Group_1","Group_2","Group_3","Group_4","Group_5","Group_5","Group_6")) %>% 
  ggplot(aes(x = x)) + geom_col(aes(y = y, color = Group, fill = Group), size = 0.8, alpha = 0.7) + 
  theme_bw() + 
  scale_colour_discrete("Accent") +
  theme(legend.position = 'none', axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + 
  xlab("Policyholder age") + 
  ylab("Partial dependency effect") + 
  scale_x_discrete(labels = c('< 21','[21,26[','[26,30[','[30,40[','[40,50[','[50,70[','\u2265 70'))
  

ggsave("NC_PDP_DrivAge_FR_Surr.pdf",
       NC_PDP_DrivAge_FR_Surr, 
       device = cairo_pdf,
       width = 7,
       height = 7,
       scale = 1.2,
       units = "cm")

# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(NC_PDP_BonusMalus_FR_Surr + theme(legend.position = "none") + 
                              labs() + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.16)), 
                            NC_PDP_DrivAge_FR_Surr + theme(legend.position = "none") + 
                              labs(y = NULL) + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12, hjust = 0.16)),
                            align = "hv")

# Make a grid of all plots, with country names
final_SURR_DBDrivage_plot <- plot_grid(#ggdraw() + draw_label(''),ggdraw() + draw_label("Out-of-sample Poisson Deviance", size = 12),ggdraw() + draw_label("Out-of-sample gamma Deviance", size = 12),
  allplotslist[[1]], allplotslist[[2]],
  ncol = 2, rel_widths = c(0.5,0.5)#, rel_heights = c(0.05,0.24,0.24,0.24,0.24)
)

final_SURR_DBDrivage_plot

# Save plot as PDF
ggsave("final_SURR_DBDrivage_plot.pdf",
       final_SURR_DBDrivage_plot, 
       device = cairo_pdf,
       width = 14,
       height = 7,
       scale = 1.2,
       units = "cm")

# -----
## ----- Plot: One-hot versus autoencoder -----

load('OHvsAE_comp_allDataSets')

OHvsAE_comp_allDataSets %>% mutate(Embedding = factor(Embedding, levels = unique(Embedding))) %>% 
  filter(Dataset == 'BE') %>%
  ggplot(aes(x=Embedding, y=Freq_OOS, color = Model, group = Model)) + 
  geom_line() + geom_point() + theme_bw()

OHvsAE_comp_allDataSets %>% 
  gather(key = 'Problem', value = 'OOS', 3:4) %>% 
  mutate(Problem = recode_factor(Problem, Freq_OOS = 'Frequency', Sev_OOS = 'Severity')) %>% 
  mutate(Embedding = factor(Embedding, levels = unique(Embedding))) %>% 
  ggplot(aes(x=Embedding, y=OOS, color = Model, group = Model)) + 
  geom_line() + geom_point() + theme_bw() + facet_grid(Problem ~ Dataset, scales = 'free')

lapply(expand_grid(c('AUS','BE','FR','NOR'),c('Frequency','Severity')), function(x){
  OHvsAE_comp_allDataSets %>% 
    gather(key = 'Problem', value = 'OOS', 3:4) %>% 
    mutate(Problem = recode_factor(Problem, Freq_OOS = 'Frequency', Sev_OOS = 'Severity')) %>% 
    mutate(Embedding = factor(Embedding, levels = unique(Embedding))) %>% 
    filter(Dataset == x[1]) %>% filter(Problem == x[2])
})

plots_grid <- expand_grid(Data = c('AUS','BE','FR','NOR'),
                          Problem = c('Frequency','Severity'))

all_compplots_list <- apply(plots_grid, 1, function(x){
  OHvsAE_comp_allDataSets %>% 
    gather(key = 'Problem', value = 'OOS', 3:4) %>% 
    mutate(Problem = recode_factor(Problem, Freq_OOS = 'Frequency', Sev_OOS = 'Severity')) %>% 
    mutate(Embedding = factor(Embedding, levels = unique(Embedding))) %>% 
    filter(Dataset == x[[1]]) %>% filter(Problem == x[[2]]) %>% 
    ggplot(aes(x=Embedding, y=OOS, color = Model, group = Model)) + 
    geom_line(size = 1.4) + geom_point(size = 1.4) + theme_bw() 
  #scale_y_continuous(expand = expansion(add = 0.01))
}) %>% setNames(paste0(plots_grid$Data,'_',plots_grid$Problem))

# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(all_compplots_list$AUS_Freq + theme(legend.position = "none") + 
                              labs(x = NULL, y = "Poisson deviance", subtitle = "Australian data") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12),
                                    axis.ticks.x=element_blank(), axis.title.y=element_text(size=12)) +
                              ylim(0.370, 0.400), 
                            all_compplots_list$AUS_Sev + theme(legend.position = "none") + 
                              labs(x = NULL, y = "Gamma deviance") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12),
                                    axis.title.y=element_text(size=12)) + 
                              ylim(1.53, 1.58), 
                            all_compplots_list$BE_Freq+ theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Belgian data") + theme(plot.margin = unit(margin_set, "cm"), 
                                                                                          axis.ticks.x=element_blank()) + 
                              ylim(0.525, 0.540),  
                            all_compplots_list$BE_Sev+ theme(legend.position = "none") + 
                              labs(x = NULL,y = NULL) + theme(plot.margin = unit(margin_set, "cm")) + 
                              ylim(2.23, 2.25),  
                            all_compplots_list$FR_Freq+ theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "French data") + theme(plot.margin = unit(margin_set, "cm"), 
                                                                                         axis.ticks.x=element_blank()) + 
                              ylim(0.32, 0.35),  
                            all_compplots_list$FR_Sev+ theme(legend.position = "none") + 
                              labs(x = NULL,y = NULL) + theme(plot.margin = unit(margin_set, "cm")) + 
                              ylim(1.6, 1.9),  
                            all_compplots_list$NOR_Freq+ theme(legend.position = "none") + 
                              labs(x = NULL,y = NULL, subtitle = "Norwegian data") + theme(plot.margin = unit(margin_set, "cm"), 
                                                                                           axis.ticks.x=element_blank()) + 
                              ylim(0.27, 0.31),  
                            all_compplots_list$NOR_Sev+ theme(legend.position = "none") + 
                              labs(x = NULL,y = NULL) + theme(plot.margin = unit(margin_set, "cm")) + 
                              ylim(1.13, 1.15),  
                            align = "hv")

# Make a grid of all plots, with country names
allplotsgrid <- plot_grid(
  #ggdraw() + draw_text("Frequency", size = 14, angle = 90),
  allplotslist[[1]], allplotslist[[3]], allplotslist[[5]], allplotslist[[7]],
  #ggdraw() + draw_text("Severity", size = 14, angle = 90),
  allplotslist[[2]], allplotslist[[4]], allplotslist[[6]], allplotslist[[8]],
  ncol = 4, rel_widths = c(0.25, 0.25, 0.25, 0.25), rel_heights = c(0.5,0.5)
)

# Final plot grid, with legend
final_OHvsAE_comp_plot <- plot_grid(
  allplotsgrid,
  get_legend(all_compplots_list$AUS_Frequency +  theme(legend.direction = "horizontal", legend.text=element_text(size=12))),
  ncol = 1, rel_heights = c(0.92,0.08)
)

final_OHvsAE_comp_plot

# Save plot as PDF
ggsave("final_OHvsAE_comp_plot.pdf",
       final_OHvsAE_comp_plot, 
       device = 'pdf',
       width = 20,
       height = 10,
       scale = 1.2,
       units = "cm")




# -----
## ----- Plots for slides IDS 2023 -----

load("NC_VI_CANN_GBM_flex_BE")
load("NC_VI_GBM_BE")
load("NC_VI_FFNN_BE")
load("CA_VI_CANN_GBM_flex_BE")
load("CA_VI_GBM_BE")
load("CA_VI_FFNN_BE")

load('NC_PDP_GBM_AllVARS_BE')
load('NC_PDP_CANN_GBM_flex_AllVARS_BE')
load('NC_PDP_FFNN_AllVARS_BE')
load('CA_PDP_GBM_AllVARS_BE')
load('CA_PDP_CANN_GBM_flex_AllVARS_BE')
load('CA_PDP_FFNN_AllVARS_BE')

load('NC_PDP_adjstNNreldiff_GBM_flex_AllVARS_BE')

### ----- VIPs for slides -----

BE_var_order <- c('use', 'fleet', 'agec', 'fuel', 'sex', 'coverage', 'postcode', 'power', 'ageph', 'bm')
BE_var_order_x <- BE_var_order %>% as_tibble_col(column_name = 'Variable') %>% left_join(vars_with_label_BE %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'))

data_VIP_CANNvsGBM_BE <- bind_rows(
  NC_VI_CANN_GBM_flex_BE %>% filter(Testfold == 1) %>% mutate(Model = "CANN GBM flex", Problem = 'Freq'),
  NC_VI_GBM_BE %>% filter(Testfold == 1) %>% mutate(Model = "GBM", Problem = 'Freq'),
  NC_VI_FFNN_BE  %>% filter(Testfold == 1) %>% mutate(Model = "FFNN", Problem = 'Freq'),
  CA_VI_CANN_GBM_flex_BE %>% filter(Testfold == 1) %>% mutate(Model = "CANN GBM flex", Problem = 'Sev'),
  CA_VI_GBM_BE %>% filter(Testfold == 1) %>% mutate(Model = "GBM", Problem = 'Sev'),
  CA_VI_FFNN_BE  %>% filter(Testfold == 1) %>% mutate(Model = "FFNN", Problem = 'Sev')
) %>% 
  mutate(Variable = fct_relevel(Variable, BE_var_order_x$Variable)) %>% 
  left_join(BE_var_order_x, by = 'Variable') %>% 
  mutate(xlabels = fct_relevel(xlabels, BE_var_order_x$xlabels)) %>% 
  mutate(Model = fct_relevel(Model, c("FFNN", "CANN GBM flex","GBM")))

NC_VIP_CANNvsGBM_BE <- data_VIP_CANNvsGBM_BE %>% filter(Model != 'FFNN') %>% 
  #left_join(vars_with_label_BE %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'), by = 'Variable') %>% 
  filter(Problem == 'Freq') %>% 
  ggplot(aes(y = xlabels)) +  
  geom_col(aes(x = scaled_VI, fill = Model), position="dodge", alpha = 0.8) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM","CANN GBM flex", "FFNN"),
                    labels = c('GBM', 'CANN GBM flexible', "FFNN"), 
                    values = c("#52BDEC","#00407A","#116E8A")) + 
  labs(subtitle = 'Frequency modelling')

CA_VIP_CANNvsGBM_BE <- data_VIP_CANNvsGBM_BE %>% filter(Model != 'FFNN') %>% 
  #left_join(vars_with_label_BE %>% add_row(Variable = 'postcode', xlabels = 'Postalcode'), by = 'Variable') %>% 
  filter(Problem == 'Sev') %>% 
  ggplot(aes(y = xlabels)) +  
  geom_col(aes(x = scaled_VI, fill = Model), position="dodge", alpha = 0.8) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Importance") + ylab("Covariates") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_fill_manual(name = 'Model', 
                    breaks = c("GBM","CANN GBM flex", "FFNN"),
                    labels = c('GBM', 'CANN GBM flexible', "FFNN"), 
                    values = c("#52BDEC","#00407A","#116E8A")) + 
  labs(subtitle = 'Severity modelling')

ggsave("NC_VIP_CANNvsGBM_BE.pdf",
       NC_VIP_CANNvsGBM_BE, 
       device = cairo_pdf,
       width = 7,
       height = 7,
       scale = 2,
       units = "cm")

ggsave("CA_VIP_CANNvsGBM_BE.pdf",
       CA_VIP_CANNvsGBM_BE, 
       device = cairo_pdf,
       width = 7,
       height = 7,
       scale = 2,
       units = "cm")

### ----- PDPs for slides -----

#### ----- Freq - Ageph -----

NC_PDP_ageph_ModelComp_BE <- rbind(
  NC_PDP_GBM_AllVARS_BE$ageph %>% mutate(Model = 'GBM'),
  NC_PDP_CANN_GBM_flex_AllVARS_BE$ageph %>% mutate(Model = 'CANN GBM flex'),
  NC_PDP_FFNN_AllVARS_BE$ageph %>% mutate(Model = 'FFNN')
)

NC_PDP_plot_ageph_BE <- NC_PDP_ageph_ModelComp_BE %>% 
  filter(Testfold == 1) %>% 
  filter(Model != 'FFNN') %>% 
  ggplot(aes(x = x, y = y)) +  
  geom_line(aes(color = Model), size = 1.4) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Policyholder age") + ylab("Average predicted claim frequency") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM","CANN GBM flex", "FFNN"),
                     labels = c('GBM', 'CANN GBM flexible', "FFNN"), 
                     values = c("#52BDEC","#00407A","#116E8A")) + 
  labs(subtitle = 'Frequency modelling')

ggsave("NC_PDP_plot_ageph_BE.pdf",
       NC_PDP_plot_ageph_BE, 
       device = cairo_pdf,
       width = 7,
       height = 7,
       scale = 2,
       units = "cm")

#### ----- Sev - Ageph -----

CA_PDP_ageph_ModelComp_BE <- rbind(
  CA_PDP_GBM_AllVARS_BE$ageph %>% mutate(Model = 'GBM'),
  CA_PDP_CANN_GBM_flex_AllVARS_BE$ageph %>% mutate(Model = 'CANN GBM flex'),
  CA_PDP_FFNN_AllVARS_BE$ageph %>% mutate(Model = 'FFNN')
)

CA_PDP_plot_ageph_BE <- CA_PDP_ageph_ModelComp_BE %>% 
  filter(Testfold == 1) %>% 
  filter(Model != 'FFNN') %>% 
  ggplot(aes(x = x, y = y)) +  
  geom_line(aes(color = Model), size = 1.4) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Policyholder age") + ylab("Average predicted claim severity") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM","CANN GBM flex", "FFNN"),
                     labels = c('GBM', 'CANN GBM flexible', "FFNN"), 
                     values = c("#52BDEC","#00407A","#116E8A")) + 
  labs(subtitle = 'Severity modelling')

ggsave("CA_PDP_plot_ageph_BE.pdf",
       CA_PDP_plot_ageph_BE, 
       device = cairo_pdf,
       width = 7,
       height = 7,
       scale = 2,
       units = "cm")

#### ----- Freq - BM -----

NC_PDP_bm_ModelComp_BE <- rbind(
  NC_PDP_GBM_AllVARS_BE$bm %>% mutate(Model = 'GBM'),
  NC_PDP_CANN_GBM_flex_AllVARS_BE$bm %>% mutate(Model = 'CANN GBM flex'),
  NC_PDP_FFNN_AllVARS_BE$bm %>% mutate(Model = 'FFNN')
)

NC_PDP_plot_bm_BE <- NC_PDP_bm_ModelComp_BE %>% 
  filter(Testfold == 1) %>% 
  filter(Model != 'FFNN') %>% 
  ggplot(aes(x = x, y = y)) +  
  geom_line(aes(color = Model), size = 1.4) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Bonus-Malus score") + ylab("Average predicted claim frequency") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM","CANN GBM flex", "FFNN"),
                     labels = c('GBM', 'CANN GBM flexible', "FFNN"), 
                     values = c("#52BDEC","#00407A","#116E8A")) + 
  labs(subtitle = 'Frequency modelling')

ggsave("NC_PDP_plot_bm_BE.pdf",
       NC_PDP_plot_bm_BE, 
       device = cairo_pdf,
       width = 7,
       height = 7,
       scale = 2,
       units = "cm")

#### ----- Sev - BM -----

CA_PDP_bm_ModelComp_BE <- rbind(
  CA_PDP_GBM_AllVARS_BE$bm %>% mutate(Model = 'GBM'),
  CA_PDP_CANN_GBM_flex_AllVARS_BE$bm %>% mutate(Model = 'CANN GBM flex'),
  CA_PDP_FFNN_AllVARS_BE$bm %>% mutate(Model = 'FFNN')
)

CA_PDP_plot_bm_BE <- CA_PDP_bm_ModelComp_BE %>% 
  filter(Testfold == 1) %>% 
  filter(Model != 'FFNN') %>% 
  ggplot(aes(x = x, y = y)) +  
  geom_line(aes(color = Model), size = 1.4) + 
  theme_bw() + 
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) + 
  xlab("Bonus-Malus score") + ylab("Average predicted claim severity") + 
  theme(legend.position="bottom", legend.direction="horizontal", 
        plot.title = element_text(size=18, margin=margin(0,0,50,0)),
        axis.title=element_text(size=16), plot.title.position = "plot",
        plot.subtitle=element_text(size=18, color="black")) + 
  scale_color_manual(name = 'Model', 
                     breaks = c("GBM","CANN GBM flex", "FFNN"),
                     labels = c('GBM', 'CANN GBM flexible', "FFNN"), 
                     values = c("#52BDEC","#00407A","#116E8A")) + 
  labs(subtitle = 'Severity modelling')

ggsave("CA_PDP_plot_bm_BE.pdf",
       CA_PDP_plot_bm_BE, 
       device = cairo_pdf,
       width = 7,
       height = 7,
       scale = 2,
       units = "cm")

#### ----- Freq - Spatial -----

NC_PDP_spatial_ModelComp_BE <- rbind(
  NC_PDP_GBM_AllVARS_BE$postcode %>% mutate(Model = 'GBM'),
  NC_PDP_CANN_GBM_flex_AllVARS_BE$postcode %>% mutate(Model = 'CANN GBM flex'),
  NC_PDP_adjstNNreldiff_GBM_flex_AllVARS_BE$postcode %>% mutate(Model = 'Relative differences')
)

sf_use_s2(FALSE) # Due to update to sf package, we need to switch of geometry check
belgium_shape_sf <- st_read('./shape file Belgie postcodes/npc96_region_Project1.shp', quiet = TRUE)
belgium_shape_sf <- st_transform(belgium_shape_sf, CRS("+proj=longlat +datum=WGS84"))

# Join pdp info with the Belgian Shape file
NC_PDP_spatial_GBM_BE <- left_join(belgium_shape_sf, 
                                   NC_PDP_GBM_AllVARS_BE$postcode %>% filter(Testfold == 1) %>% select(POSTCODE = x, NC = y), 
                                   by = "POSTCODE")

NC_PDP_spatial_CANN_GBM_flex_BE <- left_join(belgium_shape_sf, 
                                             NC_PDP_CANN_GBM_flex_AllVARS_BE$postcode %>% filter(Testfold == 1) %>% select(POSTCODE = x, NC = y), 
                                             by = "POSTCODE")

NC_PDP_spatial_adjstNNreldiff_GBM_flex_BE <- left_join(belgium_shape_sf, 
                                                       NC_PDP_adjstNNreldiff_GBM_flex_AllVARS_BE$postcode %>% filter(Testfold == 1) %>% select(POSTCODE = x, NC = y), 
                                                       by = "POSTCODE") %>% 
  mutate(NC = NC - 1)

# Plot the PDP effect over map of Belgium
NC_PDP_plot_spatial_GBM_BE <- tm_shape(NC_PDP_spatial_GBM_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "NC", title = "Average number of claims", 
          textNA = "No Policyholders", 
          style = "cont", breaks = seq(0.08, 0.24, by = 0.04),
          palette = "Blues", colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 1,
            legend.title.size=1.5,
            title= 'GBM', 
            title.position = c('left', 'top'))

NC_PDP_plot_spatial_CANN_GBM_flex_BE <- tm_shape(NC_PDP_spatial_CANN_GBM_flex_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "NC", title = "Average number of claims", 
          textNA = "No Policyholders", 
          style = "cont", breaks = seq(0.08, 0.24, by = 0.04),
          palette = "Blues", colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 1,
            legend.title.size=1.5,
            title= 'CANN GBM flexible', 
            title.position = c('left', 'top'))

NC_PDP_plot_spatial_adjstNNreldiff_GBM_flex_BE <- tm_shape(NC_PDP_spatial_adjstNNreldiff_GBM_flex_BE) + 
  tm_borders(col = "black") + 
  tm_fill(col = "NC", title = "Relative adjustements (%)", 
          textNA = "No Policyholders", 
          style = "cont", breaks = seq(-0.1, 0.1, by = 0.05),
          palette = "RdYlGn", colorNA = "white") +
  tm_layout(frame = FALSE, 
            legend.position = c("left", "bottom"),
            legend.text.size = 1,
            legend.title.size=1.5,
            title= 'Adjustement by CANN model', 
            title.position = c('left', 'top'))

tmap_save(NC_PDP_plot_spatial_GBM_BE, 
          filename = "NC_PDP_plot_spatial_GBM_BE.pdf", 
          device = cairo_pdf, 
          height = 7, 
          width = 7)

tmap_save(NC_PDP_plot_spatial_CANN_GBM_flex_BE, 
          filename = "NC_PDP_plot_spatial_CANN_GBM_flex_BE.pdf", 
          device = cairo_pdf, 
          height = 7, 
          width = 7)

tmap_save(NC_PDP_plot_spatial_adjstNNreldiff_GBM_flex_BE, 
          filename = "NC_PDP_plot_spatial_adjstNNreldiff_GBM_flex_BE.pdf", 
          device = cairo_pdf, 
          height = 7, 
          width = 7)


# -----
# ----- THE END -----
# -----