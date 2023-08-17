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
                   "doParallel", "maidrr", "zoo", "evtree")
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
# ----- ADDING TARIFF STRUCTURE -----

# With the wanted models, we make out-of-sample predictions and combine them into a tariff structure

# -----
## ----- Adding the GLM ----
### ----- Read-in info ----- 

load('binned_freq_AUS')
load('binned_sev_AUS')
load('binned_freq_FR')
load('binned_sev_FR')
load('binned_freq_NOR')
load('binned_sev_NOR')
glm_data<-readRDS("./Data/data_glm.rds") #Data used in glm model
glm_fits<-readRDS("./Data/mfits_glm.rds") #Glm's as constructed in Henckaerts et al. (2019)

# Add the Postal code back to Belgian data; we want effect of Postal code, not of lat-long separately
data_readin<-readRDS("./Data/Data.rds") #Dataset

add_oos_predictions <- function(original_data, fitted_models){
  
  # For each fold, add out of sample prediction
  lapply(1:6, function(fold){
    
    # Select data and model
    data_f <- original_data %>% filter(fold_nr == fold)
    model_set <- fitted_models[[fold]]
    
    # Predict with model on out of sample data
    if(is.null(model_set[[3]])){
      
      # GLM without EvTree binning
      bind_cols(data_f, prediction = predict(model_set[[2]], data_f, type = 'response')) %>% arrange(id)
      
    } else {
      # Binning with EvTree
      pred_per_var <- lapply(names(model_set[[3]]), function(var){
        tibble(!!var := predict(model_set[[3]][[var]], data_f %>% select(Value = all_of(var)), type = 'response'))
      }) %>% bind_cols()
      
      binned_data <- data_f %>% select(!(names(model_set[[3]]))) %>% bind_cols(pred_per_var) %>% mutate_at((names(model_set[[3]])), factor)
      
      # Predicting with GLM
      bind_cols(binned_data, prediction = predict(model_set[[2]], binned_data, type = 'response')) %>% arrange(id)
    }
  })
}

add_oos_predictions_BE <- function(original_data, fitted_models, binned_data){

  # For each fold, add out of sample prediction
  lapply(1:6, function(fold){
    ids <- original_data %>% filter(fold_nr == fold) %>% pull(id)
    pred <- predict(fitted_models[[1]], binned_data[[1]] %>% filter(id %in% ids), type = 'response')
    bind_cols(original_data %>% filter(fold_nr == fold), prediction = pred)
  }) %>% do.call(rbind,.) %>% left_join(data_readin %>% select(id,postcode), by = 'id') %>% select(!c(lat, long))
  
}

add_oos_predictions_BE(data_BE, glm_fits[1:6], glm_data[1:6]) 

### ----- Adding out-of-sample predictions -----

tarif_GLM_AUS <- reduce(list(data_AUS,
                             add_oos_predictions(data_AUS, binned_freq_AUS) %>% do.call(rbind, .) %>% select(c(id, glm_pred_poiss = prediction)),
                             add_oos_predictions(data_AUS, binned_sev_AUS) %>% do.call(rbind, .) %>% select(c(id, glm_pred_gamma = prediction))),
                        dplyr::left_join, by = 'id') %>% 
  mutate(glm_tariff = glm_pred_poiss*glm_pred_gamma)
save(tarif_GLM_AUS, file = 'tarif_GLM_AUS')

tarif_GLM_BE <- reduce(list(data_BE,
                            add_oos_predictions_BE(data_BE, glm_fits[1:6], glm_data[1:6]) %>% select(c(id, glm_pred_poiss = prediction)),
                            add_oos_predictions_BE(data_BE, glm_fits[7:12], glm_data[7:12])  %>% select(c(id, glm_pred_gamma = prediction))),
                       dplyr::left_join, by = 'id') %>% 
  mutate(glm_tariff = glm_pred_poiss*glm_pred_gamma)
save(tarif_GLM_BE, file = 'tarif_GLM_BE')

tarif_GLM_FR <- reduce(list(data_FR,
                            add_oos_predictions(data_FR, binned_freq_FR) %>% do.call(rbind, .) %>% select(c(id, glm_pred_poiss = prediction)),
                            add_oos_predictions(data_FR, binned_sev_FR) %>% do.call(rbind, .) %>% select(c(id, glm_pred_gamma = prediction))),
                       dplyr::left_join, by = 'id') %>% 
  mutate(glm_tariff = glm_pred_poiss*glm_pred_gamma)
save(tarif_GLM_FR, file = 'tarif_GLM_FR')

tarif_GLM_NOR <- reduce(list(data_NOR,
                             add_oos_predictions(data_NOR, binned_freq_NOR) %>% do.call(rbind, .) %>% select(c(id, glm_pred_poiss = prediction)),
                             add_oos_predictions(data_NOR, binned_sev_NOR) %>% do.call(rbind, .) %>% select(c(id, glm_pred_gamma = prediction))),
                        dplyr::left_join, by = 'id') %>% 
  mutate(glm_tariff = glm_pred_poiss*glm_pred_gamma)
save(tarif_GLM_NOR, file = 'tarif_GLM_NOR')

# -----
## ----- Adding the GBM ----
### ----- Read-in info ----- 

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

# Functions for easier GBM prediction
predict_model <- function(object, newdata) UseMethod('predict_model')

predict_model.gbm <- function(object, newdata) {
  predict(object, newdata, n.trees = object$n.trees, type = 'response')
}

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

# Add the Postal code back to Belgian data; we want effect of Postal code, not of lat-long separately
data_readin<-readRDS("./Data/Data.rds") #Dataset

data_BE_PC <- data_BE %>% left_join(data_readin %>% select(id,postcode), by = 'id') %>% select(!c(lat, long))

# Complete list of all postal codes in Belgium with lat long of center (for PDP use)
belgium_shape <- readOGR('./shape file Belgie postcodes/npc96_region_Project1.shp') %>% spTransform(CRS('+proj=longlat +datum=WGS84'))
all_latlong <- bind_cols(belgium_shape@data %>% as_tibble %>% select(postcode = POSTCODE), 
                         sp::coordinates(belgium_shape) %>% as_tibble %>% rename(lat = V2, long = V1) )

latlong_per_postalcode <- all_latlong

### ----- Adding out-of-sample predictions -----

tarif_GBM_AUS <- lapply(1:6, function(fold){
  bind_cols(
    data_AUS %>% filter(fold_nr == fold),
    gbm_pred_poiss = NC_gbm_prediction_perpoint(oos_freq_GBM_AUS[[fold]][[3]], data_AUS %>% filter(fold_nr == fold)),
    gbm_pred_gamma = CA_gbm_prediction_perpoint(oos_sev_GBM_AUS[[fold]][[3]], data_AUS %>% filter(fold_nr == fold))
  ) %>% 
    mutate(gbm_tariff = gbm_pred_poiss*gbm_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_GBM_AUS, file = 'tarif_GBM_AUS')

tarif_GBM_BE <- lapply(1:6, function(fold){
  bind_cols(
    data_BE_PC %>% filter(fold_nr == fold),
    gbm_pred_poiss = NC_gbm_prediction_perpoint(list(GBM_model = gbm_fits[[fold]], latlong_conversion = latlong_per_postalcode), data_BE_PC %>% filter(fold_nr == fold)),
    gbm_pred_gamma = CA_gbm_prediction_perpoint(list(GBM_model = gbm_fits[[6+fold]], latlong_conversion = latlong_per_postalcode), data_BE_PC %>% filter(fold_nr == fold))
  ) %>% 
    mutate(gbm_tariff = gbm_pred_poiss*gbm_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_GBM_BE, file = 'tarif_GBM_BE')

tarif_GBM_FR <- lapply(1:6, function(fold){
  bind_cols(
    data_FR %>% filter(fold_nr == fold),
    gbm_pred_poiss = NC_gbm_prediction_perpoint(oos_freq_GBM_FR[[fold]][[3]], data_FR %>% filter(fold_nr == fold)),
    gbm_pred_gamma = CA_gbm_prediction_perpoint(oos_sev_GBM_FR[[fold]][[3]], data_FR %>% filter(fold_nr == fold))
  ) %>% 
    mutate(gbm_tariff = gbm_pred_poiss*gbm_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_GBM_FR, file = 'tarif_GBM_FR')

tarif_GBM_NOR <- lapply(1:6, function(fold){
  bind_cols(
    data_NOR %>% filter(fold_nr == fold),
    gbm_pred_poiss = NC_gbm_prediction_perpoint(oos_freq_GBM_NOR[[fold]][[3]], data_NOR %>% filter(fold_nr == fold)),
    gbm_pred_gamma = CA_gbm_prediction_perpoint(oos_sev_GBM_NOR[[fold]][[3]], data_NOR %>% filter(fold_nr == fold))
  ) %>% 
    mutate(gbm_tariff = gbm_pred_poiss*gbm_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_GBM_NOR, file = 'tarif_GBM_NOR')

# -----
## ----- Adding the CANN GBM flexible -----
### ----- Optimal models -----

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

### ----- Adding out-of-sample predictions -----

tarif_CANN_GBM_flex_AUS <- lapply(1:6, function(fold){
  bind_cols(
    data_AUS %>% filter(fold_nr == fold),
    cann_pred_poiss = as.vector(CANN_model_predictions_perpoint(NC_object_AUS[[fold]], data_AUS %>% filter(fold_nr == fold))),
    cann_pred_gamma = as.vector(CANN_model_predictions_perpoint(CA_object_AUS[[fold]], data_AUS %>% filter(fold_nr == fold)))
  ) %>% 
    mutate(cann_tariff = cann_pred_poiss*cann_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_CANN_GBM_flex_AUS, file = 'tarif_CANN_GBM_flex_AUS')

tarif_CANN_GBM_flex_BE <- lapply(1:6, function(fold){
  bind_cols(
    data_BE_PC %>% filter(fold_nr == fold),
    cann_pred_poiss = as.vector(CANN_model_predictions_perpoint(NC_object_BE[[fold]], data_BE_PC %>% filter(fold_nr == fold))),
    cann_pred_gamma = as.vector(CANN_model_predictions_perpoint(CA_object_BE[[fold]], data_BE_PC %>% filter(fold_nr == fold)))
  ) %>% 
    mutate(cann_tariff = cann_pred_poiss*cann_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_CANN_GBM_flex_BE, file = 'tarif_CANN_GBM_flex_BE')

tarif_CANN_GBM_flex_FR <- lapply(1:6, function(fold){
  bind_cols(
    data_FR %>% filter(fold_nr == fold),
    cann_pred_poiss = as.vector(CANN_model_predictions_perpoint(NC_object_FR[[fold]], data_FR %>% filter(fold_nr == fold))),
    cann_pred_gamma = as.vector(CANN_model_predictions_perpoint(CA_object_FR[[fold]], data_FR %>% filter(fold_nr == fold)))
  ) %>% 
    mutate(cann_tariff = cann_pred_poiss*cann_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_CANN_GBM_flex_FR, file = 'tarif_CANN_GBM_flex_FR')

tarif_CANN_GBM_flex_NOR <- lapply(1:6, function(fold){
  bind_cols(
    data_NOR %>% filter(fold_nr == fold),
    cann_pred_poiss = as.vector(CANN_model_predictions_perpoint(NC_object_NOR[[fold]], data_NOR %>% filter(fold_nr == fold))),
    cann_pred_gamma = as.vector(CANN_model_predictions_perpoint(CA_object_NOR[[fold]], data_NOR %>% filter(fold_nr == fold)))
  ) %>% 
    mutate(cann_tariff = cann_pred_poiss*cann_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_CANN_GBM_flex_NOR, file = 'tarif_CANN_GBM_flex_NOR')

# -----
## ----- Adding the surrogate GLM -----
### ----- Read in surrogates -----

load('NC_CANN_GBM_flex_SURR_allFolds_AUS')
load('CA_CANN_GBM_flex_SURR_allFolds_AUS')

load('NC_CANN_GBM_flex_SURR_allFolds_BE')
load('CA_CANN_GBM_flex_SURR_allFolds_BE')

load('NC_CANN_GBM_flex_SURR_allFolds_FR')
load('CA_CANN_GBM_flex_SURR_allFolds_FR')

# Bin the data according to the surrogate GLM and make predictions
pred_with_surrogate <- function(surrogate, data, remap_variables = NULL){
  data_test_segm <- maidrr::segmentation(fx_vars = surrogate$pd_fx[names(surrogate$slct_feat)], 
                                         data = data, 
                                         type = 'ngroups', 
                                         values = surrogate$slct_feat)
  
  data_test_segm <- data_test_segm %>% select(c(ends_with("_"),'expo', 'nclaims', 'average'))
  
  # Specific for severity, some values are missing in the data and thus have no coefficient in the GLM
  # We take the closest existing bin instead
  if(!is.null(remap_variables)){
    
    for(var in remap_variables){
      
      GLM_splits <- surrogate$best_surr$R %>% 
        as.data.frame() %>% 
        rownames_to_column() %>% 
        pull(rowname) %>% as_tibble() %>% 
        filter(grepl(var, value)) %>% 
        mutate(value = gsub(var, '', value)) %>% 
        mutate(value = gsub("\\[", '', value)) %>% 
        mutate(value = gsub("\\]", '', value)) %>% 
        separate(value, c("start", "end"), ", ") %>% 
        mutate_at(c('start', 'end'), as.numeric) %>% 
        arrange(start) %>% 
        mutate(start.glm = start, end.glm = end)
      
      data_splits <- data_test_segm %>% pull(var) %>% unique() %>% as_tibble %>% 
        mutate(value = gsub(var, '', value)) %>% 
        mutate(value = gsub("\\[", '', value)) %>% 
        mutate(value = gsub("\\]", '', value)) %>% 
        separate(value, c("start", "end"), ", ") %>% 
        mutate_at(c('start', 'end'), as.numeric) %>% 
        arrange(start)
      
      split_mapping <- left_join(data_splits, GLM_splits, by = c('start', 'end')) %>% zoo::na.locf(na.rm = F) %>% zoo::na.locf(na.rm = F, fromLast = T) %>% 
        mutate(origin = paste0('[',start,', ',end,']')) %>% 
        mutate(new = paste0('[',start.glm,', ',end.glm,']')) 
      
      if(is.na(split_mapping$end[[1]])){
        split_mapping <- split_mapping %>% mutate(origin = start, new = start.glm) %>% select(origin, new) %>% 
          mutate(across(c(origin, new), factor))
      } else {
        split_mapping <- split_mapping %>% select(origin, new)
      }
      
      data_test_segm <- data_test_segm %>% 
        left_join(split_mapping, by = structure(names = var, .Data = 'origin')) %>% 
        mutate(!!var :=  new) %>% select(!c(new))
      
    }
  }
  
  predict(surrogate$best_surr, 
          data_test_segm, 
          type = 'response')
}

# Make data ordered and round selected values for surrogate use
data_AUS_ord <- data_AUS %>% mutate(VehValue = plyr::round_any(VehValue, 0.1))

data_BE_ord <- data_BE_PC %>% mutate(coverage = factor(coverage, ordered = T))

data_FR_ord <- data_FR %>% mutate(across(all_of(c('VehAge','DrivAge')), ~ 
                                           factor(., ordered = TRUE)))  %>% 
  mutate(Density = plyr::round_any(Density, 0.5)) %>% 
  mutate(BonusMalus = plyr::round_any(BonusMalus, 2))

### ----- Adding out-of-sample predictions -----

remap_AUS <- list(c('VehValue_'), NULL, NULL, c('VehValue_VehAge_', 'VehValue_VehBody_'), NULL, NULL)
tarif_SURR_GLM_AUS <- lapply(1:6, function(fold){
  bind_cols(
    data_AUS %>% filter(fold_nr == fold),
    surr_pred_poiss = as.vector(pred_with_surrogate(NC_CANN_GBM_flex_SURR_allFolds_AUS[[fold]], data_AUS_ord %>% filter(fold_nr == fold))),
    surr_pred_gamma = as.vector(pred_with_surrogate(CA_CANN_GBM_flex_SURR_allFolds_AUS[[fold]], data_AUS_ord %>% filter(fold_nr == fold), remap_variables = remap_AUS[[fold]]))
  ) %>% 
    mutate(surr_tariff = surr_pred_poiss*surr_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_SURR_GLM_AUS, file = 'tarif_SURR_GLM_AUS')

tarif_SURR_GLM_BE <- lapply(1:6, function(fold){
  bind_cols(
    data_BE_PC %>% filter(fold_nr == fold),
    cann_pred_poiss = as.vector(pred_with_surrogate(NC_CANN_GBM_flex_SURR_allFolds_BE[[fold]], data_BE_ord %>% filter(fold_nr == fold))),
    cann_pred_gamma = as.vector(pred_with_surrogate(CA_CANN_GBM_flex_SURR_allFolds_BE[[fold]], data_BE_ord %>% filter(fold_nr == fold)))
  ) %>% 
    mutate(cann_tariff = cann_pred_poiss*cann_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_SURR_GLM_BE, file = 'tarif_SURR_GLM_BE')

tarif_SURR_GLM_FR <- lapply(1:6, function(fold){
  bind_cols(
    data_FR %>% filter(fold_nr == fold),
    cann_pred_poiss = as.vector(pred_with_surrogate(NC_CANN_GBM_flex_SURR_allFolds_FR[[fold]], data_FR_ord %>% filter(fold_nr == fold))),
    cann_pred_gamma = as.vector(pred_with_surrogate(CA_CANN_GBM_flex_SURR_allFolds_FR[[fold]], data_FR_ord %>% filter(fold_nr == fold)))
  ) %>% 
    mutate(cann_tariff = cann_pred_poiss*cann_pred_gamma)
}) %>% do.call(rbind,.)
save(tarif_SURR_GLM_FR, file = 'tarif_SURR_GLM_FR')

# ----
## ------ Combining the tariff plans -----

lapply(c('tarif_GLM_AUS', 'tarif_GBM_AUS', 'tarif_CANN_GBM_flex_AUS', 'tarif_SURR_GLM_AUS'), load, .GlobalEnv)
lapply(c('tarif_GLM_BE', 'tarif_GBM_BE', 'tarif_CANN_GBM_flex_BE'), load, .GlobalEnv)
lapply(c('tarif_GLM_FR', 'tarif_GBM_FR', 'tarif_CANN_GBM_flex_FR'), load, .GlobalEnv)
lapply(c('tarif_GLM_NOR', 'tarif_GBM_NOR', 'tarif_CANN_GBM_flex_NOR'), load, .GlobalEnv)

tariff_plan_AUS <- reduce(list(data_AUS,
                               tarif_GLM_AUS %>% select(c(id, glm_pred_poiss, glm_pred_gamma, glm_tariff)),
                               tarif_GBM_AUS %>% select(c(id, gbm_pred_poiss, gbm_pred_gamma, gbm_tariff)),
                               tarif_CANN_GBM_flex_AUS %>% select(c(id, cann_pred_poiss, cann_pred_gamma, cann_tariff)),
                               tarif_SURR_GLM_AUS %>% select(c(id, surr_pred_poiss, surr_pred_gamma, surr_tariff))), 
                          dplyr::left_join, by = 'id') %>% 
  mutate(amount = nclaims*average) %>% 
  mutate_at(vars(amount), ~replace(., is.nan(.), 0))
save(tariff_plan_AUS, file = "tariff_plan_AUS")

tariff_plan_BE <- reduce(list(data_BE_PC,
                              tarif_GLM_BE %>% select(c(id, glm_pred_poiss, glm_pred_gamma, glm_tariff)),
                              tarif_GBM_BE %>% select(c(id, gbm_pred_poiss, gbm_pred_gamma, gbm_tariff)),
                              tarif_CANN_GBM_flex_BE %>% select(c(id, cann_pred_poiss, cann_pred_gamma, cann_tariff))), 
                         dplyr::left_join, by = 'id') %>% 
  mutate(amount = nclaims*average) %>% 
  mutate_at(vars(amount), ~replace(., is.nan(.), 0))
save(tariff_plan_BE, file = "tariff_plan_BE")

tariff_plan_FR <- reduce(list(data_FR,
                              tarif_GLM_FR %>% select(c(id, glm_pred_poiss, glm_pred_gamma, glm_tariff)),
                              tarif_GBM_FR %>% select(c(id, gbm_pred_poiss, gbm_pred_gamma, gbm_tariff)),
                              tarif_CANN_GBM_flex_FR %>% select(c(id, cann_pred_poiss, cann_pred_gamma, cann_tariff))), 
                         dplyr::left_join, by = 'id') %>% 
  mutate(amount = nclaims*average) %>% 
  mutate_at(vars(amount), ~replace(., is.nan(.), 0))
save(tariff_plan_FR, file = "tariff_plan_FR")

tariff_plan_NOR <- reduce(list(data_NOR,
                               tarif_GLM_NOR %>% select(c(id, glm_pred_poiss, glm_pred_gamma, glm_tariff)),
                               tarif_GBM_NOR %>% select(c(id, gbm_pred_poiss, gbm_pred_gamma, gbm_tariff)),
                               tarif_CANN_GBM_flex_NOR %>% select(c(id, cann_pred_poiss, cann_pred_gamma, cann_tariff))), 
                          dplyr::left_join, by = 'id') %>% 
  mutate(amount = nclaims*average) %>% 
  mutate_at(vars(amount), ~replace(., is.nan(.), 0))
save(tariff_plan_NOR, file = "tariff_plan_NOR")

# -----
# ----- LIFTS AND RELATIVITIES -----

# We extract the managerial insights from the tariff structures

## ----- Read in tariff plans ----- 

load('tariff_plan_AUS')
load('tariff_plan_BE')
load('tariff_plan_FR')
load('tariff_plan_NOR')

## ----- Balance ----- 

# We can do this for each data set
tariff_plan_AUS %>% group_by(fold_nr) %>% 
  summarise(tariff_glm = sum(glm_tariff),
            tariff_gbm = sum(gbm_tariff), 
            tariff_cann = sum(cann_tariff), 
            tariff_surr = sum(surr_tariff), 
            obs_losses = sum(amount)) %>% 
  mutate(balance_glm = tariff_glm / obs_losses,
         balance_gbm = tariff_gbm / obs_losses,
         balance_cann = tariff_cann / obs_losses,
         balance_surr = tariff_surr / obs_losses)

## ----- Lorenz Curves -----

# install.packages('ineq')
library(ineq)

# To name list items in a lapply loop
sn <- function(vector){
  setNames(vector,vector)
}

# Calculate Lorenz curve for all data sets and all models
LorenzCurve_alldata <- lapply(sn(c('AUS', 'BE', 'FR', 'NOR')), function(country){
  lapply(c('surr','cann','gbm','glm'), function(model){
    if(country == 'NOR' & model == 'surr'){
      return(NULL)
    } else {
      lcolc <- Lc(as.data.table(get(paste0('tariff_plan_',country))) %>% pull(paste0(model,'_tariff')))
      tibble(L = lcolc$L, p = lcolc$p, Uprob = c(1:length(lcolc$L)/length(lcolc$L)), model = model)
    }
  }) %>% do.call(rbind,.)
})
# save(LorenzCurve_alldata, file = 'LorenzCurve_alldata')
# load('LorenzCurve_alldata')

# There are to many data points in the curve to comfortably plot
LorenzCurve_alldata$AUS <- LorenzCurve_alldata$AUS %>% slice(seq(1,nrow(.),5))
LorenzCurve_alldata$BE <- LorenzCurve_alldata$BE %>% slice(seq(1,nrow(.),10))
LorenzCurve_alldata$FR <- LorenzCurve_alldata$FR %>% slice(seq(1,nrow(.),50))
LorenzCurve_alldata$NOR <- LorenzCurve_alldata$NOR %>% slice(seq(1,nrow(.),10))

linesize <- 0.8

# Make the plots of the Lorenz curves
LC_plot_AUS_CANNcomp <- ggplot(LorenzCurve_alldata$AUS %>% filter(model %in% c('gbm', 'cann'))) + 
  geom_segment(aes(x=0, xend=1, y=0, yend=1), linetype=4, lwd = 0.6) +
  geom_line(aes(x = Uprob, y = L, color = model), lwd = linesize) + 
  scale_color_manual(values=c("#52BDEC", "#00407A")) + 
  theme_bw() + xlab('Risk score') + ylab('Lorenz curve')

LC_plot_AUS_GLMcomp <- ggplot(LorenzCurve_alldata$AUS %>% filter(model %in% c('glm', 'surr'))) + 
  geom_segment(aes(x=0, xend=1, y=0, yend=1), linetype=4, lwd = 0.6) +
  geom_line(aes(x = Uprob, y = L, color = model), lwd = linesize) + 
  scale_color_manual(values=c("#DD8A2E", "#116E8A")) + 
  theme_bw() + xlab('Risk score') + ylab('Lorenz curve')

LC_plot_BE_CANNcomp <- ggplot(LorenzCurve_alldata$BE %>% filter(model %in% c('gbm', 'cann'))) + 
  geom_segment(aes(x=0, xend=1, y=0, yend=1), linetype=4, lwd = 0.6) +
  geom_line(aes(x = Uprob, y = L, color = model), lwd = linesize) + 
  scale_color_manual(values=c("#52BDEC", "#00407A")) + 
  theme_bw() + xlab('Risk score') + ylab('Lorenz curve')

LC_plot_BE_GLMcomp <- ggplot(LorenzCurve_alldata$BE %>% filter(model %in% c('glm', 'surr'))) + 
  geom_segment(aes(x=0, xend=1, y=0, yend=1), linetype=4, lwd = 0.6) +
  geom_line(aes(x = Uprob, y = L, color = model), lwd = linesize) + 
  scale_color_manual(values=c("#DD8A2E", "#116E8A")) + 
  theme_bw() + xlab('Risk score') + ylab('Lorenz curve')

LC_plot_FR_CANNcomp <- ggplot(LorenzCurve_alldata$FR %>% filter(model %in% c('gbm', 'cann'))) + 
  geom_segment(aes(x=0, xend=1, y=0, yend=1), linetype=4, lwd = 0.6) +
  geom_line(aes(x = Uprob, y = L, color = model), lwd = linesize) + 
  scale_color_manual(values=c("#52BDEC", "#00407A")) + 
  theme_bw() + xlab('Risk score') + ylab('Lorenz curve')

LC_plot_FR_GLMcomp <- ggplot(LorenzCurve_alldata$FR %>% filter(model %in% c('glm', 'surr'))) + 
  geom_segment(aes(x=0, xend=1, y=0, yend=1), linetype=4, lwd = 0.6) +
  geom_line(aes(x = Uprob, y = L, color = model), lwd = linesize) + 
  scale_color_manual(values=c("#DD8A2E", "#116E8A")) + 
  theme_bw() + xlab('Risk score') + ylab('Lorenz curve')

LC_plot_NOR_CANNcomp <- ggplot(LorenzCurve_alldata$NOR %>% filter(model %in% c('gbm', 'cann'))) + 
  geom_segment(aes(x=0, xend=1, y=0, yend=1), linetype=4, lwd = 0.6) +
  geom_line(aes(x = Uprob, y = L, color = model), lwd = linesize) + 
  scale_color_manual(values=c("#52BDEC", "#00407A")) + 
  theme_bw() + xlab('Risk score') + ylab('Lorenz curve')

#trow-away plot to get a consolidated legend
trowawayplot <- ggplot(LorenzCurve_alldata$AUS) + 
  geom_line(aes(x = Uprob, y = L, color = model), lwd = linesize) + 
  theme(legend.position="bottom", legend.direction="horizontal") + 
  scale_color_manual(name = 'Model', 
                     breaks = c("gbm", "cann", "glm", "surr"),
                     labels = c("GBM","CANN GBM flex", "GLM", "Surrogate GLM"), 
                     values = c("#52BDEC", "#00407A", "#DD8A2E", "#116E8A"))

# Set margins for plot combinations
margin_set <- c(0.2,0.2,0.2,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(LC_plot_AUS_CANNcomp + 
                              theme(legend.position = "none") + 
                              labs(subtitle = "Australia") +
                              theme(plot.margin = unit(margin_set + c(0,0,0,0), "cm"), plot.subtitle=element_text(size=12, hjust = 0)), 
                            LC_plot_AUS_GLMcomp +
                              theme(legend.position = "none") + 
                              theme(plot.margin = unit(margin_set + c(0,0,0,0), "cm"), plot.subtitle=element_text(size=12, hjust = 0)),
                            LC_plot_BE_CANNcomp +
                              theme(legend.position = "none") + 
                              labs(subtitle = "Belgium") +
                              theme(plot.margin = unit(margin_set + c(0,0,0,0), "cm"), plot.subtitle=element_text(size=12, hjust = 0)), 
                            LC_plot_BE_GLMcomp +
                              theme(legend.position = "none") + 
                              theme(plot.margin = unit(margin_set + c(0,0,0,0), "cm"), plot.subtitle=element_text(size=12, hjust = 0)),
                            LC_plot_FR_CANNcomp +
                              theme(legend.position = "none") + 
                              labs(subtitle = "French") +
                              theme(plot.margin = unit(margin_set + c(0,0,0,0), "cm"), plot.subtitle=element_text(size=12, hjust = 0)), 
                            LC_plot_FR_GLMcomp +
                              theme(legend.position = "none") + 
                              theme(plot.margin = unit(margin_set + c(0,0,0,0), "cm"), plot.subtitle=element_text(size=12, hjust = 0)),
                            LC_plot_NOR_CANNcomp +
                              theme(legend.position = "none") + 
                              labs(subtitle = "Norwegian") +
                              theme(plot.margin = unit(margin_set + c(0,0,0,0), "cm"), plot.subtitle=element_text(size=12, hjust = 0)),
                            align = "hv")

# Make a grid of all plots, with country names
allplotsgrid <- plot_grid(
  allplotslist[[1]], allplotslist[[3]], allplotslist[[5]], allplotslist[[7]],
  allplotslist[[2]], allplotslist[[4]], allplotslist[[6]], NULL,
  ncol = 4, rel_widths = c(0.25,0.25,0.25,0.25)
)

final_LorenzCurve_plot <- plot_grid(
  allplotsgrid,
  get_legend(trowawayplot),
  ncol = 1, rel_heights = c(0.9,0.1)
)

# final_LorenzCurve_plot

ggsave("LorenzCurve_plot.pdf",
       final_LorenzCurve_plot, 
       device = cairo_pdf,
       width = 14,
       height = 7,
       scale = 2.4,
       units = "cm")

# -----
# ----- END -----
# -----