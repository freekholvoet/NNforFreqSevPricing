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
                   "doParallel", "maidrr")
suppressMessages(packages <- lapply(used_packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x)
    library(x, character.only = TRUE)
  }
}))

## ---- Setup Keras and Tensorflow -----

# Point R to the appropriate instalation of Conda and Tensorflow
# use_python(---pythonlocation---)
# use_condaenv(---condalocation---)
# install_tensorflow(method = "conda", conda = ---condalocation---)

# Disable graphical plot of model training (to much memory, can cause crash)
options(keras.view_metrics = FALSE)
#options(keras.view_metrics = TRUE)

# Number of significant digits
options(pillar.sigfig = 5)

## ----- Read in Data -----

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

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
# ----- OUT OF SAMPLE -----

# Load on optimal tuning parameters for all netwerks
# Fit networks and calculate OOS performance

## ----- Read in optimal tuning -----

output_folder <- paste0(getwd(),"/Output")
file_names = lapply(list.files(path = output_folder, pattern = "\\NC_tuning"), function(x) paste0('Output/', x))

out = lapply(file_names,function(x){
  env = new.env()
  nm = load(x, envir = env)[1]
  objname = gsub(pattern = 'Output/', replacement = '', x = x, fixed = T)
  objname = gsub(pattern = 'prefix_pattern_|.RData', replacement = '', x = objname)
  # print(str(env[[nm]]))
  assign(objname, env[[nm]], envir = .GlobalEnv)
  0 # succeeded
} )

file_names = lapply(list.files(path = output_folder, pattern = "\\CA_tuning"), function(x) paste0('Output/', x))

out = lapply(file_names,function(x){
  env = new.env()
  nm = load(x, envir = env)[1]
  objname = gsub(pattern = 'Output/', replacement = '', x = x, fixed = T)
  objname = gsub(pattern = 'prefix_pattern_|.RData', replacement = '', x = objname)
  # print(str(env[[nm]]))
  assign(objname, env[[nm]], envir = .GlobalEnv)
  0 # succeeded
} )

### ----- Australian -----

# Neural network
AUS_NC_NN <- lapply(1:6, function(fold){
  NC_tuning_NN_AE_AUS %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
AUS_CA_NN <- lapply(1:6, function(fold){
  CA_tuning_NN_AE_AUS %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

# CANN with GLM input
AUS_NC_CANN_GLM_fixed <- lapply(1:6, function(fold){
  NC_tuning_CANN_GLM_fixed_AE_AUS %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
AUS_NC_CANN_GLM_flex <- lapply(1:6, function(fold){
  NC_tuning_CANN_GLM_flex_AE_AUS %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
AUS_CA_CANN_GLM_fixed <- lapply(1:6, function(fold){
  CA_tuning_CANN_GLM_fixed_AE_AUS %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
AUS_CA_CANN_GLM_flex <- lapply(1:6, function(fold){
  CA_tuning_CANN_GLM_flex_AE_AUS %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

# CANN with GBM input
AUS_NC_CANN_GBM_fixed <- lapply(1:6, function(fold){
  NC_tuning_CANN_GBM_fixed_AE_AUS %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
AUS_NC_CANN_GBM_flex <- lapply(1:6, function(fold){
  NC_tuning_CANN_GBM_flex_AE_AUS %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
AUS_CA_CANN_GBM_fixed <- lapply(1:6, function(fold){
  CA_tuning_CANN_GBM_fixed_AE_AUS %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
AUS_CA_CANN_GBM_flex <- lapply(1:6, function(fold){
  CA_tuning_CANN_GBM_flex_AE_AUS %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

save(AUS_NC_NN, AUS_CA_NN, 
     AUS_NC_CANN_GLM_fixed, AUS_NC_CANN_GLM_flex,
     AUS_CA_CANN_GLM_fixed, AUS_CA_CANN_GLM_flex, 
     AUS_NC_CANN_GBM_fixed, AUS_NC_CANN_GBM_flex,
     AUS_CA_CANN_GBM_fixed, AUS_CA_CANN_GBM_flex,
     file = "optimal_tuning_param_AUS")

### ----- Belgian -----

# Neural network
BE_NC_NN <- lapply(1:6, function(fold){
  NC_tuning_NN_AE_BE %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
BE_CA_NN <- lapply(1:6, function(fold){
  CA_tuning_NN_AE_BE %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

# CANN with GLM input
BE_NC_CANN_GLM_fixed <- lapply(1:6, function(fold){
  NC_tuning_CANN_GLM_fixed_AE_BE %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
BE_NC_CANN_GLM_flex <- lapply(1:6, function(fold){
  NC_tuning_CANN_GLM_flex_AE_BE %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
BE_CA_CANN_GLM_fixed <- lapply(1:6, function(fold){
  CA_tuning_CANN_GLM_fixed_AE_BE %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
BE_CA_CANN_GLM_flex <- lapply(1:6, function(fold){
  CA_tuning_CANN_GLM_flex_AE_BE %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

# CANN with GBM input
BE_NC_CANN_GBM_fixed <- lapply(1:6, function(fold){
  NC_tuning_CANN_GBM_fixed_AE_BE %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
BE_NC_CANN_GBM_flex <- lapply(1:6, function(fold){
  NC_tuning_CANN_GBM_flex_AE_BE %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
BE_CA_CANN_GBM_fixed <- lapply(1:6, function(fold){
  CA_tuning_CANN_GBM_fixed_AE_BE %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
BE_CA_CANN_GBM_flex <- lapply(1:6, function(fold){
  CA_tuning_CANN_GBM_flex_AE_BE %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

save(BE_NC_NN, BE_CA_NN, 
     BE_NC_CANN_GLM_fixed, BE_NC_CANN_GLM_flex,
     BE_CA_CANN_GLM_fixed, BE_CA_CANN_GLM_flex, 
     BE_NC_CANN_GBM_fixed, BE_NC_CANN_GBM_flex,
     BE_CA_CANN_GBM_fixed, BE_CA_CANN_GBM_flex,
     file = "optimal_tuning_param_BE")

### ----- French -----

# Neural network
FR_NC_NN <- lapply(1:6, function(fold){
  NC_tuning_NN_AE_FR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
FR_CA_NN <- lapply(1:6, function(fold){
  CA_tuning_NN_AE_FR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

# CANN with GLM input
FR_NC_CANN_GLM_fixed <- lapply(1:6, function(fold){
  NC_tuning_CANN_GLM_fixed_AE_FR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
FR_NC_CANN_GLM_flex <- lapply(1:6, function(fold){
  NC_tuning_CANN_GLM_flex_AE_FR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
FR_CA_CANN_GLM_fixed <- lapply(1:6, function(fold){
  CA_tuning_CANN_GLM_fixed_AE_FR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
FR_CA_CANN_GLM_flex <- lapply(1:6, function(fold){
  CA_tuning_CANN_GLM_flex_AE_FR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

# CANN with GBM input
FR_NC_CANN_GBM_fixed <- lapply(1:6, function(fold){
  NC_tuning_CANN_GBM_fixed_AE_FR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
FR_NC_CANN_GBM_flex <- lapply(1:6, function(fold){
  NC_tuning_CANN_GBM_flex_AE_FR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
FR_CA_CANN_GBM_fixed <- lapply(1:6, function(fold){
  CA_tuning_CANN_GBM_fixed_AE_FR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
FR_CA_CANN_GBM_flex <- lapply(1:6, function(fold){
  CA_tuning_CANN_GBM_flex_AE_FR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

save(FR_NC_NN, FR_CA_NN, 
     FR_NC_CANN_GLM_fixed, FR_NC_CANN_GLM_flex,
     FR_CA_CANN_GLM_fixed, FR_CA_CANN_GLM_flex, 
     FR_NC_CANN_GBM_fixed, FR_NC_CANN_GBM_flex,
     FR_CA_CANN_GBM_fixed, FR_CA_CANN_GBM_flex,
     file = "optimal_tuning_param_FR")

### ----- Norwegian -----

# Neural network
NOR_NC_NN <- lapply(1:6, function(fold){
  NC_tuning_NN_AE_NOR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
NOR_CA_NN <- lapply(1:6, function(fold){
  CA_tuning_NN_AE_NOR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

# CANN with GLM input
NOR_NC_CANN_GLM_fixed <- lapply(1:6, function(fold){
  NC_tuning_CANN_GLM_fixed_AE_NOR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
NOR_NC_CANN_GLM_flex <- lapply(1:6, function(fold){
  NC_tuning_CANN_GLM_flex_AE_NOR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
NOR_CA_CANN_GLM_fixed <- lapply(1:6, function(fold){
  CA_tuning_CANN_GLM_fixed_AE_NOR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
NOR_CA_CANN_GLM_flex <- lapply(1:6, function(fold){
  CA_tuning_CANN_GLM_flex_AE_NOR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

# CANN with GBM input
NOR_NC_CANN_GBM_fixed <- lapply(1:6, function(fold){
  NC_tuning_CANN_GBM_fixed_AE_NOR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
NOR_NC_CANN_GBM_flex <- lapply(1:6, function(fold){
  NC_tuning_CANN_GBM_flex_AE_NOR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
NOR_CA_CANN_GBM_fixed <- lapply(1:6, function(fold){
  CA_tuning_CANN_GBM_fixed_AE_NOR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})
NOR_CA_CANN_GBM_flex <- lapply(1:6, function(fold){
  CA_tuning_CANN_GBM_flex_AE_NOR %>% filter(test_fold == fold) %>% opt_tuning %>% mutate(activation_out = "exponential") %>% tuning_to_flags
})

save(NOR_NC_NN, NOR_CA_NN, 
     NOR_NC_CANN_GLM_fixed, NOR_NC_CANN_GLM_flex,
     NOR_CA_CANN_GLM_fixed, NOR_CA_CANN_GLM_flex, 
     NOR_NC_CANN_GBM_fixed, NOR_NC_CANN_GBM_flex,
     NOR_CA_CANN_GBM_fixed, NOR_CA_CANN_GBM_flex,
     file = "optimal_tuning_param_NOR")

# -----
## ----- Fitting models -----
### ----- Australian -----

# Regular neural networks
NC_NN_AUS <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_run_AE(fold_data = NC_data_AUS[[fold]], 
                                        flags_list = AUS_NC_NN[[fold]], 
                                        random_val_split = 0.2,
                                        autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                                        cat_vars = cat_AUS)$val_loss}) %>% mean
})
save(NC_NN_AUS, file = 'NC_NN_AUS')
CA_NN_AUS <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_run_AE(fold_data = CA_data_AUS[[fold]], 
                                        flags_list = AUS_CA_NN[[fold]], 
                                        random_val_split = 0.2,
                                        autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                                        cat_vars = cat_AUS)$val_loss}) %>% mean
})
save(CA_NN_AUS, file = 'CA_NN_AUS')

# CANN models with GLM input
NC_CANN_GLM_fixed_AUS <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_AUS_GLM[[fold]], 
                                             flags_list = AUS_NC_CANN_GLM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                                             cat_vars = cat_AUS)$val_loss}) %>% mean
})
save(NC_CANN_GLM_fixed_AUS, file = 'NC_CANN_GLM_fixed_AUS')
NC_CANN_GLM_flex_AUS <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_AUS_GLM[[fold]], 
                                             flags_list = AUS_NC_CANN_GLM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                                             cat_vars = cat_AUS, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(NC_CANN_GLM_flex_AUS, file = 'NC_CANN_GLM_flex_AUS')
CA_CANN_GLM_fixed_AUS <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_AUS_GLM[[fold]], 
                                             flags_list = AUS_CA_CANN_GLM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                                             cat_vars = cat_AUS)$val_loss}) %>% mean
})
save(CA_CANN_GLM_fixed_AUS, file = 'CA_CANN_GLM_fixed_AUS')
CA_CANN_GLM_flex_AUS <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_AUS_GLM[[fold]], 
                                             flags_list = AUS_CA_CANN_GLM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                                             cat_vars = cat_AUS, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(CA_CANN_GLM_flex_AUS, file = 'CA_CANN_GLM_flex_AUS')

# CANN models with GBM input
NC_CANN_GBM_fixed_AUS <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_AUS_GBM[[fold]], 
                                             flags_list = AUS_NC_CANN_GBM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                                             cat_vars = cat_AUS)$val_loss}) %>% mean
})
save(NC_CANN_GBM_fixed_AUS, file = 'NC_CANN_GBM_fixed_AUS')
NC_CANN_GBM_flex_AUS <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_AUS_GBM[[fold]], 
                                             flags_list = AUS_NC_CANN_GBM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                                             cat_vars = cat_AUS, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(NC_CANN_GBM_flex_AUS, file = 'NC_CANN_GBM_flex_AUS')
CA_CANN_GBM_fixed_AUS <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_AUS_GBM[[fold]], 
                                             flags_list = AUS_CA_CANN_GBM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                                             cat_vars = cat_AUS)$val_loss}) %>% mean
})
save(CA_CANN_GBM_fixed_AUS, file = 'CA_CANN_GBM_fixed_AUS')
CA_CANN_GBM_flex_AUS <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_AUS_GBM[[fold]], 
                                             flags_list = AUS_CA_CANN_GBM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_AUS[[fold]],
                                             cat_vars = cat_AUS, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(CA_CANN_GBM_flex_AUS, file = 'CA_CANN_GBM_flex_AUS')

# Load in every OOS file
load('NC_NN_AUS')
load('CA_NN_AUS')
load('NC_CANN_GLM_fixed_AUS')
load('NC_CANN_GLM_flex_AUS')
load('CA_CANN_GLM_fixed_AUS')
load('CA_CANN_GLM_flex_AUS')
load('NC_CANN_GBM_fixed_AUS')
load('NC_CANN_GBM_flex_AUS')
load('CA_CANN_GBM_fixed_AUS')
load('CA_CANN_GBM_flex_AUS')

# Combine in OOS table
oos_all_NN_AUS <- bind_rows(
  bind_cols(Fold = 1:6, Freq = NC_NN_AUS, Sev = CA_NN_AUS) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'AUS', Model = 'NN'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GLM_fixed_AUS, Sev = CA_CANN_GLM_fixed_AUS) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'AUS', Model = 'CANN GLM fixed'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GLM_flex_AUS, Sev = CA_CANN_GLM_flex_AUS) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'AUS', Model = 'CANN GLM flex'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GBM_fixed_AUS, Sev = CA_CANN_GBM_fixed_AUS) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'AUS', Model = 'CANN GBM fixed'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GBM_flex_AUS, Sev = CA_CANN_GBM_flex_AUS) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'AUS', Model = 'CANN GBM flex')
) %>% select(Model,Data,Problem,Fold,OOS)

save(oos_all_NN_AUS, file = "oos_all_NN_AUS")

### ----- Belgian -----

# Regular neural networks
NC_NN_BE <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_run_AE(fold_data = NC_data_BE[[fold]], 
                                        flags_list = BE_NC_NN[[fold]], 
                                        random_val_split = 0.2,
                                        autoencoder_trained = AE_weights_scaled_BE[[fold]],
                                        cat_vars = cat_BE)$val_loss}) %>% mean
})
save(NC_NN_BE, file = 'NC_NN_BE')
CA_NN_BE <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_run_AE(fold_data = CA_data_BE[[fold]], 
                                        flags_list = BE_CA_NN[[fold]], 
                                        random_val_split = 0.2,
                                        autoencoder_trained = AE_weights_scaled_BE[[fold]],
                                        cat_vars = cat_BE)$val_loss}) %>% mean
})
save(CA_NN_BE, file = 'CA_NN_BE')

# CANN models with GLM input
NC_CANN_GLM_fixed_BE <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_BE_GLM[[fold]], 
                                             flags_list = BE_NC_CANN_GLM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_BE[[fold]],
                                             cat_vars = cat_BE)$val_loss}) %>% mean
})
save(NC_CANN_GLM_fixed_BE, file = 'NC_CANN_GLM_fixed_BE')

NC_CANN_GLM_flex_BE <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_BE_GLM[[fold]], 
                                             flags_list = BE_NC_CANN_GLM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_BE[[fold]],
                                             cat_vars = cat_BE, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(NC_CANN_GLM_flex_BE, file = 'NC_CANN_GLM_flex_BE')

CA_CANN_GLM_fixed_BE <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_BE_GLM[[fold]], 
                                             flags_list = BE_CA_CANN_GLM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_BE[[fold]],
                                             cat_vars = cat_BE)$val_loss}) %>% mean
})
save(CA_CANN_GLM_fixed_BE, file = 'CA_CANN_GLM_fixed_BE')

CA_CANN_GLM_flex_BE <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_BE_GLM[[fold]], 
                                             flags_list = BE_CA_CANN_GLM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_BE[[fold]],
                                             cat_vars = cat_BE, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(CA_CANN_GLM_flex_BE, file = 'CA_CANN_GLM_flex_BE')

# CANN models with GBM input
NC_CANN_GBM_fixed_BE <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_BE_GBM[[fold]], 
                                             flags_list = BE_NC_CANN_GBM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_BE[[fold]],
                                             cat_vars = cat_BE)$val_loss}) %>% mean
})
save(NC_CANN_GBM_fixed_BE, file = 'NC_CANN_GBM_fixed_BE')

NC_CANN_GBM_flex_BE <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_BE_GBM[[fold]], 
                                             flags_list = BE_NC_CANN_GBM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_BE[[fold]],
                                             cat_vars = cat_BE, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(NC_CANN_GBM_flex_BE, file = 'NC_CANN_GBM_flex_BE')

CA_CANN_GBM_fixed_BE <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_BE_GBM[[fold]], 
                                             flags_list = BE_CA_CANN_GBM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_BE[[fold]],
                                             cat_vars = cat_BE)$val_loss}) %>% mean
})
save(CA_CANN_GBM_fixed_BE, file = 'CA_CANN_GBM_fixed_BE')

CA_CANN_GBM_flex_BE <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_BE_GBM[[fold]], 
                                             flags_list = BE_CA_CANN_GBM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_BE[[fold]],
                                             cat_vars = cat_BE, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(CA_CANN_GBM_flex_BE, file = 'CA_CANN_GBM_flex_BE')

# Combine in OOS table

load('NC_NN_BE')
load('CA_NN_BE')
load('NC_CANN_GLM_fixed_BE')
load('NC_CANN_GLM_flex_BE')
load('CA_CANN_GLM_fixed_BE')
load('CA_CANN_GLM_flex_BE')
load('NC_CANN_GBM_fixed_BE')
load('NC_CANN_GBM_flex_BE')
load('CA_CANN_GBM_fixed_BE')
load('CA_CANN_GBM_flex_BE')

oos_all_NN_BE <- bind_rows(
  bind_cols(Fold = 1:6, Freq = NC_NN_BE, Sev = CA_NN_BE) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'BE', Model = 'NN'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GLM_fixed_BE, Sev = CA_CANN_GLM_fixed_BE) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'BE', Model = 'CANN GLM fixed'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GLM_flex_BE, Sev = CA_CANN_GLM_flex_BE) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'BE', Model = 'CANN GLM flex'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GBM_fixed_BE, Sev = CA_CANN_GBM_fixed_BE) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'BE', Model = 'CANN GBM fixed'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GBM_flex_BE, Sev = CA_CANN_GBM_flex_BE) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'BE', Model = 'CANN GBM flex')
) %>% select(Model,Data,Problem,Fold,OOS)

save(oos_all_NN_BE, file = "oos_all_NN_BE")

### ----- French -----

# Regular neural networks
NC_NN_FR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_run_AE(fold_data = NC_data_FR[[fold]], 
                                        flags_list = FR_NC_NN[[fold]], 
                                        random_val_split = 0.2,
                                        autoencoder_trained = AE_weights_scaled_FR[[fold]],
                                        cat_vars = cat_FR)$val_loss}) %>% mean
})
save(NC_NN_FR, file = 'NC_NN_FR')
CA_NN_FR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_run_AE(fold_data = CA_data_FR[[fold]], 
                                        flags_list = FR_CA_NN[[fold]], 
                                        random_val_split = 0.2,
                                        autoencoder_trained = AE_weights_scaled_FR[[fold]],
                                        cat_vars = cat_FR)$val_loss}) %>% mean
})
save(CA_NN_FR, file = 'CA_NN_FR')

# CANN models with GLM input
NC_CANN_GLM_fixed_FR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_FR_GLM[[fold]], 
                                             flags_list = FR_NC_CANN_GLM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_FR[[fold]],
                                             cat_vars = cat_FR)$val_loss}) %>% mean
})
save(NC_CANN_GLM_fixed_FR, file = 'NC_CANN_GLM_fixed_FR')
NC_CANN_GLM_flex_FR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_FR_GLM[[fold]], 
                                             flags_list = FR_NC_CANN_GLM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_FR[[fold]],
                                             cat_vars = cat_FR, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(NC_CANN_GLM_flex_FR, file = 'NC_CANN_GLM_flex_FR')
CA_CANN_GLM_fixed_FR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_FR_GLM[[fold]], 
                                             flags_list = FR_CA_CANN_GLM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_FR[[fold]],
                                             cat_vars = cat_FR)$val_loss}) %>% mean
})
save(CA_CANN_GLM_fixed_FR, file = 'CA_CANN_GLM_fixed_FR')
CA_CANN_GLM_flex_FR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_FR_GLM[[fold]], 
                                             flags_list = FR_CA_CANN_GLM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_FR[[fold]],
                                             cat_vars = cat_FR, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(CA_CANN_GLM_flex_FR, file = 'CA_CANN_GLM_flex_FR')

# CANN models with GBM input
NC_CANN_GBM_fixed_FR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_FR_GBM[[fold]], 
                                             flags_list = FR_NC_CANN_GBM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_FR[[fold]],
                                             cat_vars = cat_FR)$val_loss}) %>% mean
})
save(NC_CANN_GBM_fixed_FR, file = 'NC_CANN_GBM_fixed_FR')
NC_CANN_GBM_flex_FR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_FR_GBM[[fold]], 
                                             flags_list = FR_NC_CANN_GBM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_FR[[fold]],
                                             cat_vars = cat_FR, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(NC_CANN_GBM_flex_FR, file = 'NC_CANN_GBM_flex_FR')
CA_CANN_GBM_fixed_FR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_FR_GBM[[fold]], 
                                             flags_list = FR_CA_CANN_GBM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_FR[[fold]],
                                             cat_vars = cat_FR)$val_loss}) %>% mean
})
save(CA_CANN_GBM_fixed_FR, file = 'CA_CANN_GBM_fixed_FR')
CA_CANN_GBM_flex_FR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_FR_GBM[[fold]], 
                                             flags_list = FR_CA_CANN_GBM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_FR[[fold]],
                                             cat_vars = cat_FR, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(CA_CANN_GBM_flex_FR, file = 'CA_CANN_GBM_flex_FR')

# Combine in OOS table

load('NC_NN_FR')
load('CA_NN_FR')
load('NC_CANN_GLM_fixed_FR')
load('NC_CANN_GLM_flex_FR')
load('CA_CANN_GLM_fixed_FR')
load('CA_CANN_GLM_flex_FR')
load('NC_CANN_GBM_fixed_FR')
load('NC_CANN_GBM_flex_FR')
load('CA_CANN_GBM_fixed_FR')
load('CA_CANN_GBM_flex_FR')

oos_all_NN_FR <- bind_rows(
  bind_cols(Fold = 1:6, Freq = NC_NN_FR, Sev = CA_NN_FR) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'FR', Model = 'NN'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GLM_fixed_FR, Sev = CA_CANN_GLM_fixed_FR) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'FR', Model = 'CANN GLM fixed'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GLM_flex_FR, Sev = CA_CANN_GLM_flex_FR) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'FR', Model = 'CANN GLM flex'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GBM_fixed_FR, Sev = CA_CANN_GBM_fixed_FR) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'FR', Model = 'CANN GBM fixed'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GBM_flex_FR, Sev = CA_CANN_GBM_flex_FR) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'FR', Model = 'CANN GBM flex')
) %>% select(Model,Data,Problem,Fold,OOS)

save(oos_all_NN_FR, file = "oos_all_NN_FR")

### ----- Norwegian -----

# Regular neural networks
NC_NN_NOR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_run_AE(fold_data = NC_data_NOR[[fold]], 
                                        flags_list = NOR_NC_NN[[fold]], 
                                        random_val_split = 0.2,
                                        autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                                        cat_vars = cat_NOR)$val_loss}) %>% mean
})
save(NC_NN_NOR, file = 'NC_NN_NOR')
CA_NN_NOR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_run_AE(fold_data = CA_data_NOR[[fold]], 
                                        flags_list = NOR_CA_NN[[fold]], 
                                        random_val_split = 0.2,
                                        autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                                        cat_vars = cat_NOR)$val_loss}) %>% mean
})
save(CA_NN_NOR, file = 'CA_NN_NOR')

# CANN models with GLM input
NC_CANN_GLM_fixed_NOR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_NOR_GLM[[fold]], 
                                             flags_list = NOR_NC_CANN_GLM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                                             cat_vars = cat_NOR)$val_loss}) %>% mean
})
save(NC_CANN_GLM_fixed_NOR, file = 'NC_CANN_GLM_fixed_NOR')
NC_CANN_GLM_flex_NOR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_NOR_GLM[[fold]], 
                                             flags_list = NOR_NC_CANN_GLM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                                             cat_vars = cat_NOR, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(NC_CANN_GLM_flex_NOR, file = 'NC_CANN_GLM_flex_NOR')

CA_CANN_GLM_fixed_NOR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_NOR_GLM[[fold]], 
                                             flags_list = NOR_CA_CANN_GLM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                                             cat_vars = cat_NOR)$val_loss}) %>% mean
})
save(CA_CANN_GLM_fixed_NOR, file = 'CA_CANN_GLM_fixed_NOR')

CA_CANN_GLM_flex_NOR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_NOR_GLM[[fold]], 
                                             flags_list = NOR_CA_CANN_GLM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                                             cat_vars = cat_NOR, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(CA_CANN_GLM_flex_NOR, file = 'CA_CANN_GLM_flex_NOR')

# CANN models with GBM input
NC_CANN_GBM_fixed_NOR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_NOR_GBM[[fold]], 
                                             flags_list = NOR_NC_CANN_GBM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                                             cat_vars = cat_NOR)$val_loss}) %>% mean
})
save(NC_CANN_GBM_fixed_NOR, file = 'NC_CANN_GBM_fixed_NOR')
NC_CANN_GBM_flex_NOR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = NC_data_NOR_GBM[[fold]], 
                                             flags_list = NOR_NC_CANN_GBM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                                             cat_vars = cat_NOR, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(NC_CANN_GBM_flex_NOR, file = 'NC_CANN_GBM_flex_NOR')

CA_CANN_GBM_fixed_NOR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_NOR_GBM[[fold]], 
                                             flags_list = NOR_CA_CANN_GBM_fixed[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                                             cat_vars = cat_NOR)$val_loss}) %>% mean
})
save(CA_CANN_GBM_fixed_NOR, file = 'CA_CANN_GBM_fixed_NOR')

CA_CANN_GBM_flex_NOR <- sapply(1:6, function(fold){
  sapply(1:3, function(x){single_CANN_run_AE(fold_data = CA_data_NOR_GBM[[fold]], 
                                             flags_list = NOR_CA_CANN_GBM_flex[[fold]], 
                                             random_val_split = 0.2,
                                             autoencoder_trained = AE_weights_scaled_NOR[[fold]],
                                             cat_vars = cat_NOR, 
                                             trainable_output = TRUE)$val_loss}) %>% mean
})
save(CA_CANN_GBM_flex_NOR, file = 'CA_CANN_GBM_flex_NOR')

# Combine in OOS table

load('NC_NN_NOR')
load('CA_NN_NOR')
load('NC_CANN_GLM_fixed_NOR')
load('NC_CANN_GLM_flex_NOR')
load('CA_CANN_GLM_fixed_NOR')
load('CA_CANN_GLM_flex_NOR')
load('NC_CANN_GBM_fixed_NOR')
load('NC_CANN_GBM_flex_NOR')
load('CA_CANN_GBM_fixed_NOR')
load('CA_CANN_GBM_flex_NOR')

oos_all_NN_NOR <- bind_rows(
  bind_cols(Fold = 1:6, Freq = NC_NN_NOR, Sev = CA_NN_NOR) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'NOR', Model = 'NN'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GLM_fixed_NOR, Sev = CA_CANN_GLM_fixed_NOR) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'NOR', Model = 'CANN GLM fixed'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GLM_flex_NOR, Sev = CA_CANN_GLM_flex_NOR) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'NOR', Model = 'CANN GLM flex'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GBM_fixed_NOR, Sev = CA_CANN_GBM_fixed_NOR) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'NOR', Model = 'CANN GBM fixed'),
  bind_cols(Fold = 1:6, Freq = NC_CANN_GBM_flex_NOR, Sev = CA_CANN_GBM_flex_NOR) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'NOR', Model = 'CANN GBM flex')
) %>% select(Model,Data,Problem,Fold,OOS)

save(oos_all_NN_NOR, file = "oos_all_NN_NOR")

# -----
## ----- Combine OOS in table -----

# Load OOS from GLM
load('oos_all_GLMs')

# Load OOS from GBM
load('oos_all_GBMs')

# Load OOS from all NN architectures
load("oos_all_NN_AUS")
load("oos_all_NN_BE")
load("oos_all_NN_FR")
load("oos_all_NN_NOR")

# Combine all OOS into one table
oos_all_models <- bind_rows(
  oos_all_GLMs, oos_all_GBMs,
  oos_all_NN_AUS,
  oos_all_NN_BE,
  oos_all_NN_FR,
  oos_all_NN_NOR
) %>% 
  mutate(Model = replace(Model, Model == 'NN', 'FFNN'))

### ----- Plot preparation -----

# Generate 5 colors
brewer.pal(5,"Set1")

# Select color for each model type (An empty color is added to make the legend better)
colorsNeeded <- tibble(Model = c('Binned GLM', 'GBM', 'FFNN', '', 'CANN GLM fixed', 'CANN GLM flex', 'CANN GBM fixed', 'CANN GBM flex')) %>% 
  bind_cols(Color = c("#E41A1C", "#377EB8", "#4DAF4A", '#FFFFFF', "#984EA3", "#984EA3", "#FF7F00","#FF7F00")) %>% 
  bind_cols(LineType = c(1,1,1,1,1,2,1,2))

# Add color to the OOS table and add linetype for different CANN models
oos_with_color <- oos_all_models %>% 
  left_join(colorsNeeded, by = c('Model')) %>% 
  mutate(Model = factor(Model, levels = c('Binned GLM', 'GBM', 'FFNN', '', 'CANN GLM fixed', 'CANN GLM flex', 'CANN GBM fixed', 'CANN GBM flex')))

# Number of points per fold in each data set, to reweight the average loss
point_per_fold  <- bind_rows(
  data_AUS %>% group_by(Fold = fold_nr) %>% summarise(Numbr = n()) %>% mutate(Data = 'AUS'),
  data_BE %>% group_by(Fold = fold_nr) %>% summarise(Numbr = n()) %>% mutate(Data = 'BE'),
  data_FR %>% group_by(Fold = fold_nr) %>% summarise(Numbr = n()) %>% mutate(Data = 'FR'),
  data_NOR %>% group_by(Fold = fold_nr) %>% summarise(Numbr = n()) %>% mutate(Data = 'NOR')
)

# Add min, mean and max over the test folds
oos_reshape <- oos_with_color %>% 
  left_join(point_per_fold, by = c('Data','Fold')) %>% 
  group_by(Model, Data, Problem, Color, LineType) %>% 
  summarise(min = min(OOS), mean = weighted.mean(OOS,Numbr), max = max(OOS))

# Vectors with colors and linetypes per model type
Colors <- colorsNeeded %>% pull(Color) %>% as.character() %>% set_names(colorsNeeded$Model)
LineTypes <- colorsNeeded %>% pull(LineType) %>% as.double() %>% set_names(colorsNeeded$Model)

### ---- Grid of all OOS plots -----

# Grid of each needed plot
all_problems <- expand_grid(Data = oos_with_color %>% pull(Data) %>% unique, 
                            Problem = oos_with_color %>% pull(Problem) %>% unique)

# Generate each plot
all_OOS_plots <- lapply(1:nrow(all_problems), function(set){
  
  oos_with_color %>% 
    filter(Data == all_problems$Data[[set]], Problem == all_problems$Problem[[set]]) %>% 
    ggplot(aes(x = Fold, y = OOS)) + 
    geom_point(size=2, aes(color = Model)) + 
    geom_line(aes(group = Model, color = Model, linetype = Model)) + 
    scale_color_manual(name = "Model", values = Colors, drop = FALSE, guide = guide_legend(override.aes = list(shape = NA))) + 
    scale_linetype_manual(name = "Model", values = LineTypes, drop = FALSE, guide = guide_legend(override.aes = list(shape = NA))) + 
    theme_bw() + 
    guides(col = guide_legend(nrow = 2, byrow = TRUE)) + 
    theme(legend.position="bottom", 
          legend.direction="horizontal") + 
    xlim('1','2','3','4','5','6') + xlab("Test set") + 
    scale_y_continuous(breaks= scales::pretty_breaks(n=5))
  
}) %>% setNames(paste0(all_problems$Data,'_',all_problems$Problem))

# Set margins for plot combinations
margin_set <- c(0.2,0.2,-1,0.2)

# Align all plots, so the size of the plot itself is equal for each plot, independend of axis sizes
allplotslist <- align_plots(all_OOS_plots$AUS_Freq + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Out-of-sample Poisson deviance") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12)), 
                            all_OOS_plots$AUS_Sev + theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL, subtitle = "Out-of-sample gamma deviance") + 
                              theme(plot.margin = unit(margin_set, "cm"), plot.subtitle=element_text(size=12)), 
                            all_OOS_plots$BE_Freq+ theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL) + theme(plot.margin = unit(margin_set, "cm")), 
                            all_OOS_plots$BE_Sev+ theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL) + theme(plot.margin = unit(margin_set, "cm")), 
                            all_OOS_plots$FR_Freq+ theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL) + theme(plot.margin = unit(margin_set, "cm")), 
                            all_OOS_plots$FR_Sev+ theme(legend.position = "none") + 
                              labs(x = NULL, y = NULL) + theme(plot.margin = unit(margin_set, "cm")), 
                            all_OOS_plots$NOR_Freq+ theme(legend.position = "none") + 
                              labs(y = NULL) + theme(plot.margin = unit(margin_set, "cm")), 
                            all_OOS_plots$NOR_Sev+ theme(legend.position = "none") + 
                              labs(y = NULL) + theme(plot.margin = unit(margin_set, "cm")), 
                            align = "hv")

# Make a grid of all plots, with country names
allplotsgrid <- plot_grid(#ggdraw() + draw_label(''),ggdraw() + draw_label("Out-of-sample Poisson Deviance", size = 12),ggdraw() + draw_label("Out-of-sample gamma Deviance", size = 12),
  ggdraw() + draw_label("Australian Data", angle = 90, size = 12),allplotslist[[1]], allplotslist[[2]],
  ggdraw() + draw_label("Belgian Data", angle = 90, size = 12),allplotslist[[3]], allplotslist[[4]],
  ggdraw() + draw_label("French Data", angle = 90, size = 12),allplotslist[[5]], allplotslist[[6]],
  ggdraw() + draw_label("Norwegian Data", angle = 90, size = 12),allplotslist[[7]], allplotslist[[8]],
  ncol = 3, rel_widths = c(0.05,0.47,0.47)#, rel_heights = c(0.05,0.24,0.24,0.24,0.24)
)

# Final plot grid, with legend
final_OOS_plot <- plot_grid(
  allplotsgrid,
  get_legend(all_OOS_plots$AUS_Freq),
  ncol = 1, rel_heights = c(0.9,0.1)
)

final_OOS_plot

# Save plot as PDF
ggsave("final_OOS_plot.pdf",
       final_OOS_plot, 
       device = 'pdf',
       width = 20,
       height = 25,
       scale = 1.2,
       units = "cm")

# ----- Other plots -----

min_oos_pC <- oos_reshape %>% group_by(Data,Problem) %>% summarise(overall_min = min(mean))

bp_oos <- oos_reshape %>% filter(Problem == 'Freq') %>% 
  ggplot(aes(x = Model)) + 
  geom_boxplot(
    aes(ymin = min, lower = min, middle = mean, upper = max, ymax = max, color = Model, linetype = Model),
    stat = "identity",
    width=0.1,
    position=position_dodge(0.2) 
  ) +  
  geom_hline(data = min_oos_pC %>% filter(Problem == 'Freq'), aes(yintercept = overall_min), linetype = 'dotted', col = 'red') + 
  facet_wrap(~Data, scales = 'free') + 
  theme_bw() +
  scale_color_manual(name = "Model", values = Colors) + 
  scale_linetype_manual(name = "Model", values = LineTypes) + 
  theme(legend.position="bottom", 
        legend.direction="horizontal") + 
  guides(col = guide_legend(nrow = 2, byrow = TRUE)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Save plot as PDF
ggsave("bp_oos.pdf",
       bp_oos, 
       device = 'pdf',
       width = 20,
       height = 20,
       scale = 2,
       units = "cm")


# -----
# ----- THE END -----
# -----
