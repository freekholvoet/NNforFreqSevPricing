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
#tensorflow::use_condaenv( "tf_noGpu")
#conda_python(envname = "tf_noGpu")

# Running on local Desktop PC
#tensorflow::use_condaenv( "my_env")
#conda_python(envname = "my_env")

# Disable graphical plot of model training (to much memory, can cause crash)
options(keras.view_metrics = FALSE)
#options(keras.view_metrics = TRUE)

# Number of significant digits
options(pillar.sigfig = 5)

## ----- Read in Data -----

load("data_AUS_prepared.RData")
load("data_BE_prepared.RData")
load("data_FR_prepared.RData")
load("data_NOR_prepared.RData")

# Load prepared data sets for neural networks
load("ClaimAmount_all_data_sets.RData")
load("NClaims_all_data_sets.RData")

# Read in Functions File
source("Functions.R")

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
# ----- AUTOENCODER TUNING -----
## ----- Tune the autoencoders for each country -----

# Countries and parameters (already chosen here)
countries <- expand.grid(c("AUS", "FR", "NOR", "BE"), c(5,10,15), c(500))

# Fit the autoencoders
AE_allDataSets <- lapply(1:nrow(countries), function(i){
  
  country <- countries[i,1]
  embedding_dim <- countries[i,2]
  max_epochs <- countries[i,3]
  cat_vars <- get(paste0("cat_",country))
  
  print(paste("Now training AutoEncoder with encoding dimension",embedding_dim,"for country",country,"with max epochs",max_epochs))
  
  lapply(1:6, function(fold){
    
    # Exclude some validation set from the data
    train_data <- lapply(cat_vars,function(var){
      get(paste0("data_",country)) %>% filter(fold_nr != fold) %>% 
        pull(var) %>% 
        as.data.table() %>% 
        one_hot(cols=".") %>% 
        data.matrix()
    })
    
    # Train autoencoder
    autoencoder_train(train_data_list = train_data,
                      encode_dimension = embedding_dim,
                      random_val_split = 0.2,
                      activation = 'softmax',
                      optimizer = 'nadam',
                      lossfunction = 'binary_crossentropy',
                      epochs = max_epochs,
                      batch = 1000,
                      verbose = 0)
  })
}) %>% setNames(paste0(countries[,1], countries[,2]))

# Look at the val-loss of each model (binary_crossentropy)
AE_loss <- sapply(AE_allDataSets,function(set){
  sapply(1:6, function(fold){set[[fold]]$AE_results$metrics$loss %>% last})
}) %>% format(digits = 3, scientific=FALSE)
save(AE_loss, file = 'AE_tuning_losses')

# Extract and save the weights from each autoencoder
AE_weights_AUS <- lapply(1:6, function(fold){
  get_weights(AE_allDataSets$AUS10[[fold]]$Encoder)
})

AE_weights_BE <- lapply(1:6, function(fold){
  get_weights(AE_allDataSets$BE5[[fold]]$Encoder)
})

AE_weights_FR <- lapply(1:6, function(fold){
  get_weights(AE_allDataSets$FR15[[fold]]$Encoder)
})

AE_weights_NOR <- lapply(1:6, function(fold){
  get_weights(AE_allDataSets$NOR10[[fold]]$Encoder)
})

save(AE_weights_AUS, file = 'AE_weights_AUS')
save(AE_weights_BE, file = 'AE_weights_BE')
save(AE_weights_FR, file = 'AE_weights_FR')
save(AE_weights_NOR, file = 'AE_weights_NOR')

## ----- Embedding normalization -----

# Function to adjust the weights from the trained autoencoder so the latent space is centered around zero
# This makes both the continuous variables and the encoded categorical variables on the same scale
scale_weights <- function(data, trained_AE_weights, cat_vars){
  
  # For each testfold, scale the weights from the trained autoencoder
  lapply(1:6, function(fold){
    
    # Select all data not in fold
    data_f <- data %>% filter(fold_nr != fold)
    
    # One-hot encode each categorical variable
    cat_OH <- lapply(cat_vars ,function(var){
      data_f %>% 
        pull(var) %>% 
        as.data.table() %>% 
        one_hot(cols=".") %>% 
        data.matrix()
    }) %>% do.call(cbind,.)
    
    # Calculate the vectors in the latent space
    latent_space <- sapply(1:nrow(cat_OH), function(datapoint){
      (t(as.matrix(trained_AE_weights[[1]][[1]])) %*% as.matrix(cat_OH[datapoint,]) + as.vector(trained_AE_weights[[1]][[2]]))
    }) %>% t
    
    # Calculate the mean and standard deviation of each encoded node
    AE_scaling_param <- sapply(1:ncol(latent_space),function(encoded_node){
      return(c(mu = mean(latent_space[,encoded_node]), sigma = sd(latent_space[,encoded_node])))
    }) %>% t
    
    # Calculate scaled weights and biases
    scaled_weights <- t(as.matrix(trained_AE_weights[[1]][[1]]))/as.vector(AE_scaling_param[,2] %>% as.matrix)
    scaled_bias <- (t(as.matrix(trained_AE_weights[[1]][[2]])) - as.vector(AE_scaling_param[,1] %>% as.matrix))/as.vector(AE_scaling_param[,2] %>% as.matrix)
    
    # Combine in list
    AE_weights_scaled <- list(scaled_weights %>% t,
                              scaled_bias %>% array)
    
    return(AE_weights_scaled)
  })
}

AE_weights_scaled_AUS <- scale_weights(data_AUS, AE_weights_AUS, cat_AUS)
AE_weights_scaled_BE <- scale_weights(data_BE, AE_weights_BE, cat_BE)
AE_weights_scaled_FR <- scale_weights(data_FR, AE_weights_FR, cat_FR)
AE_weights_scaled_NOR <- scale_weights(data_NOR, AE_weights_NOR, cat_NOR)

save(AE_weights_scaled_AUS, file = 'AE_weights_scaled_AUS')
save(AE_weights_scaled_BE, file = 'AE_weights_scaled_BE')
save(AE_weights_scaled_FR, file = 'AE_weights_scaled_FR')
save(AE_weights_scaled_NOR, file = 'AE_weights_scaled_NOR')

## ----- Read in already tuned AE weights -----

load('AE_weights_scaled_AUS')
load('AE_weights_scaled_BE')
load('AE_weights_scaled_FR')
load('AE_weights_scaled_NOR')

# -----
# ----- TUNING -----

# Here we tune all the network structures for each data set

## ----- Random grid construction -----

# Number of repeats to avoid local minima solutions
repeating_number = 1

# Total number of drawn tuning options
grid_size <- 40

### ---- Frequency grid -----

# Settings for the random grid seach tuning

# Fixed ranges
activation_h_range <- c("relu","sigmoid","softmax")

# Numerical ranges for geometric drawing
range <- tibble(
  batch = c(10000, 50000),
  layers = c(1,4),
  nodes = c(10,50),
  dropout = c(0.0001,0.1)
)

# Fixed values for the runs
optimizer <-  c("adam")
activation_out <- "exponential"
epochs <- 500
loss <- "poisson"


# Create a grid for random grid search from geometrically drawn values
NC_random_grid <- tibble(
  optimizer = rep(optimizer,grid_size),
  batch = runif(grid_size,log(range$batch[1]),log(range$batch[2])) %>% exp %>% round,
  activation_h = runif(grid_size,1,3) %>% round,
  layers = runif(grid_size,log(range$layers[1]),log(range$layers[2])) %>% exp %>% round,
  nodes = runif(grid_size,log(range$nodes[1]),log(range$nodes[2])) %>% exp %>% round,
  dropout = runif(grid_size,log(range$dropout[1]),log(range$dropout[2])) %>% exp %>% round(digits = 4),
  activation_out = rep(activation_out,grid_size),
  epochs = rep(epochs,grid_size),
  loss = rep(loss,grid_size)
) %>% rowwise %>% 
  mutate(hiddennodes = list(rep(nodes,layers))) %>% 
  ungroup %>% 
  select(!c("layers","nodes")) %>% 
  mutate(activation_h = activation_h_range[activation_h]) %>% 
  as.data.table
save(NC_random_grid,file="NC_random_grid.RData")

### ---- Severity grid -----

# Fixed ranges
activation_h_range <- c("relu","sigmoid","softmax")

# Numerical ranges for geometric drawing
range <- tibble(
  batch = c(200, 10000),
  layers = c(1,4),
  nodes = c(10,50),
  dropout = c(0.0001,0.1)
)

# Fixed values for the runs
optimizer <-  c("adam")
activation_out <- "exponential"
epochs <- 500
loss <- "gamma"

# Create a grid for random grid search from geometrically drawn values
CA_random_grid <- tibble(
  optimizer = rep(optimizer,grid_size),
  batch = runif(grid_size,log(range$batch[1]),log(range$batch[2])) %>% exp %>% round,
  activation_h = runif(grid_size,1,3) %>% round,
  layers = runif(grid_size,log(range$layers[1]),log(range$layers[2])) %>% exp %>% round,
  nodes = runif(grid_size,log(range$nodes[1]),log(range$nodes[2])) %>% exp %>% round,
  dropout = runif(grid_size,log(range$dropout[1]),log(range$dropout[2])) %>% exp %>% round(digits = 4),
  activation_out = rep(activation_out,grid_size),
  epochs = rep(epochs,grid_size),
  loss = rep(loss,grid_size)
) %>% rowwise %>% 
  mutate(hiddennodes = list(rep(nodes,layers))) %>% 
  ungroup %>% 
  select(!c("layers","nodes")) %>% 
  mutate(activation_h = activation_h_range[activation_h]) %>% 
  as.data.table
save(CA_random_grid,file="CA_random_grid.RData")

### ----- Read in tuning grid if already constructed -----

load("NC_random_grid.RData")
load("CA_random_grid.RData")

# -----
## ----- Australian MTPL tuning -----

# Neural network
NC_tuning_NN_AE_AUS <- alltestfolds_tuningrun(all_folds_data = NC_data_AUS,
                                              flags_list = NC_random_grid,
                                              flags_are_grid = TRUE,
                                              repeating_runs = 1,
                                              embedding = FALSE,
                                              embedding_output_dim = repeating_number,
                                              autoencoders = TRUE,
                                              autoencoder_trained = AE_weights_scaled_AUS,
                                              cat_vars = cat_AUS,
                                              CANN = FALSE,
                                              cann_variable = "prediction",
                                              trainable_output = FALSE,
                                              fold_var="fold_nr")
save(NC_tuning_NN_AE_AUS, file ="NC_tuning_NN_AE_AUS.RData")
Sys.time()

CA_tuning_NN_AE_AUS <- alltestfolds_tuningrun(all_folds_data = CA_data_AUS,
                                              flags_list = CA_random_grid,
                                              flags_are_grid = TRUE,
                                              repeating_runs = repeating_number,
                                              embedding = FALSE,
                                              embedding_output_dim = 1,
                                              autoencoders = TRUE,
                                              autoencoder_trained = AE_weights_scaled_AUS,
                                              cat_vars = cat_AUS,
                                              CANN = FALSE,
                                              cann_variable = "prediction",
                                              trainable_output = FALSE,
                                              fold_var="fold_nr")
save(CA_tuning_NN_AE_AUS, file ="CA_tuning_NN_AE_AUS.RData")
Sys.time()

# CANN with GLM input
NC_tuning_CANN_GLM_fixed_AE_AUS <- alltestfolds_tuningrun(all_folds_data = NC_data_AUS_GLM,
                                                      flags_list = NC_random_grid,
                                                      flags_are_grid = TRUE,
                                                      repeating_runs = repeating_number,
                                                      embedding = FALSE,
                                                      embedding_output_dim = 1,
                                                      autoencoders = TRUE,
                                                      autoencoder_trained = AE_weights_scaled_AUS,
                                                      cat_vars = cat_AUS,
                                                      CANN = TRUE,
                                                      cann_variable = "prediction",
                                                      trainable_output = FALSE,
                                                      fold_var="fold_nr")
save(NC_tuning_CANN_GLM_fixed_AE_AUS, file ="NC_tuning_CANN_GLM_fixed_AE_AUS.RData")
Sys.time()

CA_tuning_CANN_GLM_fixed_AE_AUS <- alltestfolds_tuningrun(all_folds_data = CA_data_AUS_GLM,
                                                      flags_list = CA_random_grid,
                                                      flags_are_grid = TRUE,
                                                      repeating_runs = repeating_number,
                                                      embedding = FALSE,
                                                      embedding_output_dim = 1,
                                                      autoencoders = TRUE,
                                                      autoencoder_trained = AE_weights_scaled_AUS,
                                                      cat_vars = cat_AUS,
                                                      CANN = TRUE,
                                                      cann_variable = "prediction",
                                                      trainable_output = FALSE,
                                                      fold_var="fold_nr")
save(CA_tuning_CANN_GLM_fixed_AE_AUS, file ="CA_tuning_CANN_GLM_fixed_AE_AUS.RData")
Sys.time()

NC_tuning_CANN_GLM_flex_AE_AUS <- alltestfolds_tuningrun(all_folds_data = NC_data_AUS_GLM,
                                                      flags_list = NC_random_grid,
                                                      flags_are_grid = TRUE,
                                                      repeating_runs = repeating_number,
                                                      embedding = FALSE,
                                                      embedding_output_dim = 1,
                                                      autoencoders = TRUE,
                                                      autoencoder_trained = AE_weights_scaled_AUS,
                                                      cat_vars = cat_AUS,
                                                      CANN = TRUE,
                                                      cann_variable = "prediction",
                                                      trainable_output = TRUE,
                                                      fold_var="fold_nr")
save(NC_tuning_CANN_GLM_flex_AE_AUS, file ="NC_tuning_CANN_GLM_flex_AE_AUS.RData")
Sys.time()

CA_tuning_CANN_GLM_flex_AE_AUS <- alltestfolds_tuningrun(all_folds_data = CA_data_AUS_GLM,
                                                      flags_list = CA_random_grid,
                                                      flags_are_grid = TRUE,
                                                      repeating_runs = repeating_number,
                                                      embedding = FALSE,
                                                      embedding_output_dim = 1,
                                                      autoencoders = TRUE,
                                                      autoencoder_trained = AE_weights_scaled_AUS,
                                                      cat_vars = cat_AUS,
                                                      CANN = TRUE,
                                                      cann_variable = "prediction",
                                                      trainable_output = TRUE,
                                                      fold_var="fold_nr")
save(CA_tuning_CANN_GLM_flex_AE_AUS, file ="CA_tuning_CANN_GLM_flex_AE_AUS.RData")
Sys.time()

# CANN with GBM input
NC_tuning_CANN_GBM_fixed_AE_AUS <- alltestfolds_tuningrun(all_folds_data = NC_data_AUS_GBM,
                                                          flags_list = NC_random_grid,
                                                          flags_are_grid = TRUE,
                                                          repeating_runs = repeating_number,
                                                          embedding = FALSE,
                                                          embedding_output_dim = 1,
                                                          autoencoders = TRUE,
                                                          autoencoder_trained = AE_weights_scaled_AUS,
                                                          cat_vars = cat_AUS,
                                                          CANN = TRUE,
                                                          cann_variable = "prediction",
                                                          trainable_output = FALSE,
                                                          fold_var="fold_nr")
save(NC_tuning_CANN_GBM_fixed_AE_AUS, file ="NC_tuning_CANN_GBM_fixed_AE_AUS.RData")
Sys.time()

CA_tuning_CANN_GBM_fixed_AE_AUS <- alltestfolds_tuningrun(all_folds_data = CA_data_AUS_GBM,
                                                          flags_list = CA_random_grid,
                                                          flags_are_grid = TRUE,
                                                          repeating_runs = repeating_number,
                                                          embedding = FALSE,
                                                          embedding_output_dim = 1,
                                                          autoencoders = TRUE,
                                                          autoencoder_trained = AE_weights_scaled_AUS,
                                                          cat_vars = cat_AUS,
                                                          CANN = TRUE,
                                                          cann_variable = "prediction",
                                                          trainable_output = FALSE,
                                                          fold_var="fold_nr")
save(CA_tuning_CANN_GBM_fixed_AE_AUS, file ="CA_tuning_CANN_GBM_fixed_AE_AUS.RData")
Sys.time()

NC_tuning_CANN_GBM_flex_AE_AUS <- alltestfolds_tuningrun(all_folds_data = NC_data_AUS_GBM,
                                                         flags_list = NC_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_AUS,
                                                         cat_vars = cat_AUS,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = TRUE,
                                                         fold_var="fold_nr")
save(NC_tuning_CANN_GBM_flex_AE_AUS, file ="NC_tuning_CANN_GBM_flex_AE_AUS.RData")
Sys.time()

CA_tuning_CANN_GBM_flex_AE_AUS <- alltestfolds_tuningrun(all_folds_data = CA_data_AUS_GBM,
                                                         flags_list = CA_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_AUS,
                                                         cat_vars = cat_AUS,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = TRUE,
                                                         fold_var="fold_nr")
save(CA_tuning_CANN_GBM_flex_AE_AUS, file ="CA_tuning_CANN_GBM_flex_AE_AUS.RData")
Sys.time()

## ----- Belgian MTPL tuning -----

# Neural network
NC_tuning_NN_AE_BE <- alltestfolds_tuningrun(all_folds_data = NC_data_BE,
                                              flags_list = NC_random_grid,
                                              flags_are_grid = TRUE,
                                              repeating_runs = 1,
                                              embedding = FALSE,
                                              embedding_output_dim = repeating_number,
                                              autoencoders = TRUE,
                                              autoencoder_trained = AE_weights_scaled_BE,
                                              cat_vars = cat_BE,
                                              CANN = FALSE,
                                              cann_variable = "prediction",
                                              trainable_output = FALSE,
                                              fold_var="fold_nr")
save(NC_tuning_NN_AE_BE, file ="NC_tuning_NN_AE_BE.RData")
Sys.time()

CA_tuning_NN_AE_BE <- alltestfolds_tuningrun(all_folds_data = CA_data_BE,
                                              flags_list = CA_random_grid,
                                              flags_are_grid = TRUE,
                                              repeating_runs = repeating_number,
                                              embedding = FALSE,
                                              embedding_output_dim = 1,
                                              autoencoders = TRUE,
                                              autoencoder_trained = AE_weights_scaled_BE,
                                              cat_vars = cat_BE,
                                              CANN = FALSE,
                                              cann_variable = "prediction",
                                              trainable_output = FALSE,
                                              fold_var="fold_nr")
save(CA_tuning_NN_AE_BE, file ="CA_tuning_NN_AE_BE.RData")
Sys.time()

# CANN with GLM input
NC_tuning_CANN_GLM_fixed_AE_BE <- alltestfolds_tuningrun(all_folds_data = NC_data_BE_GLM,
                                                          flags_list = NC_random_grid,
                                                          flags_are_grid = TRUE,
                                                          repeating_runs = repeating_number,
                                                          embedding = FALSE,
                                                          embedding_output_dim = 1,
                                                          autoencoders = TRUE,
                                                          autoencoder_trained = AE_weights_scaled_BE,
                                                          cat_vars = cat_BE,
                                                          CANN = TRUE,
                                                          cann_variable = "prediction",
                                                          trainable_output = FALSE,
                                                          fold_var="fold_nr")
save(NC_tuning_CANN_GLM_fixed_AE_BE, file ="NC_tuning_CANN_GLM_fixed_AE_BE.RData")
Sys.time()

CA_tuning_CANN_GLM_fixed_AE_BE <- alltestfolds_tuningrun(all_folds_data = CA_data_BE_GLM,
                                                          flags_list = CA_random_grid,
                                                          flags_are_grid = TRUE,
                                                          repeating_runs = repeating_number,
                                                          embedding = FALSE,
                                                          embedding_output_dim = 1,
                                                          autoencoders = TRUE,
                                                          autoencoder_trained = AE_weights_scaled_BE,
                                                          cat_vars = cat_BE,
                                                          CANN = TRUE,
                                                          cann_variable = "prediction",
                                                          trainable_output = FALSE,
                                                          fold_var="fold_nr")
save(CA_tuning_CANN_GLM_fixed_AE_BE, file ="CA_tuning_CANN_GLM_fixed_AE_BE.RData")
Sys.time()

NC_tuning_CANN_GLM_flex_AE_BE <- alltestfolds_tuningrun(all_folds_data = NC_data_BE_GLM,
                                                         flags_list = NC_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_BE,
                                                         cat_vars = cat_BE,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = TRUE,
                                                         fold_var="fold_nr")
save(NC_tuning_CANN_GLM_flex_AE_BE, file ="NC_tuning_CANN_GLM_flex_AE_BE.RData")
Sys.time()

CA_tuning_CANN_GLM_flex_AE_BE <- alltestfolds_tuningrun(all_folds_data = CA_data_BE_GLM,
                                                         flags_list = CA_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_BE,
                                                         cat_vars = cat_BE,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = TRUE,
                                                         fold_var="fold_nr")
save(CA_tuning_CANN_GLM_flex_AE_BE, file ="CA_tuning_CANN_GLM_flex_AE_BE.RData")
Sys.time()

# CANN with GBM input
NC_tuning_CANN_GBM_fixed_AE_BE <- alltestfolds_tuningrun(all_folds_data = NC_data_BE_GBM,
                                                          flags_list = NC_random_grid,
                                                          flags_are_grid = TRUE,
                                                          repeating_runs = repeating_number,
                                                          embedding = FALSE,
                                                          embedding_output_dim = 1,
                                                          autoencoders = TRUE,
                                                          autoencoder_trained = AE_weights_scaled_BE,
                                                          cat_vars = cat_BE,
                                                          CANN = TRUE,
                                                          cann_variable = "prediction",
                                                          trainable_output = FALSE,
                                                          fold_var="fold_nr")
save(NC_tuning_CANN_GBM_fixed_AE_BE, file ="NC_tuning_CANN_GBM_fixed_AE_BE.RData")
Sys.time()

CA_tuning_CANN_GBM_fixed_AE_BE <- alltestfolds_tuningrun(all_folds_data = CA_data_BE_GBM,
                                                          flags_list = CA_random_grid,
                                                          flags_are_grid = TRUE,
                                                          repeating_runs = repeating_number,
                                                          embedding = FALSE,
                                                          embedding_output_dim = 1,
                                                          autoencoders = TRUE,
                                                          autoencoder_trained = AE_weights_scaled_BE,
                                                          cat_vars = cat_BE,
                                                          CANN = TRUE,
                                                          cann_variable = "prediction",
                                                          trainable_output = FALSE,
                                                          fold_var="fold_nr")
save(CA_tuning_CANN_GBM_fixed_AE_BE, file ="CA_tuning_CANN_GBM_fixed_AE_BE.RData")
Sys.time()

NC_tuning_CANN_GBM_flex_AE_BE <- alltestfolds_tuningrun(all_folds_data = NC_data_BE_GBM,
                                                         flags_list = NC_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_BE,
                                                         cat_vars = cat_BE,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = TRUE,
                                                         fold_var="fold_nr")
save(NC_tuning_CANN_GBM_flex_AE_BE, file ="NC_tuning_CANN_GBM_flex_AE_BE.RData")
Sys.time()

CA_tuning_CANN_GBM_flex_AE_BE <- alltestfolds_tuningrun(all_folds_data = CA_data_BE_GBM,
                                                         flags_list = CA_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_BE,
                                                         cat_vars = cat_BE,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = TRUE,
                                                         fold_var="fold_nr")
save(CA_tuning_CANN_GBM_flex_AE_BE, file ="CA_tuning_CANN_GBM_flex_AE_BE.RData")
Sys.time()

## ----- French MTPL tuning -----

# Neural network
NC_tuning_NN_AE_FR <- alltestfolds_tuningrun(all_folds_data = NC_data_FR,
                                             flags_list = NC_random_grid,
                                             flags_are_grid = TRUE,
                                             repeating_runs = 1,
                                             embedding = FALSE,
                                             embedding_output_dim = repeating_number,
                                             autoencoders = TRUE,
                                             autoencoder_trained = AE_weights_scaled_FR,
                                             cat_vars = cat_FR,
                                             CANN = FALSE,
                                             cann_variable = "prediction",
                                             trainable_output = FALSE,
                                             fold_var="fold_nr")
save(NC_tuning_NN_AE_FR, file ="NC_tuning_NN_AE_FR.RData")
Sys.time()

CA_tuning_NN_AE_FR <- alltestfolds_tuningrun(all_folds_data = CA_data_FR,
                                             flags_list = CA_random_grid,
                                             flags_are_grid = TRUE,
                                             repeating_runs = repeating_number,
                                             embedding = FALSE,
                                             embedding_output_dim = 1,
                                             autoencoders = TRUE,
                                             autoencoder_trained = AE_weights_scaled_FR,
                                             cat_vars = cat_FR,
                                             CANN = FALSE,
                                             cann_variable = "prediction",
                                             trainable_output = FALSE,
                                             fold_var="fold_nr")
save(CA_tuning_NN_AE_FR, file ="CA_tuning_NN_AE_FR.RData")
Sys.time()

# CANN with GLM input
NC_tuning_CANN_GLM_fixed_AE_FR <- alltestfolds_tuningrun(all_folds_data = NC_data_FR_GLM,
                                                         flags_list = NC_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_FR,
                                                         cat_vars = cat_FR,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = FALSE,
                                                         fold_var="fold_nr")
save(NC_tuning_CANN_GLM_fixed_AE_FR, file ="NC_tuning_CANN_GLM_fixed_AE_FR.RData")
Sys.time()

CA_tuning_CANN_GLM_fixed_AE_FR <- alltestfolds_tuningrun(all_folds_data = CA_data_FR_GLM,
                                                         flags_list = CA_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_FR,
                                                         cat_vars = cat_FR,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = FALSE,
                                                         fold_var="fold_nr")
save(CA_tuning_CANN_GLM_fixed_AE_FR, file ="CA_tuning_CANN_GLM_fixed_AE_FR.RData")
Sys.time()

NC_tuning_CANN_GLM_flex_AE_FR <- alltestfolds_tuningrun(all_folds_data = NC_data_FR_GLM,
                                                        flags_list = NC_random_grid,
                                                        flags_are_grid = TRUE,
                                                        repeating_runs = repeating_number,
                                                        embedding = FALSE,
                                                        embedding_output_dim = 1,
                                                        autoencoders = TRUE,
                                                        autoencoder_trained = AE_weights_scaled_FR,
                                                        cat_vars = cat_FR,
                                                        CANN = TRUE,
                                                        cann_variable = "prediction",
                                                        trainable_output = TRUE,
                                                        fold_var="fold_nr")
save(NC_tuning_CANN_GLM_flex_AE_FR, file ="NC_tuning_CANN_GLM_flex_AE_FR.RData")
Sys.time()

CA_tuning_CANN_GLM_flex_AE_FR <- alltestfolds_tuningrun(all_folds_data = CA_data_FR_GLM,
                                                        flags_list = CA_random_grid,
                                                        flags_are_grid = TRUE,
                                                        repeating_runs = repeating_number,
                                                        embedding = FALSE,
                                                        embedding_output_dim = 1,
                                                        autoencoders = TRUE,
                                                        autoencoder_trained = AE_weights_scaled_FR,
                                                        cat_vars = cat_FR,
                                                        CANN = TRUE,
                                                        cann_variable = "prediction",
                                                        trainable_output = TRUE,
                                                        fold_var="fold_nr")
save(CA_tuning_CANN_GLM_flex_AE_FR, file ="CA_tuning_CANN_GLM_flex_AE_FR.RData")
Sys.time()

# CANN with GBM input
NC_tuning_CANN_GBM_fixed_AE_FR <- alltestfolds_tuningrun(all_folds_data = NC_data_FR_GBM,
                                                         flags_list = NC_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_FR,
                                                         cat_vars = cat_FR,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = FALSE,
                                                         fold_var="fold_nr")
save(NC_tuning_CANN_GBM_fixed_AE_FR, file ="NC_tuning_CANN_GBM_fixed_AE_FR.RData")
Sys.time()

CA_tuning_CANN_GBM_fixed_AE_FR <- alltestfolds_tuningrun(all_folds_data = CA_data_FR_GBM,
                                                         flags_list = CA_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_FR,
                                                         cat_vars = cat_FR,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = FALSE,
                                                         fold_var="fold_nr")
save(CA_tuning_CANN_GBM_fixed_AE_FR, file ="CA_tuning_CANN_GBM_fixed_AE_FR.RData")
Sys.time()

NC_tuning_CANN_GBM_flex_AE_FR <- alltestfolds_tuningrun(all_folds_data = NC_data_FR_GBM,
                                                        flags_list = NC_random_grid,
                                                        flags_are_grid = TRUE,
                                                        repeating_runs = repeating_number,
                                                        embedding = FALSE,
                                                        embedding_output_dim = 1,
                                                        autoencoders = TRUE,
                                                        autoencoder_trained = AE_weights_scaled_FR,
                                                        cat_vars = cat_FR,
                                                        CANN = TRUE,
                                                        cann_variable = "prediction",
                                                        trainable_output = TRUE,
                                                        fold_var="fold_nr")
save(NC_tuning_CANN_GBM_flex_AE_FR, file ="NC_tuning_CANN_GBM_flex_AE_FR.RData")
Sys.time()

CA_tuning_CANN_GBM_flex_AE_FR <- alltestfolds_tuningrun(all_folds_data = CA_data_FR_GBM,
                                                        flags_list = CA_random_grid,
                                                        flags_are_grid = TRUE,
                                                        repeating_runs = repeating_number,
                                                        embedding = FALSE,
                                                        embedding_output_dim = 1,
                                                        autoencoders = TRUE,
                                                        autoencoder_trained = AE_weights_scaled_FR,
                                                        cat_vars = cat_FR,
                                                        CANN = TRUE,
                                                        cann_variable = "prediction",
                                                        trainable_output = TRUE,
                                                        fold_var="fold_nr")
save(CA_tuning_CANN_GBM_flex_AE_FR, file ="CA_tuning_CANN_GBM_flex_AE_FR.RData")
Sys.time()


## ----- Norwegian MTPL tuning -----

# Neural network
NC_tuning_NN_AE_NOR <- alltestfolds_tuningrun(all_folds_data = NC_data_NOR,
                                              flags_list = NC_random_grid,
                                              flags_are_grid = TRUE,
                                              repeating_runs = 1,
                                              embedding = FALSE,
                                              embedding_output_dim = repeating_number,
                                              autoencoders = TRUE,
                                              autoencoder_trained = AE_weights_scaled_NOR,
                                              cat_vars = cat_NOR,
                                              CANN = FALSE,
                                              cann_variable = "prediction",
                                              trainable_output = FALSE,
                                              fold_var="fold_nr")
save(NC_tuning_NN_AE_NOR, file ="NC_tuning_NN_AE_NOR.RData")
Sys.time()

CA_tuning_NN_AE_NOR <- alltestfolds_tuningrun(all_folds_data = CA_data_NOR,
                                              flags_list = CA_random_grid,
                                              flags_are_grid = TRUE,
                                              repeating_runs = repeating_number,
                                              embedding = FALSE,
                                              embedding_output_dim = 1,
                                              autoencoders = TRUE,
                                              autoencoder_trained = AE_weights_scaled_NOR,
                                              cat_vars = cat_NOR,
                                              CANN = FALSE,
                                              cann_variable = "prediction",
                                              trainable_output = FALSE,
                                              fold_var="fold_nr")
save(CA_tuning_NN_AE_NOR, file ="CA_tuning_NN_AE_NOR.RData")
Sys.time()

# CANN with GLM input
NC_tuning_CANN_GLM_fixed_AE_NOR <- alltestfolds_tuningrun(all_folds_data = NC_data_NOR_GLM,
                                                          flags_list = NC_random_grid,
                                                          flags_are_grid = TRUE,
                                                          repeating_runs = repeating_number,
                                                          embedding = FALSE,
                                                          embedding_output_dim = 1,
                                                          autoencoders = TRUE,
                                                          autoencoder_trained = AE_weights_scaled_NOR,
                                                          cat_vars = cat_NOR,
                                                          CANN = TRUE,
                                                          cann_variable = "prediction",
                                                          trainable_output = FALSE,
                                                          fold_var="fold_nr")
save(NC_tuning_CANN_GLM_fixed_AE_NOR, file ="NC_tuning_CANN_GLM_fixed_AE_NOR.RData")
Sys.time()

CA_tuning_CANN_GLM_fixed_AE_NOR <- alltestfolds_tuningrun(all_folds_data = CA_data_NOR_GLM,
                                                          flags_list = CA_random_grid,
                                                          flags_are_grid = TRUE,
                                                          repeating_runs = repeating_number,
                                                          embedding = FALSE,
                                                          embedding_output_dim = 1,
                                                          autoencoders = TRUE,
                                                          autoencoder_trained = AE_weights_scaled_NOR,
                                                          cat_vars = cat_NOR,
                                                          CANN = TRUE,
                                                          cann_variable = "prediction",
                                                          trainable_output = FALSE,
                                                          fold_var="fold_nr")
save(CA_tuning_CANN_GLM_fixed_AE_NOR, file ="CA_tuning_CANN_GLM_fixed_AE_NOR.RData")
Sys.time()

NC_tuning_CANN_GLM_flex_AE_NOR <- alltestfolds_tuningrun(all_folds_data = NC_data_NOR_GLM,
                                                         flags_list = NC_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_NOR,
                                                         cat_vars = cat_NOR,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = TRUE,
                                                         fold_var="fold_nr")
save(NC_tuning_CANN_GLM_flex_AE_NOR, file ="NC_tuning_CANN_GLM_flex_AE_NOR.RData")
Sys.time()

CA_tuning_CANN_GLM_flex_AE_NOR <- alltestfolds_tuningrun(all_folds_data = CA_data_NOR_GLM,
                                                         flags_list = CA_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_NOR,
                                                         cat_vars = cat_NOR,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = TRUE,
                                                         fold_var="fold_nr")
save(CA_tuning_CANN_GLM_flex_AE_NOR, file ="CA_tuning_CANN_GLM_flex_AE_NOR.RData")
Sys.time()

# CANN with GBM input
NC_tuning_CANN_GBM_fixed_AE_NOR <- alltestfolds_tuningrun(all_folds_data = NC_data_NOR_GBM,
                                                          flags_list = NC_random_grid,
                                                          flags_are_grid = TRUE,
                                                          repeating_runs = repeating_number,
                                                          embedding = FALSE,
                                                          embedding_output_dim = 1,
                                                          autoencoders = TRUE,
                                                          autoencoder_trained = AE_weights_scaled_NOR,
                                                          cat_vars = cat_NOR,
                                                          CANN = TRUE,
                                                          cann_variable = "prediction",
                                                          trainable_output = FALSE,
                                                          fold_var="fold_nr")
save(NC_tuning_CANN_GBM_fixed_AE_NOR, file ="NC_tuning_CANN_GBM_fixed_AE_NOR.RData")
Sys.time()

CA_tuning_CANN_GBM_fixed_AE_NOR <- alltestfolds_tuningrun(all_folds_data = CA_data_NOR_GBM,
                                                          flags_list = CA_random_grid,
                                                          flags_are_grid = TRUE,
                                                          repeating_runs = repeating_number,
                                                          embedding = FALSE,
                                                          embedding_output_dim = 1,
                                                          autoencoders = TRUE,
                                                          autoencoder_trained = AE_weights_scaled_NOR,
                                                          cat_vars = cat_NOR,
                                                          CANN = TRUE,
                                                          cann_variable = "prediction",
                                                          trainable_output = FALSE,
                                                          fold_var="fold_nr")
save(CA_tuning_CANN_GBM_fixed_AE_NOR, file ="CA_tuning_CANN_GBM_fixed_AE_NOR.RData")
Sys.time()

NC_tuning_CANN_GBM_flex_AE_NOR <- alltestfolds_tuningrun(all_folds_data = NC_data_NOR_GBM,
                                                         flags_list = NC_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_NOR,
                                                         cat_vars = cat_NOR,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = TRUE,
                                                         fold_var="fold_nr")
save(NC_tuning_CANN_GBM_flex_AE_NOR, file ="NC_tuning_CANN_GBM_flex_AE_NOR.RData")
Sys.time()

CA_tuning_CANN_GBM_flex_AE_NOR <- alltestfolds_tuningrun(all_folds_data = CA_data_NOR_GBM,
                                                         flags_list = CA_random_grid,
                                                         flags_are_grid = TRUE,
                                                         repeating_runs = repeating_number,
                                                         embedding = FALSE,
                                                         embedding_output_dim = 1,
                                                         autoencoders = TRUE,
                                                         autoencoder_trained = AE_weights_scaled_NOR,
                                                         cat_vars = cat_NOR,
                                                         CANN = TRUE,
                                                         cann_variable = "prediction",
                                                         trainable_output = TRUE,
                                                         fold_var="fold_nr")
save(CA_tuning_CANN_GBM_flex_AE_NOR, file ="CA_tuning_CANN_GBM_flex_AE_NOR.RData")
Sys.time()

# -----
# ----- THE END -----
# -----

# Just testing to see if everything works on the server
flag_list <- list(
  dropout = 0.05282115,
  batch = 10956,
  hiddennodes = list(18,18),
  activation_h = "relu",
  activation_out = "exponential",
  optimizer = "adam",
  epochs = 500,
  loss = "poisson")

# Test a single run to see if everything is setup correctly
single_run_AE(NC_data_NOR[[1]], flag_list, autoencoder_trained = AE_weights_scaled_NOR[[1]], cat_vars = cat_NOR)
