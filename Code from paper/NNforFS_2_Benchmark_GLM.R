# ----- SETUP R -----

# All setup is done in this section
# Installing and loading all packages, setting up tensorflow and keras
# Reading in data and small data prep
# Define metrics for later use

## ----- Install packages needed -----

used_packages <- c("data.table", "tidyverse",
                   "doParallel","pbapply", "gam", "evtree")
suppressMessages(packages <- lapply(used_packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x)
    library(x, character.only = TRUE)
  }
}))

## ----- Read in data -----

# Data files input and results from Henckaerts et al. 2019
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Location of the extra data files
location_datasets <- "./Data"

load("data_AUS_prepared.RData")
load("data_FR_prepared.RData")
load("data_NOR_prepared.RData")

## ----- Prediction en loss functions -----

# Generic prediction function
predict_model <- function(object, newdata) UseMethod('predict_model')

# Prediction function for a GBM
predict_model.gbm <- function(object, newdata) {
  predict(object, newdata, n.trees = object$n.trees, type = 'response')
}

# Poisson deviance
dev_poiss <- function(ytrue, yhat) {
  -2 * mean(dpois(ytrue, yhat, log = TRUE) - dpois(ytrue, ytrue, log = TRUE), na.rm = TRUE)
}
# Gamma deviance
dev_gamma <- function(ytrue, yhat, wcase) {
  -2 * mean(wcase * (log(ytrue/yhat) - (ytrue - yhat)/yhat), na.rm = TRUE)
}

# -----
# ----- GAM -----

# Fitting the GAMs
# We fit a GAM on all possible selection of models
# Models are selected based on BIC value
# Interaction effects are tested for continuous variables

## ----- Fitting the GAM model without interaction -----

# Function to construct a GAM with all possible selections of variables, but without interaction effects
fitAllGAMs_noInteraction <- function(data, features, cat_features, 
                                     cont_features, response_var, offset_var = NULL, 
                                     family = 'gaussian', weights = NULL){
  
  # Define grid of possible combinations of variables
  comb <- do.call(tidyr::expand_grid, data.frame(replicate(length(features), 0:1, simplify=FALSE))) %>% set_names(features)
  
  all_fits <- sapply(1:nrow(comb), function(feat_set){
    
    # Define features to be used from combination matrix
    feat_select <- features[!!as.matrix(comb[feat_set,])]
    
    # Split up into cat and cont features
    cat_feat_select <- intersect(feat_select,cat_features)
    cont_feat_select <- intersect(feat_select,cont_features)
    
    
    # Fit a GAM with selected features
    gam(formula = as.formula(paste(paste(response_var,'~'),
                                   if(!is.null(offset_var)){paste('offset(',offset_var,')')},
                                   if(length(cat_feat_select) !=0){'+'}, 
                                   paste(cat_feat_select, collapse = ' + '), 
                                   if(length(cont_feat_select) !=0){'+'}, 
                                   paste(sapply(cont_feat_select,function(x){paste0('s(',x,')')}), collapse = ' + '),
                                   if(is.null(offset_var) & length(cat_feat_select) ==0 & length(cont_feat_select) ==0){1}
    )),
    data = data, 
    family=eval(family),
    weights = if (is.null(weights)) NULL else eval(as.symbol(weights))
    ) %>% BIC
  })
  
  # Add BIC values to each feature option in the grid
  bind_cols(comb, BIC_value = all_fits)
}

### ----- Frequency fitting ------

# Fit all GAMs for Australian data
all_fits_AUS_freq <- lapply(1:6,function(testset){
  fitAllGAMs_noInteraction(
    data = data_AUS %>% filter(fold_nr != testset),
    features = feat_AUS,
    cat_features = cat_AUS,
    cont_features = setdiff(feat_AUS,cat_AUS),
    response_var = 'nclaims',
    offset_var = 'log(expo)',
    family = 'poisson'
  )
})
save(all_fits_AUS_freq, file = 'all_fits_AUS_freq')

# Fit all GAMs for French data
all_fits_FR_freq <-  lapply(1:6,function(testset){
  fitAllGAMs_noInteraction(
    data = data_FR %>% filter(fold_nr != testset),
    features = feat_FR,
    cat_features = cat_FR,
    cont_features = setdiff(feat_FR,cat_FR),
    response_var = 'nclaims',
    offset_var = 'log(expo)',
    family = poisson
  )
})
save(all_fits_FR_freq, file = 'all_fits_FR_freq')

# Fit all GAMs for Norwegian data
all_fits_NOR_freq <-  lapply(1:6,function(testset){
  fitAllGAMs_noInteraction(
    data = data_NOR %>% filter(fold_nr != testset),
    features = feat_NOR,
    cat_features = cat_NOR,
    cont_features = setdiff(feat_NOR,cat_NOR),
    response_var = 'nclaims',
    offset_var = 'log(expo)',
    family = poisson
  )
})
save(all_fits_NOR_freq, file = 'all_fits_NOR_freq')

### ----- Severity fitting ------

# Fit all GAMs for Australian data
all_fits_AUS_sev <- lapply(1:6,function(testset){
  fitAllGAMs_noInteraction(
    data = data_AUS %>% filter(!is.na(average)) %>% filter(fold_nr != testset),
    features = feat_AUS,
    cat_features = cat_AUS,
    cont_features = setdiff(feat_AUS,cat_AUS),
    response_var = 'average',
    family = Gamma(link = "log"),
    weights = 'nclaims'
  )
})
save(all_fits_AUS_sev, file = 'all_fits_AUS_sev')

# Fit all GAMs for French data
all_fits_FR_sev <-  lapply(1:6,function(testset){
  fitAllGAMs_noInteraction(
    data = data_FR %>% filter(!is.na(average)) %>% filter(fold_nr != testset),
    features = feat_FR,
    cat_features = cat_FR,
    cont_features = setdiff(feat_FR,cat_FR),
    response_var = 'average',
    family = Gamma(link = "log"),
    weights = 'nclaims'
  )
})
save(all_fits_FR_sev, file = 'all_fits_FR_sev')

# Fit all GAMs for Norwegian data
all_fits_NOR_sev <-  lapply(1:6,function(testset){
  fitAllGAMs_noInteraction(
    data = data_NOR %>% filter(!is.na(average)) %>% filter(fold_nr != testset),
    features = feat_NOR,
    cat_features = cat_NOR,
    cont_features = setdiff(feat_NOR,cat_NOR),
    response_var = 'average',
    family = Gamma(link = "log"),
    weights = 'nclaims'
  )
})
save(all_fits_NOR_sev, file = 'all_fits_NOR_sev')

### ----- Load the fitted GAMS to get optimal fit -----
load('all_fits_AUS_freq')
load('all_fits_FR_freq')
load('all_fits_NOR_freq')
load('all_fits_AUS_sev')
load('all_fits_FR_sev')
load('all_fits_NOR_sev')

# Slice the best fit
opt_fit_GAM_freq_AUS <- lapply(all_fits_AUS_freq,function(x){x %>% slice(which.min(BIC_value))})
opt_fit_GAM_freq_FR <- lapply(all_fits_FR_freq,function(x){x %>% slice(which.min(BIC_value))})
opt_fit_GAM_freq_NOR <- lapply(all_fits_NOR_freq,function(x){x %>% slice(which.min(BIC_value))})

opt_fit_GAM_sev_AUS <- lapply(all_fits_AUS_sev,function(x){x %>% slice(which.min(BIC_value))})
opt_fit_GAM_sev_FR <- lapply(all_fits_FR_sev,function(x){x %>% slice(which.min(BIC_value))})
opt_fit_GAM_sev_NOR <- lapply(all_fits_NOR_sev,function(x){x %>% slice(which.min(BIC_value))})

# Function for single GAM based on best feature selection
fitSingleGAM <- function(data, optimal_features, cat_features, 
                         cont_features, response_var, offset_var = NULL, 
                         family = 'gaussian', weights = NULL){
  
  # Split up into cat and cont features
  cat_feat_select <- intersect(optimal_features,cat_features)
  cont_feat_select <- intersect(optimal_features,cont_features)
  
  # Fit a GAM with selected features
  gam(formula = as.formula(paste(paste(response_var,'~'),
                                 if(!is.null(offset_var)){paste('offset(',offset_var,')')},
                                 if(length(cat_feat_select) !=0){'+'}, 
                                 paste(cat_feat_select, collapse = ' + '), 
                                 if(length(cont_feat_select) !=0){'+'}, 
                                 paste(sapply(cont_feat_select,function(x){paste0('s(',x,')')}), collapse = ' + '),
                                 if(is.null(offset_var) & length(cat_feat_select) ==0 & length(cont_feat_select) ==0){1}
  )),
  data = data, 
  family=eval(family),
  weights = if (is.null(weights)) NULL else eval(as.symbol(weights))
  )
}

### ----- Best fits frequency -----

noInt_GAM_freq_AUS <- lapply(1:6,function(fold){
  fitSingleGAM(data = data_AUS %>% filter(fold_nr != fold),
               optimal_features = feat_AUS[!!as.matrix(opt_fit_GAM_freq_AUS[[fold]] %>% select(!BIC_value))],
               cat_features = cat_AUS,
               cont_features = setdiff(feat_AUS,cat_AUS),
               response_var = 'nclaims',
               offset_var = 'log(expo)',
               family = 'poisson')
})
noInt_GAM_freq_FR <- lapply(1:6,function(fold){
  fitSingleGAM(data = data_FR %>% filter(fold_nr != fold),
               optimal_features = feat_FR[!!as.matrix(opt_fit_GAM_freq_FR[[fold]] %>% select(!BIC_value))],
               cat_features = cat_FR,
               cont_features = setdiff(feat_FR,cat_FR),
               response_var = 'nclaims',
               offset_var = 'log(expo)',
               family = 'poisson')
})
noInt_GAM_freq_NOR <- lapply(1:6,function(fold){
  fitSingleGAM(data = data_NOR %>% filter(fold_nr != fold),
               optimal_features = feat_NOR[!!as.matrix(opt_fit_GAM_freq_NOR[[fold]] %>% select(!BIC_value))],
               cat_features = cat_NOR,
               cont_features = setdiff(feat_NOR,cat_NOR),
               response_var = 'nclaims',
               offset_var = 'log(expo)',
               family = 'poisson')
})

# REMARK: we already know the BIC value of the best fit. We re-fit the best model here, to have the model itself for later use

# BIC values for each fit
lapply(noInt_GAM_freq_AUS, function(x) x %>% BIC)
lapply(noInt_GAM_freq_FR, function(x) x %>% BIC)
lapply(noInt_GAM_freq_NOR, function(x) x %>% BIC) 

### ----- Best fits severity -----

noInt_GAM_sev_AUS <- lapply(1:6,function(fold){
  fitSingleGAM(data = data_AUS %>% filter(!is.na(average)) %>% filter(fold_nr != fold),
               optimal_features = feat_AUS[!!as.matrix(opt_fit_GAM_sev_AUS[[fold]] %>% select(!BIC_value))],
               cat_features = cat_AUS,
               cont_features = setdiff(feat_AUS,cat_AUS),
               response_var = 'average',
               family = Gamma(link = "log"),
               weights = 'nclaims')
})
noInt_GAM_sev_FR <- lapply(1:6,function(fold){
  fitSingleGAM(data = data_FR %>% filter(!is.na(average)) %>% filter(fold_nr != fold),
               optimal_features = feat_FR[!!as.matrix(opt_fit_GAM_sev_FR[[fold]] %>% select(!BIC_value))],
               cat_features = cat_FR,
               cont_features = setdiff(feat_FR,cat_FR),
               response_var = 'average',
               family = Gamma(link = "log"),
               weights = 'nclaims')
})
noInt_GAM_sev_NOR <- lapply(1:6,function(fold){
  fitSingleGAM(data = data_NOR %>% filter(!is.na(average)) %>% filter(nclaims > 0) %>% filter(fold_nr != fold),
               optimal_features = feat_NOR[!!as.matrix(opt_fit_GAM_sev_NOR[[fold]] %>% select(!BIC_value))],
               cat_features = cat_NOR,
               cont_features = setdiff(feat_NOR,cat_NOR),
               response_var = 'average',
               family = Gamma(link = "log"),
               weights = 'nclaims')
})

# BIC values for each fit
lapply(noInt_GAM_sev_AUS, function(x) x %>% BIC)
lapply(noInt_GAM_sev_FR, function(x) x %>% BIC)
lapply(noInt_GAM_sev_NOR, function(x) x %>% BIC) 

# -----
## ----- Adding interaction to optimal GAM -----

# Interaction is only looked at between continuous variables
# Only the French dataset has 2 continuous variables in the best selection

fit_Intr_freq_FR <- lapply(1:6, function(fold){
  
  FR_interactVar <- paste(intersect(feat_FR[!!as.matrix(opt_fit_GAM_freq_FR[[fold]] %>% select(!BIC_value))],setdiff(feat_FR,cat_FR)),collapse = ',')
  
  fitSingleGAM(data = data_FR %>% filter(fold_nr != fold),
               optimal_features = feat_FR[!!as.matrix(opt_fit_GAM_freq_FR[[fold]] %>% select(!BIC_value))] %>% append(FR_interactVar),
               cat_features = cat_FR,
               cont_features = setdiff(feat_FR,cat_FR) %>% append(FR_interactVar),
               response_var = 'nclaims',
               offset_var = 'log(expo)',
               family = 'poisson') %>% BIC
})

# The interaction leads to an increase in BIC, so it is not added to the model
# The previously fitted GAMs without interaction are the final fitted GAMs to be used in the next part
# -----
## ----- Deviance calculation -----

oos_GAM_freq_AUS <- lapply(1:6, function(fold){
  pred <- predict.Gam(noInt_GAM_freq_AUS[[fold]],newdata = data_AUS %>% filter(fold_nr == fold), type = 'response')
  dev_poiss(data_AUS %>% filter(fold_nr == fold) %>% pull(nclaims), pred)
})
oos_GAM_freq_FR <- lapply(1:6, function(fold){
  pred <- predict.Gam(noInt_GAM_freq_FR[[fold]],newdata = data_FR %>% filter(fold_nr == fold), type = 'response')
  dev_poiss(data_FR %>% filter(fold_nr == fold) %>% pull(nclaims), pred)
})
oos_GAM_freq_NOR <- lapply(1:6, function(fold){
  pred <- predict.Gam(noInt_GAM_freq_NOR[[fold]],newdata = data_NOR %>% filter(fold_nr == fold), type = 'response')
  dev_poiss(data_NOR %>% filter(fold_nr == fold) %>% pull(nclaims), pred)
})

oos_GAM_sev_AUS <- lapply(1:6, function(fold){
  pred <- predict.Gam(noInt_GAM_sev_AUS[[fold]],newdata = data_AUS %>% filter(!is.na(average)) %>% filter(fold_nr == fold), type = 'response')
  dev_gamma(data_AUS %>% filter(!is.na(average)) %>% filter(fold_nr == fold) %>% pull(average), 
            pred, 
            data_AUS %>% filter(!is.na(average)) %>% filter(fold_nr == fold) %>% pull(nclaims))
})
oos_GAM_sev_FR <- lapply(1:6, function(fold){
  pred <- predict.Gam(noInt_GAM_sev_FR[[fold]],newdata = data_FR %>% filter(!is.na(average)) %>% filter(fold_nr == fold), type = 'response')
  dev_gamma(data_FR %>% filter(!is.na(average)) %>% filter(fold_nr == fold) %>% pull(average), 
            pred, 
            data_FR %>% filter(!is.na(average)) %>% filter(fold_nr == fold) %>% pull(nclaims))
})
oos_GAM_sev_NOR <- lapply(1:6, function(fold){
  pred <- predict.Gam(noInt_GAM_sev_NOR[[fold]],newdata = data_NOR %>% filter(!is.na(average)) %>% filter(fold_nr == fold), type = 'response')
  dev_gamma(data_NOR %>% filter(!is.na(average)) %>% filter(fold_nr == fold) %>% pull(average), 
            pred, 
            data_NOR %>% filter(!is.na(average)) %>% filter(fold_nr == fold) %>% pull(nclaims))
})

# -----
# ----- EVOLUTION TREES -----

# Fitting the EvTrees on the insights of the GAMs
# The goal is to bin the continuous variables

## ----- EvTree setup -----

# Tuning parameters
alpha_tune <- c(seq(1,9.5,0.5),seq(10,95,5),seq(100,950,50))

# Hyperparameters
bucket_perc = 0.05
max_depth = 10
iterations = 500
seed = 12345

# Function to extract smooth effects for each continuous variable in the optimal fitted GAM
extract_smooths <- function(best_fitted_GAM, all_features, cat_features, optimal_features){
  
  # For each testfold extract smooth effects
  lapply(1:6, function(fold){
    
    # Extract number of continuous variables
    vars <- setdiff(all_features[!!as.matrix(optimal_features[[fold]] %>% select(!BIC_value))],cat_features)
    
    # For each variable, bind the values with the smooth effects
    lapply(1:length(vars), function(v){
      if(is_empty(best_fitted_GAM[[fold]]$smooth.frame)){return(NULL)} else {
        bind_cols(best_fitted_GAM[[fold]]$smooth.frame[,v], best_fitted_GAM[[fold]]$smooth[,v]) %>% 
          set_names(c('Value','Smooth')) %>% 
          arrange(Value) %>% 
          as_tibble %>%
          mutate(Value = as.numeric(Value)) %>% 
          group_by(Value) %>% 
          summarise(Smooth = mean(Smooth), Weights = n())
      }
    }) %>% setNames(vars)
  })
}

# Function for tuning the alpha value for frequency modelling
alpha_tuning_freq <- function(alpha_tune, input_data, smooth_variables, all_features, optimal_features){
  
  lapply(1:6, function(fold){
    # Select the correct data folds and smooth effects
    data_fold <- input_data %>% filter(fold_nr != fold)
    smooth_data <- smooth_variables[[fold]] 
    features_for_GLM <- optimal_features[[fold]]
    
    # Smooth effect will be NULL if no continuous effect is included in the optimal GAM
    if(is.null(smooth_data[[1]])){return(NULL)} else {
      
      # For each value of alpha, fit the EvTrees and the binned GLM, return the BIC
      sapply(alpha_tune, function(alpha){
        suppressWarnings({
          
          # Fit a tree with the same alpha for each continuous variable
          binning_tree <- lapply(smooth_data %>% names, function(var){
            evtree(formula = Smooth ~ Value,
                   data = smooth_data[[var]], 
                   weights = Weights, 
                   control = evtree.control(minbucket = bucket_perc*nrow(data_fold), 
                                            maxdepth = max_depth, 
                                            niterations = iterations, 
                                            alpha = alpha, 
                                            seed = seed)
            )
          }) %>% set_names(smooth_data %>% names)
        })
        
        # Predict with the fitted EvTrees to get the binned smooth effect
        pred_per_var <- lapply(smooth_data %>% names, function(var){
          tibble(!!var := predict(binning_tree[[var]], data_fold %>% select(Value = all_of(var)), type = 'response'))
        }) %>% bind_cols()
        
        binned_data <- data_fold %>% select(!(smooth_data %>% names)) %>% bind_cols(pred_per_var) %>% mutate_at((smooth_data %>% names), factor)

        # Fit a GLM on the binned data and return the BIC value
        glm_bic <- glm(as.formula(paste('nclaims ~', paste(all_features[!!as.matrix(features_for_GLM %>% select(!BIC_value))], collapse = ' + '))), 
                                  offset = log(expo), 
                                  data = binned_data, 
                                  family = poisson(link = "log")) %>% BIC
        
        return(c(alpha,glm_bic))
      }) %>% t %>% as_tibble %>% `colnames<-`(c("Alpha", "BIC"))
    }
  })
}

# Function for tuning the alpha value for severity modelling
alpha_tuning_sev <- function(alpha_tune, input_data, smooth_variables, all_features, optimal_features){
  
  lapply(1:6, function(fold){
    # Select the correct data folds and smooth effects
    data_fold <- input_data %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != fold)
    smooth_data <- smooth_variables[[fold]] 
    features_for_GLM <- optimal_features[[fold]]
    
    # Smooth effect will be NULL if no continuous effect is included in the optimal GAM
    if(is.null(smooth_data[[1]])){return(NULL)} else {
      
      # For each value of alpha, fit the EvTrees and the binned GLM, return the BIC
      sapply(alpha_tune, function(alpha){
        suppressWarnings({
          
          # Fit a tree with the same alpha for each continuous variable
          binning_tree <- lapply(smooth_data %>% names, function(var){
            evtree(formula = Smooth ~ Value,
                   data = smooth_data[[var]], 
                   weights = Weights, 
                   control = evtree.control(minbucket = bucket_perc*nrow(data_fold), 
                                            maxdepth = max_depth, 
                                            niterations = iterations, 
                                            alpha = alpha, 
                                            seed = seed)
            )
          }) %>% set_names(smooth_data %>% names)
        })
        
        # Predict with the fitted EvTrees to get the binned smooth effect
        pred_per_var <- lapply(smooth_data %>% names, function(var){
          tibble(!!var := predict(binning_tree[[var]], data_fold %>% select(Value = all_of(var)), type = 'response'))
        }) %>% bind_cols()
        
        binned_data <- data_fold %>% select(!(smooth_data %>% names)) %>% bind_cols(pred_per_var) %>% mutate_at((smooth_data %>% names), factor)
        
        # Fit a GLM on the binned data and return the BIC value
        glm_bic <- glm(as.formula(paste('average ~', paste(all_features[!!as.matrix(features_for_GLM %>% select(!BIC_value))], collapse = ' + '))), 
                                  data = binned_data, 
                                  family = Gamma(link = "log"),
                                  weights = nclaims) %>% BIC
        
        return(c(alpha,glm_bic))
      }) %>% t %>% as_tibble %>% `colnames<-`(c("Alpha", "BIC"))
    }
  })
}

## ----- AUS Alpha Tuning -----

smooth_freq_AUS <- extract_smooths(noInt_GAM_freq_AUS,feat_AUS,cat_AUS,opt_fit_GAM_freq_AUS)
smooth_sev_AUS <- extract_smooths(noInt_GAM_sev_AUS,feat_AUS,cat_AUS,opt_fit_GAM_sev_AUS)

alpha_tune_freq_AUS <- alpha_tuning_freq(alpha_tune, data_AUS, smooth_freq_AUS, feat_AUS, opt_fit_GAM_freq_AUS)
save(alpha_tune_freq_AUS, file = 'alpha_tune_freq_AUS')

opt_alpha_freq_AUS <- lapply(alpha_tune_freq_AUS,function(x){if(is.null(x)){NULL} else {x %>% slice(which.min(BIC))}})

alpha_tune_sev_AUS <- alpha_tuning_sev(alpha_tune, data_AUS, smooth_sev_AUS, feat_AUS, opt_fit_GAM_sev_AUS)
save(alpha_tune_sev_AUS, file = 'alpha_tune_sev_AUS')

opt_alpha_sev_AUS <- lapply(alpha_tune_sev_AUS,function(x){if(is.null(x)){NULL} else {x %>% slice(which.min(BIC))}})

## ----- FR Alpha Tuning -----

smooth_freq_FR <- extract_smooths(noInt_GAM_freq_FR,feat_FR,cat_FR,opt_fit_GAM_freq_FR)
smooth_sev_FR <- extract_smooths(noInt_GAM_sev_FR,feat_FR,cat_FR,opt_fit_GAM_sev_FR)

alpha_tune_freq_FR <- alpha_tuning_freq(alpha_tune, data_FR, smooth_freq_FR, feat_FR, opt_fit_GAM_freq_FR)
save(alpha_tune_freq_FR, file = 'alpha_tune_freq_FR')

opt_alpha_freq_FR <- lapply(alpha_tune_freq_FR,function(x){if(is.null(x)){NULL} else {x %>% slice(which.min(BIC))}})

alpha_tune_sev_FR <- alpha_tuning_sev(alpha_tune, data_FR, smooth_sev_FR, feat_FR, opt_fit_GAM_sev_FR)
save(alpha_tune_sev_FR, file = 'alpha_tune_sev_FR')

opt_alpha_sev_FR <- lapply(alpha_tune_sev_FR,function(x){if(is.null(x)){NULL} else {x %>% slice(which.min(BIC))}})

## ----- NOR Alpha Tuning -----

smooth_freq_NOR <- extract_smooths(noInt_GAM_freq_NOR,feat_NOR,cat_NOR,opt_fit_GAM_freq_NOR)
smooth_sev_NOR <- extract_smooths(noInt_GAM_sev_NOR,feat_NOR,cat_NOR,opt_fit_GAM_sev_NOR)

alpha_tune_freq_NOR <- alpha_tuning_freq(alpha_tune, data_NOR, smooth_freq_NOR, feat_NOR, opt_fit_GAM_freq_NOR)
save(alpha_tune_freq_NOR, file = 'alpha_tune_freq_NOR')

opt_alpha_freq_NOR <- lapply(alpha_tune_freq_NOR,function(x){if(is.null(x)){NULL} else {x %>% slice(which.min(BIC))}})

alpha_tune_sev_NOR <- alpha_tuning_sev(alpha_tune, data_NOR, smooth_sev_NOR, feat_NOR, opt_fit_GAM_sev_NOR)
save(alpha_tune_sev_NOR, file = 'alpha_tune_sev_NOR')

opt_alpha_sev_NOR <- lapply(alpha_tune_sev_NOR,function(x){if(is.null(x)){NULL} else {x %>% slice(which.min(BIC))}})

# -----
# ----- BINNING AND PREDICTING -----

# Create the final binned data based on optimal alpha
# Make GLM predictions to add to the data

## ----- Optimal Alpha read-in -----

load('alpha_tune_freq_AUS')
load('alpha_tune_sev_AUS')
load('alpha_tune_freq_FR')
load('alpha_tune_sev_FR')
load('alpha_tune_freq_NOR')
load('alpha_tune_sev_NOR')

## ----- Binning data and fitting GLM -----

# Function for binning and fitting the GLM with optimal alpha - Frequency
binning_opt_alpha_freq <- function(alpha, input_data, smooth_variables, all_features, optimal_features){
  
  lapply(1:6, function(fold){
    
    # Select the correct data folds and smooth effects
    data_fold <- input_data %>% filter(fold_nr != fold)
    smooth_data <- smooth_variables[[fold]] 
    features_for_GLM <- optimal_features[[fold]]
    
    alpha_opt <- alpha[[fold]]$Alpha
    
    # Smooth effect will be NULL if no continuous effect is included in the optimal GAM
    if(is.null(smooth_data[[1]])){
      
      # This means there are no continuous variables in the optimal GAM fit
      # We fit a GLM on the variables selected by the GAM
      
      features_for_GLM <- 
        if(length(all_features[!!as.matrix(features_for_GLM %>% select(!BIC_value))]) == 0){
          1
        } else {
          paste(all_features[!!as.matrix(features_for_GLM %>% select(!BIC_value))], collapse = ' + ')
        }
      
      # Fit the GLM on the original data, only features as selected by the GAM
      fit_glm <- glm(as.formula(paste('nclaims ~', features_for_GLM)), 
                     offset = log(expo), 
                     data = data_fold, 
                     family = poisson(link = "log"))
      
      # Combine the data with the predictions of the fitted GLM
      data_with_pred <- bind_cols(data_fold, prediction = predict(fit_glm, data_fold, type = 'response'))
      
      # Return the data with GLM predictions
      return(list(data_with_pred, fit_glm, NULL))
      
    } else {
      
      # Fit the EvTrees with optimal alpha for each continuous variable
      suppressWarnings({
        # Fit a tree with the same alpha for each continuous variable
        binning_tree <- lapply(smooth_data %>% names, function(var){
          evtree(formula = Smooth ~ Value,
                 data = smooth_data[[var]], 
                 weights = Weights, 
                 control = evtree.control(minbucket = bucket_perc*nrow(data_fold), 
                                          maxdepth = max_depth, 
                                          niterations = iterations, 
                                          alpha = alpha_opt, 
                                          seed = seed)
          )
        }) %>% set_names(smooth_data %>% names)
      })
      
      # Predict with the fitted EvTrees to get the binned smooth effect
      pred_per_var <- lapply(smooth_data %>% names, function(var){
        tibble(!!var := predict(binning_tree[[var]], data_fold %>% select(Value = all_of(var)), type = 'response'))
      }) %>% bind_cols()
      
      binned_data <- data_fold %>% select(!(smooth_data %>% names)) %>% bind_cols(pred_per_var) %>% mutate_at((smooth_data %>% names), factor)
      
      features_for_GLM <- 
        if(length(all_features[!!as.matrix(features_for_GLM %>% select(!BIC_value))]) == 0){
          1
        } else {
          paste(all_features[!!as.matrix(features_for_GLM %>% select(!BIC_value))], collapse = ' + ')
        }
      
      # Fit the GLM on the binned data
      fit_glm <- glm(as.formula(paste('nclaims ~', features_for_GLM)), 
                     offset = log(expo), 
                     data = binned_data, 
                     family = poisson(link = "log"))
      
      # Combine the binned data with the predictions of the binned_GLM
      binned_with_pred <- bind_cols(binned_data, prediction = predict(fit_glm, binned_data, type = 'response'))
      
      # Return the binned data set with the predictions of the binned GLM
      return(list(binned_with_pred, fit_glm, binning_tree))
    }
  })
}

# Function for binning and fitting the GLM with optimal alpha - Severity
binning_opt_alpha_sev <- function(alpha, input_data, smooth_variables, all_features, optimal_features){
  
  lapply(1:6, function(fold){
    
    # Select the correct data folds and smooth effects
    data_fold <- input_data %>% filter(nclaims > 0) %>% filter(!is.na(average)) %>% filter(fold_nr != fold)
    smooth_data <- smooth_variables[[fold]] 
    features_for_GLM <- optimal_features[[fold]]
    
    alpha_opt <- alpha[[fold]]$Alpha
    
    # Smooth effect will be NULL if no continuous effect is included in the optimal GAM
    if(is.null(smooth_data[[1]])){
      
      # This means there are no continuous variables in the optimal GAM fit
      # We fit a GLM on the variables selected by the GAM
      
      features_for_GLM <- 
        if(length(all_features[!!as.matrix(features_for_GLM %>% select(!BIC_value))]) == 0){
          1
        } else {
          paste(all_features[!!as.matrix(features_for_GLM %>% select(!BIC_value))], collapse = ' + ')
        }
      
      # Fit the GLM on the original data, only features as selected by the GAM
      fit_glm <- glm(as.formula(paste('average ~', features_for_GLM)),  
                     data = data_fold, 
                     family = Gamma(link = "log"),
                     weights = nclaims)
      
      # Combine the data with the predictions of the fitted GLM
      data_with_pred <- bind_cols(data_fold, prediction = predict(fit_glm, data_fold, type = 'response'))
      
      # Return the data with GLM predictions (add a NULL pointer for the missing EvTree, this makes it easier later)
      return(list(data_with_pred, fit_glm, NULL))
      
    } else {
      
      # Fit the EvTrees with optimal alpha for each continuous variable
      suppressWarnings({
        # Fit a tree with the same alpha for each continuous variable
        binning_tree <- lapply(smooth_data %>% names, function(var){
          evtree(formula = Smooth ~ Value,
                 data = smooth_data[[var]], 
                 weights = Weights, 
                 control = evtree.control(minbucket = bucket_perc*nrow(data_fold), 
                                          maxdepth = max_depth, 
                                          niterations = iterations, 
                                          alpha = alpha_opt, 
                                          seed = seed)
          )
        }) %>% set_names(smooth_data %>% names)
      })
      
      # Predict with the fitted EvTrees to get the binned smooth effect
      pred_per_var <- lapply(smooth_data %>% names, function(var){
        tibble(!!var := predict(binning_tree[[var]], data_fold %>% select(Value = all_of(var)), type = 'response'))
      }) %>% bind_cols()
      
      binned_data <- data_fold %>% select(!(smooth_data %>% names)) %>% bind_cols(pred_per_var) %>% mutate_at((smooth_data %>% names), factor)
      
      features_for_GLM <- 
        if(length(all_features[!!as.matrix(features_for_GLM %>% select(!BIC_value))]) == 0){
          1
        } else {
          paste(all_features[!!as.matrix(features_for_GLM %>% select(!BIC_value))], collapse = ' + ')
        }
      
      # Fit the GLM on the binned data
      fit_glm <- glm(as.formula(paste('average ~', features_for_GLM)), 
                     data = binned_data, 
                     family = Gamma(link = "log"),
                     weights = nclaims)
      
      # Combine the binned data with the predictions of the binned_GLM
      binned_with_pred <- bind_cols(binned_data, prediction = predict(fit_glm, binned_data, type = 'response'))
      
      # Return the binned data set with the predictions of the binned GLM
      return(list(binned_with_pred, fit_glm, binning_tree))
    }
  })
}

# Hyperparameters (Same as for tuning)
bucket_perc = 0.05
max_depth = 10
iterations = 500
seed = 12345

binned_freq_AUS <- binning_opt_alpha_freq(opt_alpha_freq_AUS, data_AUS, smooth_freq_AUS, feat_AUS, opt_fit_GAM_freq_AUS)
binned_sev_AUS <- binning_opt_alpha_sev(opt_alpha_sev_AUS, data_AUS, smooth_sev_AUS, feat_AUS, opt_fit_GAM_sev_AUS)

binned_freq_FR <- binning_opt_alpha_freq(opt_alpha_freq_FR, data_FR, smooth_freq_FR, feat_FR, opt_fit_GAM_freq_FR)
binned_sev_FR <- binning_opt_alpha_sev(opt_alpha_sev_FR, data_FR, smooth_sev_FR, feat_FR, opt_fit_GAM_sev_FR)

binned_freq_NOR <- binning_opt_alpha_freq(opt_alpha_freq_NOR, data_NOR, smooth_freq_NOR, feat_NOR, opt_fit_GAM_freq_NOR)
binned_sev_NOR <- binning_opt_alpha_sev(opt_alpha_sev_NOR, data_NOR, smooth_sev_NOR, feat_NOR, opt_fit_GAM_sev_NOR)

save(binned_freq_AUS, file = 'binned_freq_AUS')
save(binned_sev_AUS, file = 'binned_sev_AUS')
save(binned_freq_FR, file = 'binned_freq_FR')
save(binned_sev_FR, file = 'binned_sev_FR')
save(binned_freq_NOR, file = 'binned_freq_NOR')
save(binned_sev_NOR, file = 'binned_sev_NOR')

# -----
# ----- OUT OF SAMPLE PREDICTION -----

# Bin data with out of sample models
# Make final predictions and add to data. 
# We get 6 complete data sets were each time one fold has out of sample predictions and one fold in sample

## ----- Load fitted binning models -----

load('binned_freq_AUS')
load('binned_sev_AUS')
load('binned_freq_FR')
load('binned_sev_FR')
load('binned_freq_NOR')
load('binned_sev_NOR')

## ----- Predicting out of sample -----

add_oos_predictions <- function(original_data, fitted_models){
  
  # For each fold, add out of sample prediction
  lapply(1:6, function(fold){
    
    # Select data and model
    data_f <- original_data %>% filter(fold_nr == fold)
    model_set <- fitted_models[[fold]]
    
    # Predict with model on out of sample data
    if(is.null(model_set[[3]])){

      # GLM without EvTree binning
      oos_pred <- bind_cols(data_f, prediction = predict(model_set[[2]], data_f, type = 'response'))
      
      # Return full data set with oos predictions for the selected fold and in sample predictions for the other folds
      bind_rows(oos_pred, model_set[[1]]) %>% arrange(id)
      
    } else {
      # Binning with EvTree
      pred_per_var <- lapply(names(model_set[[3]]), function(var){
        tibble(!!var := predict(model_set[[3]][[var]], data_f %>% select(Value = all_of(var)), type = 'response'))
      }) %>% bind_cols()
      
      binned_data <- data_f %>% select(!(names(model_set[[3]]))) %>% bind_cols(pred_per_var) %>% mutate_at((names(model_set[[3]])), factor)
      
      # Predicting with GLM
      oos_pred <- bind_cols(binned_data, prediction = predict(model_set[[2]], binned_data, type = 'response'))
      
      # Return full data set with oos predictions for the selected fold and in sample predictions for the other folds
      bind_rows(oos_pred, model_set[[1]]) %>% arrange(id)
    }
  })
}

oos_binned_freq_GLM_AUS <- add_oos_predictions(data_AUS, binned_freq_AUS)
oos_binned_sev_GLM_AUS <- add_oos_predictions(data_AUS %>% filter(nclaims > 0) %>% filter(!is.na(average)), binned_sev_AUS)

oos_binned_freq_GLM_FR <- add_oos_predictions(data_FR, binned_freq_FR)
oos_binned_sev_GLM_FR <- add_oos_predictions(data_FR %>% filter(nclaims > 0) %>% filter(!is.na(average)), binned_sev_FR)

oos_binned_freq_GLM_NOR <- add_oos_predictions(data_NOR, binned_freq_NOR)
oos_binned_sev_GLM_NOR <- add_oos_predictions(data_NOR %>% filter(nclaims > 0) %>% filter(!is.na(average)), binned_sev_NOR)

save(oos_binned_freq_GLM_AUS, file = 'oos_binned_freq_GLM_AUS')
save(oos_binned_sev_GLM_AUS, file = 'oos_binned_sev_GLM_AUS')
save(oos_binned_freq_GLM_FR, file = 'oos_binned_freq_GLM_FR')
save(oos_binned_sev_GLM_FR, file = 'oos_binned_sev_GLM_FR')
save(oos_binned_freq_GLM_NOR, file = 'oos_binned_freq_GLM_NOR')
save(oos_binned_sev_GLM_NOR, file = 'oos_binned_sev_GLM_NOR')

## ----- Calculate out-of-sample deviances -----

oos_freq_GLM_AUS <- lapply(1:6, function(fold){ 
  dev_poiss(oos_binned_freq_GLM_AUS[[fold]] %>% filter(fold_nr == fold) %>% pull(nclaims), oos_binned_freq_GLM_AUS[[fold]] %>% filter(fold_nr == fold) %>% pull(prediction))
})
oos_sev_GLM_AUS <-lapply(1:6, function(fold){ 
  dev_gamma(oos_binned_sev_GLM_AUS[[fold]] %>% filter(fold_nr == fold) %>% pull(average), 
            oos_binned_sev_GLM_AUS[[fold]] %>% filter(fold_nr == fold) %>% pull(prediction),
            oos_binned_sev_GLM_AUS[[fold]] %>% filter(fold_nr == fold) %>% pull(nclaims))
})

oos_freq_GLM_FR <-lapply(1:6, function(fold){ 
  dev_poiss(oos_binned_freq_GLM_FR[[fold]] %>% filter(fold_nr == fold) %>% pull(nclaims), oos_binned_freq_GLM_FR[[fold]] %>% filter(fold_nr == fold) %>% pull(prediction))
})
oos_sev_GLM_FR <-lapply(1:6, function(fold){ 
  dev_gamma(oos_binned_sev_GLM_FR[[fold]] %>% filter(fold_nr == fold) %>% pull(average), 
            oos_binned_sev_GLM_FR[[fold]] %>% filter(fold_nr == fold) %>% pull(prediction),
            oos_binned_sev_GLM_FR[[fold]] %>% filter(fold_nr == fold) %>% pull(nclaims))
})

oos_freq_GLM_NOR <-lapply(1:6, function(fold){ 
  dev_poiss(oos_binned_freq_GLM_NOR[[fold]] %>% filter(fold_nr == fold) %>% pull(nclaims), oos_binned_freq_GLM_NOR[[fold]] %>% filter(fold_nr == fold) %>% pull(prediction))
})
oos_sev_GLM_NOR <-lapply(1:6, function(fold){ 
  dev_gamma(oos_binned_sev_GLM_NOR[[fold]] %>% filter(fold_nr == fold) %>% pull(average), 
            oos_binned_sev_GLM_NOR[[fold]] %>% filter(fold_nr == fold) %>% pull(prediction),
            oos_binned_sev_GLM_NOR[[fold]] %>% filter(fold_nr == fold) %>% pull(nclaims))
})

# Add the Belgian OOS from data sets Henckaerts
load("NClaims_datasets.RData")
load("ClaimAmount_datasets.RData")

oos_freq_GLM_BE <- sapply(1:6, function(x) dev_poiss(NC_all_testfolds_CANNglm[[x]]$testset$response %>% pull(nclaims), 
                                                     NC_all_testfolds_CANNglm[[x]]$testset$data %>% pull(prediction)) )
oos_sev_GLM_BE <- sapply(1:6, function(x) dev_gamma(CA_all_testfolds_CANNglm[[x]]$testset$response %>% pull(average), 
                                                     CA_all_testfolds_CANNglm[[x]]$testset$data %>% pull(prediction),
                                                     CA_all_testfolds_CANNglm[[x]]$testset$weights %>% pull(nclaims)) )

# Combine all out-of-sample performances into a table
oos_all_GLMs <- bind_rows(
  bind_cols(Fold = 1:6, Freq = oos_freq_GLM_BE, Sev = oos_sev_GLM_BE) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'BE'),
  bind_cols(Fold = 1:6, Freq = unlist(oos_freq_GLM_AUS), Sev = unlist(oos_sev_GLM_AUS)) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'AUS'),
  bind_cols(Fold = 1:6, Freq = unlist(oos_freq_GLM_FR), Sev = unlist(oos_sev_GLM_FR)) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'FR'),
  bind_cols(Fold = 1:6, Freq = unlist(oos_freq_GLM_NOR), Sev = unlist(oos_sev_GLM_NOR)) %>% 
    gather('Freq', 'Sev', key = 'Problem', value = 'OOS') %>% mutate(Data = 'NOR')
) %>% mutate(Model = 'Binned GLM') %>% select(Model,Data,Problem,Fold,OOS)
save(oos_all_GLMs, file = 'oos_all_GLMs')

# -----
# ----- THE END -----
# -----