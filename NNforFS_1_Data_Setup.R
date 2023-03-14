# -----
# ----- DATA READ IN AND SETUP -----

# Read in raw data from the Maidrr package and pre-process to correct format

# ----- Install packages needed -----

#library(reticulate)
#use_python("C:/Users/Frynn/.conda/envs/tf_noGpu/python")
#reticulate::use_condaenv("my_env")

used_packages <- c('tidyverse')
suppressMessages(packages <- lapply(used_packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x)
    library(x, character.only = TRUE)
  }
}))

# ----- Data reading and manupulation -----

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Location of the extra data files
location_datasets <- "/home/lynn/Dropbox/MTPL Data Sets"
#location_datasets <- "C:/Users/u0086713/Dropbox/MTPL Data Sets"

# Read in Functions File
source("Functions.R")

## ----- Australian -----

# Australian data set
data_AUS <- readRDS(paste0(location_datasets,"/ausprivauto/data.rds")) %>% 
  as_tibble %>% 
  mutate(fold_nr = ((row_number() - 1) %% 6) + 1)
cat_AUS <- c("VehAge", "VehBody", "Gender", "DrivAge")
feat_AUS <- c("VehValue","VehAge", "VehBody", "Gender", "DrivAge")

## ----- Belgian -----

# Belgian Data
data_readin<-readRDS("./Data/Data.rds") #Dataset
data_BE <- data_readin %>%
  as_tibble() %>%
  mutate(fold_nr = as.numeric(substring(fold,5))) %>%
  arrange(fold_nr, nclaims, average, expo) %>%
  select(!c("claim","postcode", "fold","amount"))
cat_BE <- c("coverage", "fuel", "sex", "use", "fleet")
feat_BE <- c("ageph","bm","agec","power","long","lat","coverage","fuel","sex","use","fleet")

## ----- French -----

# French data set
data_FR <- readRDS(paste0(location_datasets,"/fremtpl2/data.rds")) %>% 
  as_tibble %>% 
  mutate(fold_nr = ((row_number() - 1) %% 6) + 1) %>% 
  mutate(Density = log(Density)) %>% 
  mutate(DrivAge = case_when(
    DrivAge < 21 ~ 1,
    21 <= DrivAge & DrivAge < 26 ~ 2,
    26 <= DrivAge & DrivAge < 30 ~ 3,
    30 <= DrivAge & DrivAge < 40 ~ 4,
    40 <= DrivAge & DrivAge < 50 ~ 5,
    50 <= DrivAge & DrivAge < 70 ~ 6,
    70 <= DrivAge ~ 7,
    TRUE ~ 7
  )) %>% 
  mutate(VehAge = case_when(
    VehAge == 0 ~ 1,
    0 < VehAge & VehAge < 11 ~ 2,
    11 <= VehAge ~ 3,
    TRUE ~ 3
  )) %>% 
  mutate(VehAge = factor(VehAge),DrivAge=factor(DrivAge)) %>% 
  group_by(across(c(-average))) %>% # We need to group, as this data set contains one line per claim
  summarise(average = mean(average))  %>% 
  filter(row_number()==1) %>% # With filter on row_number we only keep track of one line per group, which is what we need
  ungroup
cat_FR <- c("Area","VehPower","VehBrand","Region", "VehGas", "VehAge", "DrivAge")
feat_FR <- c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")

## ----- Norwegian -----

# Norwegian data set
data_NOR <- readRDS(paste0(location_datasets,"/norauto/data.rds")) %>% 
  as_tibble %>% 
  mutate(fold_nr = ((row_number() - 1) %% 6) + 1)
cat_NOR <- c("Male","Young","DistLimit","GeoRegion")
feat_NOR <- c("Male", "Young", "DistLimit", "GeoRegion")

# ----- Save all data files ----

save(data_AUS, cat_AUS, feat_AUS, file = "data_AUS_prepared.RData")
save(data_BE, cat_BE, feat_BE, file = "data_BE_prepared.RData")
save(data_FR, cat_FR, feat_FR, file = "data_FR_prepared.RData")
save(data_NOR, cat_NOR, feat_NOR, file = "data_NOR_prepared.RData")

# -----
# ----- DATA PREP FOR NEURAL NETWORKS

# This step can only be done after fitting the Binned GLM and the GBM models

## ----- Read in additional data info -----

# Read in GLM data for AUS, FR and NOR
load('oos_binned_freq_GLM_AUS')
load('oos_binned_sev_GLM_AUS')
load('oos_binned_freq_GLM_FR')
load('oos_binned_sev_GLM_FR')
load('oos_binned_freq_GLM_NOR')
load('oos_binned_sev_GLM_NOR')

# Read in GBM data for AUS, FR and NOR
load('oos_freq_GBM_AUS')
load('oos_sev_GBM_AUS')
load('oos_freq_GBM_FR')
load('oos_sev_GBM_FR')
load('oos_freq_GBM_NOR')
load('oos_sev_GBM_NOR')

# Read in GLM and GBM data for BE
glm_data<-readRDS("./Data/data_glm.rds") #Data used in glm model
gbm_fits<-readRDS("./Data/mfits_gbm.rds") #Gbm's as constructed in Henckaerts et al. (2019)
glm_fits<-readRDS("./Data/mfits_glm.rds") #Glm's as constructed in Henckaerts et al. (2019)
gbm_fits <- gbm_fits[c(1:6,13:18)] # We do not use the log_normal GBM fits

# Define scale and output var
NC_output <- "nclaims"
CA_output <- "average"

# Define variables to be scaled
scale_AUS <- setdiff(names(data_AUS),c(cat_AUS,"expo","id","nclaims","average","fold_nr"))
scale_FR <- setdiff(names(data_FR),c(cat_FR,"expo","id","nclaims","average","fold_nr"))
scale_NOR <- setdiff(names(data_NOR),c(cat_NOR,"expo","id","nclaims","average","fold_nr"))
scale_BE <- c("ageph","bm","agec","power","long","lat")


## ----- Freq data for NN -----

NC_data_AUS <- lapply(1:6, function(fold){
  train_test(fold, scaling(fold, data_AUS, scale_AUS), input_vars = names(data_AUS)[names(data_AUS)!=CA_output], NC_output)
})

NC_data_BE <- lapply(1:6, function(fold){
  train_test(fold, scaling(fold, data_BE, scale_BE), names(data_BE)[names(data_BE)!=CA_output], NC_output)
})

NC_data_FR <- lapply(1:6, function(fold){
  train_test(fold, scaling(fold, data_FR, scale_FR), names(data_FR)[names(data_FR)!=CA_output], NC_output)
})

NC_data_NOR <- lapply(1:6, function(fold){
  train_test(fold, scaling(fold, data_NOR, scale_NOR), names(data_NOR)[names(data_NOR)!=CA_output], NC_output)
})

## ----- Freq data for CANN GLM -----

NC_data_AUS_GLM <- NC_data_AUS
for(x in 1:6){
  # Add predicitions to data
  NC_data_AUS_GLM[[x]]$testset$data <- left_join(NC_data_AUS_GLM[[x]]$testset$data, oos_binned_freq_GLM_AUS[[x]] %>% select(prediction,id), by=c("id")) %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  NC_data_AUS_GLM[[x]]$trainset$data <- left_join(NC_data_AUS_GLM[[x]]$trainset$data, oos_binned_freq_GLM_AUS[[x]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

NC_data_FR_GLM <- NC_data_FR
for(x in 1:6){
  # Add predicitions to data
  NC_data_FR_GLM[[x]]$testset$data <- left_join(NC_data_FR[[x]]$testset$data, oos_binned_freq_GLM_FR[[x]] %>% select(prediction,id), by=c("id")) %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  NC_data_FR_GLM[[x]]$trainset$data <- left_join(NC_data_FR[[x]]$trainset$data, oos_binned_freq_GLM_FR[[x]] %>% select(prediction,id), by=c("id")) %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

NC_data_NOR_GLM <- NC_data_NOR
for(x in 1:6){
  # Add predicitions to data
  NC_data_NOR_GLM[[x]]$testset$data <- left_join(NC_data_NOR_GLM[[x]]$testset$data, oos_binned_freq_GLM_NOR[[x]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  NC_data_NOR_GLM[[x]]$trainset$data <- left_join(NC_data_NOR_GLM[[x]]$trainset$data, oos_binned_freq_GLM_NOR[[x]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

NC_data_BE_GLM <- NC_all_testfolds_CANNglm

## ----- Freq data for CANN GBM -----

NC_data_AUS_GBM <- NC_data_AUS
for(x in 1:6){
  # Add predicitions to data
  NC_data_AUS_GBM[[x]]$testset$data <- left_join(NC_data_AUS_GBM[[x]]$testset$data, oos_freq_GBM_AUS[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  NC_data_AUS_GBM[[x]]$trainset$data <- left_join(NC_data_AUS_GBM[[x]]$trainset$data, oos_freq_GBM_AUS[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

NC_data_FR_GBM <- NC_data_FR
for(x in 1:6){
  # Add predicitions to data
  NC_data_FR_GBM[[x]]$testset$data <- left_join(NC_data_FR_GBM[[x]]$testset$data, oos_freq_GBM_FR[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  NC_data_FR_GBM[[x]]$trainset$data <- left_join(NC_data_FR_GBM[[x]]$trainset$data, oos_freq_GBM_FR[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

NC_data_NOR_GBM <- NC_data_NOR
for(x in 1:6){
  # Add predicitions to data
  NC_data_NOR_GBM[[x]]$testset$data <- left_join(NC_data_NOR_GBM[[x]]$testset$data, oos_freq_GBM_NOR[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  NC_data_NOR_GBM[[x]]$trainset$data <- left_join(NC_data_NOR_GBM[[x]]$trainset$data, oos_freq_GBM_NOR[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

NC_data_BE_GBM <- NC_all_testfolds_CANNgbm

save(NC_data_AUS, NC_data_BE, NC_data_FR, NC_data_NOR,
     NC_data_AUS_GLM, NC_data_BE_GLM, NC_data_FR_GLM, NC_data_NOR_GLM,
     NC_data_AUS_GBM, NC_data_BE_GBM, NC_data_FR_GBM, NC_data_NOR_GBM, file = "NClaims_all_data_sets.RData")
#load("NClaims_all_data_sets.RData")

## ----- Sev data for NN -----

CA_data_AUS <- lapply(1:6, function(fold){
  train_test(fold, scaling(fold,data_AUS %>% filter(nclaims > 0) %>% filter(!is.na(average)),scale_AUS), names(data_AUS)[names(data_AUS)!='expo'], CA_output, weight_var = 'nclaims')
})

CA_data_BE <- lapply(1:6, function(fold){
  train_test(fold, scaling(fold,data_BE %>% filter(nclaims > 0) %>% filter(!is.na(average)),scale_BE), names(data_BE)[names(data_BE)!='expo'], CA_output, weight_var = 'nclaims')
})

CA_data_FR <- lapply(1:6, function(fold){
  train_test(fold, scaling(fold,data_FR %>% filter(nclaims > 0) %>% filter(!is.na(average)),scale_FR), names(data_FR)[names(data_FR)!='expo'], CA_output, weight_var = 'nclaims')
})

CA_data_NOR <- lapply(1:6, function(fold){
  train_test(fold, scaling(fold,data_NOR %>% filter(nclaims > 0) %>% filter(!is.na(average)),scale_NOR), names(data_NOR)[names(data_NOR)!='expo'], CA_output, weight_var = 'nclaims')
})

## ----- Sev data for CANN GLM -----

CA_data_AUS_GLM <- CA_data_AUS
for(x in 1:6){
  # Add predicitions to data
  CA_data_AUS_GLM[[x]]$testset$data <- left_join(CA_data_AUS_GLM[[x]]$testset$data, oos_binned_sev_GLM_AUS[[x]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  CA_data_AUS_GLM[[x]]$trainset$data <- left_join(CA_data_AUS_GLM[[x]]$trainset$data, oos_binned_sev_GLM_AUS[[x]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

CA_data_FR_GLM <- CA_data_FR
for(x in 1:6){
  # Add predicitions to data
  CA_data_FR_GLM[[x]]$testset$data <- left_join(CA_data_FR_GLM[[x]]$testset$data, oos_binned_sev_GLM_FR[[x]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  CA_data_FR_GLM[[x]]$trainset$data <- left_join(CA_data_FR_GLM[[x]]$trainset$data, oos_binned_sev_GLM_FR[[x]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

CA_data_NOR_GLM <- CA_data_NOR
for(x in 1:6){
  # Add predicitions to data
  CA_data_NOR_GLM[[x]]$testset$data <- left_join(CA_data_NOR_GLM[[x]]$testset$data, oos_binned_sev_GLM_NOR[[x]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  CA_data_NOR_GLM[[x]]$trainset$data <- left_join(CA_data_NOR_GLM[[x]]$trainset$data, oos_binned_sev_GLM_NOR[[x]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

CA_data_BE_GLM <- CA_all_testfolds_CANNglm

## ----- Sev data for CANN GBM -----

CA_data_AUS_GBM <- CA_data_AUS
for(x in 1:6){
  # Add predicitions to data
  CA_data_AUS_GBM[[x]]$testset$data <- left_join(CA_data_AUS_GBM[[x]]$testset$data, oos_sev_GBM_AUS[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  CA_data_AUS_GBM[[x]]$trainset$data <- left_join(CA_data_AUS_GBM[[x]]$trainset$data, oos_sev_GBM_AUS[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

CA_data_FR_GBM <- CA_data_FR
for(x in 1:6){
  # Add predicitions to data
  CA_data_FR_GBM[[x]]$testset$data <- left_join(CA_data_FR_GBM[[x]]$testset$data, oos_sev_GBM_FR[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  CA_data_FR_GBM[[x]]$trainset$data <- left_join(CA_data_FR_GBM[[x]]$trainset$data, oos_sev_GBM_FR[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

CA_data_NOR_GBM <- CA_data_NOR
for(x in 1:6){
  # Add predicitions to data
  CA_data_NOR_GBM[[x]]$testset$data <- left_join(CA_data_NOR_GBM[[x]]$testset$data, oos_sev_GBM_NOR[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
  CA_data_NOR_GBM[[x]]$trainset$data <- left_join(CA_data_NOR_GBM[[x]]$trainset$data, oos_sev_GBM_NOR[[x]][[1]] %>% select(prediction,id), by="id") %>% 
    select(!c("id")) %>% 
    select(!any_of("expo"))
} 

CA_data_BE_GBM <- CA_all_testfolds_CANNgbm

save(CA_data_AUS, CA_data_BE, CA_data_FR, CA_data_NOR,
     CA_data_AUS_GLM, CA_data_BE_GLM, CA_data_FR_GLM, CA_data_NOR_GLM,
     CA_data_AUS_GBM, CA_data_BE_GBM, CA_data_FR_GBM, CA_data_NOR_GBM, file = "ClaimAmount_all_data_sets.RData")
#load("ClaimAmount_all_data_sets.RData")

# -----
# ----- THE END -----
# -----