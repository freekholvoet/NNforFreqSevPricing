# Scaling the train and test data, dependent on test_fold nr
scaling <- function(test_nr,dataset,scale_vars,fold_var="fold_nr", order_data = TRUE){
  
  # Filter out the test set, and apply scaling on the wanted variables
  trainset <- dataset %>% 
    filter_at(fold_var,any_vars(.!=test_nr)) %>%
    mutate_at(scale_vars,scale)
  
  # Extract the mean and st.dev used in scaling the variables
  train_colMean <- dataset %>% filter_at(fold_var,any_vars(.!=test_nr)) %>% summarise_at(scale_vars,mean)
  train_st.dev <- dataset %>% filter_at(fold_var,any_vars(.!=test_nr)) %>% summarise_at(scale_vars, sd)
  
  # Apply scaling on the test set with the mean and st.dev from the trainset
  testset <- dataset %>% 
    filter_at(fold_var,any_vars(.==test_nr)) %>%
    select(all_of(scale_vars)) %>% 
    as.matrix() %>%
    scale(.,center = train_colMean, scale = train_st.dev) %>% 
    as_tibble() %>%
    bind_cols(dataset %>% filter_at(fold_var,any_vars(.==test_nr)) %>% select(!all_of(scale_vars)))
  # Cleaner would be to use mutate_at, but the function scale would not accept vectors of parameters 
  # So a select is done, mutation applied, and rebound with the other columns
  
  #rebind the training and test set
  data_scaled <- bind_rows(trainset,testset) 
  if(order_data == TRUE){data_scaled %>% arrange(fold_nr, nclaims, average, expo)}
  
  # Return the scaled data set
  return(data_scaled)
}

# Split data into training and test fold, data and response, and applis one_hot_encoding if asked
train_test <- function(test_nr,dataset,input_vars,output_vars, one_hot_vars = list(), weight_var = "", fold_var="fold_nr"){
  # Function to split the data into training folds and test fold
  # Categorical vars are one_hot encoded (leave one_hot_vars empty to skip onehot enconding)
  # Returns list with train data and response and test data and response
  # Fold_nr is left in the training data for validation splitting
  
  data_oh <- dataset %>% 
    select(any_of(c(input_vars,output_vars,fold_var,weight_var))) %>%
    as.data.table() %>% 
    one_hot(cols=one_hot_vars) 
  
  trainset <- data_oh %>% 
    filter_at(fold_var,any_vars(.!=test_nr))
  testset <- data_oh %>% 
    filter_at(fold_var,any_vars(.==test_nr))
  
  train_data <- list(data = trainset %>% select(!any_of(c(output_vars,weight_var))) %>% as_tibble(),
                     response = trainset %>% select(output_vars) %>% as_tibble())
  test_data <- list(data = testset %>% select(!any_of(c(output_vars,weight_var))) %>% as_tibble(), 
                    response = testset %>% select(output_vars) %>% as_tibble())
  if(weight_var != ""){
    train_data <- c(train_data, list(weights = trainset %>% select(weight_var) %>% as_tibble()))
    test_data <- c(test_data, list(weights = testset %>% select(weight_var) %>% as_tibble()))
  }
  return(list(trainset = train_data,testset = test_data))
}

# Split training data into train and validation sets, apply after train_test function
train_val <- function(fold_data, val_nr, fold_var="fold_nr", id_var = "id"){
  
  if(id_var %in% colnames(fold_data$trainset$data)){
    fold_data$trainset$data <- fold_data$trainset$data %>% select(!c(id_var))
  }
  
  # check the output variable used
  output_vars <- colnames(fold_data$trainset$response)
  if(exists("weights",fold_data$trainset)){weight_var = names(fold_data$trainset$weights)}
  
  # Remake the training data and response in a tibble
  tibble_data <- bind_cols(fold_data$trainset$data,fold_data$trainset$response)
  if(exists("weights",fold_data$trainset)){tibble_data <- bind_cols(tibble_data,fold_data$trainset$weights)}
  
  # Split up training and validation data
  train_data <- list(data = tibble_data %>% filter_at(fold_var,any_vars(.!=val_nr)) %>% 
                       select(!any_of(c(output_vars,fold_var))),
                     response = tibble_data %>% filter_at(fold_var,any_vars(.!=val_nr)) %>% 
                       select(output_vars))
  test_data <- list(data = tibble_data %>% filter_at(fold_var,any_vars(.==val_nr)) %>% 
                      select(!any_of(c(output_vars,fold_var))),
                    response = tibble_data %>% filter_at(fold_var,any_vars(.==val_nr)) %>% select(output_vars))
  
  if(exists("weights",fold_data$trainset)){
    train_data <- c(train_data, list(weights = train_data$data %>% select(weight_var) %>% as_tibble()))
    test_data <- c(test_data, list(weights = test_data$data %>% select(weight_var) %>% as_tibble()))
    train_data$data <- train_data$data %>% select(!weight_var)
    test_data$data <- test_data$data %>% select(!weight_var)
  }
  return(list(trainset = train_data, testset = test_data))
}

# AutoEncoder training with random validation set for early stopping
autoencoder_train <- function(train_data_list, encode_dimension, random_val_split = 0.2,
                              activation, optimizer, lossfunction, epochs, batch, 
                              verbose=FALSE){
  
  # Train data concatenated as input
  train_concatenate <- do.call(cbind,train_data_list)
  
  ## Input size
  input_size = dim(train_concatenate)[2]
  latent_size = encode_dimension
  
  ## Encoder
  enc_input = layer_input(shape = input_size)
  enc_output = enc_input %>% 
    layer_dense(units=latent_size)
  encoder = keras_model(enc_input, enc_output)
  
  ## Decoder
  dec_input = layer_input(shape = latent_size)
  decoder_list <- lapply(train_data_list, function(var){
    dec_input %>% layer_dense(units = dim(var)[2], activation = activation)
  })
  decoder = keras_model(dec_input, decoder_list)
  
  # Autoencoder
  aen_input = layer_input(shape = input_size)
  aen_output = aen_input %>% 
    encoder() %>% 
    decoder()
  aen = keras_model(aen_input, aen_output)
  
  # Compile and fit the autoencoder
  aen %>% compile(optimizer = optimizer,
                  loss = lossfunction,
                  loss_weights = rep(1,length(train_data_list)))
  
  history <- aen %>% 
    fit(train_concatenate,
        train_data_list, 
        epochs = epochs, batch = batch, verbose=verbose,
        #callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 0.0001 ,patience = 5)),
        #validation_split = random_val_split
    ) 
  
  return(list(Encoder = encoder,Decoder = decoder, AutoEncoder = aen, AE_results = history))
  
}

# Perform a single run, provide data from train_test or train_val function 
single_run <- function(fold_data, flags_list, random_val_split = 0, 
                       fold_var = "fold_nr", id_var = "id", early_stopping = TRUE, 
                       GLM_Bias_Regularization = FALSE, output_modelinfo = FALSE){
  
  # For time elapsed tracking
  time <- Sys.time()
  
  dropout <- flags_list[["dropout"]] 
  batch <- flags_list[["batch"]]
  hiddennodes <- flags_list[["hiddennodes"]]
  activation_h <- flags_list[["activation_h"]]
  activation_out <- flags_list[["activation_out"]]
  optimizer <- flags_list[["optimizer"]]
  epochs <- flags_list[["epochs"]]
  loss <- flags_list[["loss"]]
  
  # If class hiddennodes is list, unlist (used for tuning dropout rates)
  if(class(hiddennodes)=="list"){hiddennodes <- unlist(hiddennodes)}
  # Error message if a different number of dropouts and layers are provided
  if((length(hiddennodes)!=length(dropout)) & length(dropout) != 1){
    stop(paste(" The number of dropouts provided (",
               paste(dropout, collapse=", "),
               ") is not equal to the hidden nodes provided (",
               paste(hiddennodes, collapse=", "),")"))}
  # If one dropoutrate is provided, apply it for all hidden layers
  if(length(dropout) == 1){dropout <- rep(dropout,length(hiddennodes))}
  
  # Remove the fold_nr and id variables if in the data
  fold_data$trainset$data <- fold_data$trainset$data %>% select(!any_of(c('id','fold_nr')))
  fold_data$testset$data <- fold_data$testset$data %>% select(!any_of(c('id','fold_nr')))
  
  # Model setup in Keras
  model <- keras_model_sequential()
  # Add all layers with applicable nodes and dropout
  for(i in 1:length(hiddennodes)){
    model %>% 
      layer_dense(units = hiddennodes[i], input_shape = ncol(fold_data$trainset$data), 
                  name = paste0("layer_",i)) %>%
      layer_activation(activation = activation_h)
    if(dropout[i]!=0){model %>% layer_dropout(rate=dropout[i])}
  }
  # Output activation
  model %>% layer_dense(units = 1, activation = activation_out, name = "output_node") 
  
  if(loss == "poisson"){
    model %>% compile(
      optimizer = optimizer,
      loss = "poisson",
      metrics = metric_poisson_metric
    )
    # Weights for the poisson do not matter, so we take 1 for everything. Does not influence the outcome
    train_weights <- rep(1,nrow(fold_data$trainset$data))
    val_weights <- rep(1,nrow(fold_data$testset$data))
  } else if(loss == "gamma"){
    model %>% compile(
      optimizer = optimizer,
      loss = custom_metric('metric',gamma_metric()),
      metrics = custom_metric('metric',gamma_metric()),
      weighted_metrics = custom_metric('weighted_metric', gamma_metric())
    )
    # Setting the weights for the Gamma to have weighted loss
    train_weights <- fold_data$trainset$weights %>% data.matrix()
    val_weights <- fold_data$testset$weights %>% data.matrix()
  }
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20) #The val_loss is weighted if weights are avaliable
  
  # Fit the model on the train data with the test/val data for early stopping 
  history <- model %>%
    fit(x = fold_data$trainset$data %>% data.matrix(), 
        y = fold_data$trainset$response %>% data.matrix(),
        sample_weight =  train_weights,
        epochs = epochs,
        batch_size = batch,
        callbacks = if(early_stopping==TRUE){list(early_stop)},
        validation_split = random_val_split,
        validation_data = if(random_val_split == 0){list(x_val = fold_data$testset$data %>% data.matrix(), 
                                                         y_val = fold_data$testset$response %>% data.matrix(),
                                                         val_sample_weight = val_weights)},
        verbose=0)
  
  if(GLM_Bias_Regularization == TRUE){
    # Use the weights of the last hidden layer as inputs for a glm
    intermediate_model <- keras_model(inputs = model$input,
                                      outputs = get_layer(model,paste0("layer_",length(hiddennodes)))$output)
    intermediate_output <- predict(intermediate_model, fold_data$trainset$data %>% data.matrix())
    
    if(loss=="gamma"){
      model_bias_regulated <- glm(fold_data$trainset$response %>% data.matrix() ~ intermediate_output,  family= Gamma("log"))
    } else {
      model_bias_regulated <- glm(fold_data$trainset$response %>% data.matrix() ~ intermediate_output,  family= poisson("log"))
    }
    
    # Calculate the loss on the validation set
    intermediate_test_output <- predict(intermediate_model, fold_data$testset$data %>% data.matrix())
    predict_val <- predict(object=model_bias_regulated,newdata=list("intermediate_output"=intermediate_test_output),type="response")
  } else {
    # Calculate the loss on the validation set
    predict_val <- model %>% predict(fold_data$testset$data %>% data.matrix())
  }
  
  loss_val <- ifelse(loss=="gamma", 
                     dev_gamma(fold_data$testset$response %>% data.matrix(), predict_val, fold_data$testset$weights %>% data.matrix()), 
                     dev_poiss_2(fold_data$testset$response %>% data.matrix(), predict_val))
  
  # Return the parameters used, the validation error and the time elapsed
  results <- tibble(val_loss = loss_val, 
                    epochs_used = length(history$metrics$loss), 
                    time_elapsed = difftime(time1 = Sys.time(), time2 = time, units = "min"),
                    Portfolio_total_prediction = sum(predict_val),
                    Portfolio_total = sum(fold_data$testset$response),
                    Balance_ratio = Portfolio_total_prediction/Portfolio_total)
  # Also output model and variables used if asked (used for PDP)
  if(GLM_Bias_Regularization == TRUE){model_info <- list(intermediate_model, model_bias_regulated)} else {model_info <- model}
  output <- if(output_modelinfo == FALSE){results} else {
    list(results = results, model = model_info, 
         other_vars = colnames(fold_data$trainset$data), cat_vars = NULL, cann_variable = NULL)}
  return(output)
}

# Perform a cross validation run over all folds in the training data provided, returns the average validation error
crossvalidation_run <- function(fold_data, flags_list, fold_var="fold_nr"){
  
  # For time elapsed tracking
  time <- Sys.time()
  
  dropout <- flags_list[["dropout"]] 
  batch <- flags_list[["batch"]]
  hiddennodes <- flags_list[["hiddennodes"]]
  activation_h <- flags_list[["activation_h"]]
  activation_out <- flags_list[["activation_out"]]
  optimizer <- flags_list[["optimizer"]]
  epochs <- flags_list[["epochs"]]
  loss <- flags_list[["loss"]]
  
  # If class hiddennodes is list, unlist (used for tuning dropout rates)
  if(class(hiddennodes)=="list"){hiddennodes <- unlist(hiddennodes)}
  # Error message if a different number of dropouts and layers are provided
  if((length(hiddennodes)!=length(dropout)) & length(dropout) != 1){
    stop(paste(" The number of dropouts provided (",
               paste(dropout, collapse=", "),
               ") is not equal to the hidden nodes provided (",
               paste(hiddennodes, collapse=", "),")"))}
  # If one dropoutrate is provided, apply it for all hidden layers
  if(length(dropout) == 1){dropout <- rep(dropout,length(hiddennodes))}
  
  # Check which folds are in the training data
  folds <- unlist(unique(fold_data$trainset$data[,fold_var]))
  # Vector for validation set losses
  val_losses <- matrix(0, nrow = length(folds), ncol = 2)
  
  # Loop over all validation folds in the data, this can be replaced with a apply commando instead, but it seems to run very efficient as is
  for(f in folds){
    # Split the data into train and validation set
    val_split <- train_val(fold_data, f, fold_var)
    
    # Model setup in Keras
    model <- keras_model_sequential()
    # Add all layers with applicable nodes and dropout
    for(i in 1:length(hiddennodes)){
      model %>% 
        layer_dense(units = hiddennodes[i], input_shape = ncol(val_split$trainset$data), 
                    kernel_initializer=initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104)) %>%
        layer_activation(activation = activation_h)
      if(dropout[i]!=0){model %>% layer_dropout(rate=dropout[i])}
    }
    # Output activation
    model %>% layer_dense(units = 1, activation = activation_out) 
    
    if(loss == "poisson"){
      model %>% compile(
        loss = "poisson",
        optimizer = optimizer,
        metrics = metric_poisson_metric
      )
      # Weights for the poisson do not matter, so we take 1 for everything. Does not influence the outcome
      train_weights <- rep(1,nrow(val_split$trainset$data))
      val_weights <- rep(1,nrow(val_split$testset$data))
    } else if(loss == "gamma"){
      model %>% compile(
        loss = custom_metric('metric',gamma_metric()),
        optimizer = optimizer,
        metrics = custom_metric('metric',gamma_metric()),
        weighted_metrics = custom_metric('weighted_metric', gamma_metric())
      )
      # Setting the weights for the Gamma to have weighted loss
      train_weights <- val_split$trainset$weights %>% data.matrix()
      val_weights <- val_split$testset$weights %>% data.matrix()
    }
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)
    
    # Fit the model on the train data with the test/val data for early stopping 
    history <- model %>%
      fit(x = val_split$trainset$data %>% data.matrix(), 
          y = val_split$trainset$response %>% data.matrix(),
          sample_weight =  train_weights,
          epochs = epochs,
          batch_size = batch,
          callbacks = list(early_stop),
          validation_data = list(val_split$testset$data %>% data.matrix(), 
                                 val_split$testset$response %>% data.matrix(),
                                 val_sample_weight = val_weights),
          verbose=0)
    
    # Calculate the loss on the validation set
    predict_val <- model %>% predict(val_split$testset$data %>% data.matrix())
    val_losses[match(f,folds),1] <- ifelse(loss=="gamma", 
                                           dev_gamma(val_split$testset$response %>% data.matrix(), predict_val, val_split$testset$weights %>% data.matrix()), 
                                           dev_poiss_2(val_split$testset$response %>% data.matrix(), predict_val))
    val_losses[match(f,folds),2] <- length(history$metrics$loss)
  }
  
  # Return the average validation loss, the run time for info, and the used parameters
  return(tibble("validation_error" = mean(val_losses[,1]),
                "cross_run_time" = difftime(time1 = Sys.time(), time2 = time, units = "min"),
                "epochs_used" = list(val_losses[,2]), 
                "dropout" = list(dropout), 
                "batch" = batch, 
                "hiddennodes" = list(hiddennodes), 
                "activation_h" = activation_h, 
                "activation_out" = activation_out, 
                "optimizer" = optimizer, 
                "epochs" = epochs, 
                "loss" = loss))
}

# Single NN run with embedding layers for the cat_vars provided
single_run_Embedding <- function(fold_data, flags_list, random_val_split = 0, 
                                 embedding_output_dim = 1,  cat_vars = list(), fold_var = "fold_nr", 
                                 id_var = "id", early_stopping = TRUE, GLM_Bias_Regularization = FALSE,
                                 output_modelinfo = FALSE){
  
  # For time elapsed tracking
  time <- Sys.time()
  
  dropout <- flags_list[["dropout"]] 
  batch <- flags_list[["batch"]]
  hiddennodes <- flags_list[["hiddennodes"]]
  activation_h <- flags_list[["activation_h"]]
  activation_out <- flags_list[["activation_out"]]
  optimizer <- flags_list[["optimizer"]]
  epochs <- flags_list[["epochs"]]
  loss <- flags_list[["loss"]]
  
  # If class hiddennodes is list, unlist (used for tuning dropout rates)
  if(class(hiddennodes)=="list"){hiddennodes <- unlist(hiddennodes)}
  # Error message if a different number of dropouts and layers are provided
  if((length(hiddennodes)!=length(dropout)) & length(dropout) != 1){
    stop(paste(" The number of dropouts provided (",
               paste(dropout, collapse=", "),
               ") is not equal to the hidden nodes provided (",
               paste(hiddennodes, collapse=", "),")"))}
  # If one dropoutrate is provided, apply it for all hidden layers
  if(length(dropout) == 1){dropout <- rep(dropout,length(hiddennodes))}
  
  # Remove the fold_nr and id variables if in the data
  fold_data$trainset$data <- fold_data$trainset$data %>% select(!any_of(c('id','fold_nr')))
  fold_data$testset$data <- fold_data$testset$data %>% select(!any_of(c('id','fold_nr')))
  
  # If embedding output size is length 1, make vector
  if(length(embedding_output_dim)==1){
    embedding_output_dim <- rep(1,length(cat_vars))
  } else if (length(embedding_output_dim) != length(cat_vars)){
    stop(paste("Embedding output dimension vector is not the same as the number of variables to embed (dimensions =",
               length(embedding_output_dim), "but", length(cat_vars), "variables to embed provided"))}
  
  # Model setup in Keras
  model <- keras_model_sequential()
  
  # Determine non-embedding variables
  other_vars <- setdiff(colnames(fold_data$trainset$data),cat_vars)
  # Make input layer for the non-embedding variables
  other_layer <- layer_input(shape=length(other_vars), dtype='float32',name="Design")
  
  # Make the embedding layers for each embedding variable
  cat_input_layers <- lapply(cat_vars, function(x){
    layer_input(
      shape=c(1),
      dtype='float32',
      name = paste(x,"_input",sep = ""))})
  emb_layers <- lapply(cat_input_layers, function(x){
    x %>% layer_embedding(
      input_dim = nrow(unique(fold_data$trainset$data[,sub("\\_.*", "", x$name)]))+1, #WHY??? 
      output_dim = embedding_output_dim[match(sub("\\_.*", "", x$name), cat_vars)],
      input_length = 1 ) %>% 
      layer_flatten()})
  all_input_layers <- c(other_layer,unlist(emb_layers))
  
  # Concatenate the embedded layers with the regular input layers, and add the hidden layers and output layer
  net = all_input_layers %>% layer_concatenate()
  
  for(i in 1:length(hiddennodes)){
    net = net %>%
      layer_dense(units = hiddennodes[i], 
                  input_shape = ncol(fold_data$trainset$data), 
                  name = paste0("layer_",i), 
                  activation = activation_h)
    if(dropout[i]!=0){net = net %>% layer_dropout(rate=dropout[i])}
  }
  
  net = net %>% layer_dense(units = 1, activation = activation_out, name = "output_node") 
  
  # Make the model
  model <- keras_model(inputs = c(other_layer,unlist(cat_input_layers)), outputs = net)
  
  if(loss == "poisson"){
    model %>% compile(
      loss = "poisson",
      optimizer = optimizer,
      metrics = metric_poisson_metric
    )
    # Weights for the poisson do not matter, so we take 1 for everything. Does not influence the outcome
    train_weights <- rep(1,nrow(fold_data$trainset$data))
    val_weights <- rep(1,nrow(fold_data$testset$data))
  } else if(loss == "gamma"){
    model %>% compile(
      loss = custom_metric('metric',gamma_metric()),
      optimizer = optimizer,
      metrics = custom_metric('metric',gamma_metric()),
      weighted_metrics = custom_metric('weighted_metric', gamma_metric())
    )
    # Setting the weights for the Gamma to have weighted loss
    train_weights <- fold_data$trainset$weights %>% data.matrix()
    val_weights <- fold_data$testset$weights %>% data.matrix()
  }
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20) #The val_loss is weighted if weights are available
  
  train_mat <- lapply(c(list(other_vars),unlist(cat_vars)),  
                      function(data, x){ 
                        data[,x]
                      }, data = fold_data$trainset$data %>% data.matrix())
  test_mat <- lapply(c(list(other_vars),unlist(cat_vars)),  
                     function(data, x){ 
                       data[,x]
                     }, data = fold_data$testset$data %>% data.matrix())
  
  history <- model %>%
    fit(x = train_mat, 
        y = fold_data$trainset$response %>% data.matrix(),
        sample_weight =  train_weights,
        epochs = epochs,
        batch_size = batch,
        callbacks = if(early_stopping==TRUE){list(early_stop)},
        validation_split = random_val_split,
        validation_data = if(random_val_split == 0){list(x_val = test_mat, 
                                                         y_val = fold_data$testset$response %>% data.matrix(),
                                                         val_sample_weight = val_weights)},
        verbose=0)
  
  if(GLM_Bias_Regularization == TRUE){
    # Use the weights of the last hidden layer as inputs for a glm
    intermediate_model <- keras_model(inputs = model$input,
                                      outputs = get_layer(model,paste0("layer_",length(hiddennodes)))$output)
    intermediate_output <- predict(intermediate_model, train_mat)
    
    if(loss=="gamma"){
      model_bias_regulated <- glm(fold_data$trainset$response %>% data.matrix() ~ intermediate_output,  family= Gamma("log"))
    } else {
      model_bias_regulated <- glm(fold_data$trainset$response %>% data.matrix() ~ intermediate_output,  family= poisson("log"))
    } 
    
    # Calculate the loss on the validation set
    #intermediate_test_output <- predict(intermediate_model, test_mat)
    #predict_val <- predict(object=model_bias_regulated,newdata=list(intermediate_output=intermediate_test_output),type="response")
    
    predict_val <- predict_BRoption(list(intermediate_model,model_bias_regulated), test_mat, BR = TRUE, cann = FALSE)
    
  } else {
    # Calculate the loss on the validation set
    predict_val <- model %>% predict(test_mat)
  }
  
  if(loss=="gamma"){
    loss_val <-  dev_gamma(fold_data$testset$response %>% data.matrix(), predict_val, fold_data$testset$weights %>% data.matrix())
  } else {
    loss_val <-  dev_poiss_2(fold_data$testset$response %>% data.matrix(), predict_val)
  }
  
  # Return the parameters used, the validation error and the time elapsed
  results <- tibble(val_loss = loss_val, 
                    epochs_used = length(history$metrics$loss), 
                    time_elapsed = difftime(time1 = Sys.time(), time2 = time, units = "min"),
                    Portfolio_total_prediction = sum(predict_val),
                    Portfolio_total = sum(fold_data$testset$response),
                    Balance_ratio = Portfolio_total_prediction/Portfolio_total)
  # Also output model and variables used if asked (used for PDP)
  
  if(GLM_Bias_Regularization == TRUE){model_info <- list(intermediate_model, model_bias_regulated)} else {model_info <- model}
  output <- if(output_modelinfo == FALSE){results} else {
    list(results = results, model = model_info, 
         other_vars = other_vars, cat_vars = cat_vars, cann_variable = NULL)}
  return(output)
}

# Cross-validation run over all folds in data provided, with Embedding layers for the cat_vars provided
crossvalidation_run_Embedding <- function(fold_data, flags_list, embedding_output_dim = 1,  
                                          cat_vars = list(), fold_var="fold_nr", id_var = "id"){
  
  # For time elapsed tracking
  time <- Sys.time()
  
  dropout <- flags_list[["dropout"]] 
  batch <- flags_list[["batch"]]
  hiddennodes <- flags_list[["hiddennodes"]]
  activation_h <- flags_list[["activation_h"]]
  activation_out <- flags_list[["activation_out"]]
  optimizer <- flags_list[["optimizer"]]
  epochs <- flags_list[["epochs"]]
  loss <- flags_list[["loss"]]
  
  # If class hiddennodes is list, unlist (used for tuning dropout rates)
  if(class(hiddennodes)=="list"){hiddennodes <- unlist(hiddennodes)}
  # Error message if a different number of dropouts and layers are provided
  if((length(hiddennodes)!=length(dropout)) & length(dropout) != 1){
    stop(paste(" The number of dropouts provided (",
               paste(dropout, collapse=", "),
               ") is not equal to the hidden nodes provided (",
               paste(hiddennodes, collapse=", "),")"))}
  # If one dropoutrate is provided, apply it for all hidden layers
  if(length(dropout) == 1){dropout <- rep(dropout,length(hiddennodes))}
  
  # If embedding output size is length 1, make vector
  if(length(embedding_output_dim)==1){
    embedding_output_dim <- rep(1,length(cat_vars))
  } else if (length(embedding_output_dim) != length(cat_vars)){
    stop(paste("Embedding output dimension vector is not the same as the number of variables to embed (dimensions =",
               length(embedding_output_dim), "but", length(cat_vars), "variables to embed provided"))}
  
  # Check which folds are in the training data
  folds <- unlist(unique(fold_data$trainset$data[,fold_var]))
  # Vector for validation set losses
  val_losses <- matrix(0, nrow = length(folds), ncol = 2)
  
  # Determine non-embedding variables
  other_vars <- setdiff(colnames(fold_data$trainset$data),c(cat_vars,id_var))
  other_vars <- other_vars[other_vars!= fold_var]
  
  # Make input layer for the non-embedding variables
  other_layer <- layer_input(shape=length(other_vars), dtype='float32',name="Design")
  
  # Make the embedding layers for each embedding variable
  cat_input_layers <- lapply(cat_vars, function(x){
    layer_input(
      shape=c(1),
      dtype='float32',
      name = paste(x,"_input",sep = ""))})
  emb_layers <- lapply(cat_input_layers, function(x){
    x %>% layer_embedding(
      input_dim = nrow(unique(fold_data$trainset$data[,sub("\\_.*", "", x$name)]))+1, #WHY??? 
      output_dim = embedding_output_dim[match(sub("\\_.*", "", x$name), cat_vars)],
      input_length = 1 ) %>% 
      layer_flatten()})
  all_input_layers <- c(other_layer,unlist(emb_layers))
  
  # Loop over all validation folds in the data, this can be replaced with a apply commando instead, but it seems to run very efficient as is
  for(f in folds){
    
    # Split the data into train and validation set
    val_split <- train_val(fold_data, f, fold_var, id_var)
    
    # Concatenate the embedded layers with the regular input layers, and add the hidden layers and output layer
    net = all_input_layers %>% layer_concatenate()
    for(i in 1:length(hiddennodes)){
      net = net %>%
        layer_dense(units = hiddennodes[i], 
                    input_shape = ncol(val_split$trainset$data), 
                    kernel_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104), 
                    activation = activation_h)
      if(dropout[i]!=0){net = net %>% layer_dropout(rate=dropout[i])}
    }
    net = net %>% layer_dense(units = 1, activation = activation_out) 
    
    # Make the model
    model <- keras_model(inputs = c(other_layer,unlist(cat_input_layers)), outputs = net)
    
    if(loss == "poisson"){
      model %>% compile(
        loss = "poisson",
        optimizer = optimizer,
        metrics = metric_poisson_metric
      )
      # Weights for the poisson do not matter, so we take 1 for everything. Does not influence the outcome
      train_weights <- rep(1,nrow(val_split$trainset$data))
      val_weights <- rep(1,nrow(val_split$testset$data))
    } else if(loss == "gamma"){
      model %>% compile(
        loss = custom_metric('metric',gamma_metric()),
        optimizer = optimizer,
        metrics = custom_metric('metric',gamma_metric()),
        weighted_metrics = custom_metric('weighted_metric', gamma_metric())
      )
      # Setting the weights for the Gamma to have weighted loss
      train_weights <- val_split$trainset$weights %>% data.matrix()
      val_weights <- val_split$testset$weights %>% data.matrix()
    }
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20) #The val_loss is weighted if weights are avaliable
    
    # Make all input matrices. Each embedding variable gets a separate matrix, and one for the other inputs
    train_mat <- lapply(c(list(other_vars),unlist(cat_vars)),  
                        function(data, x){ 
                          data[,x]
                        }, data = val_split$trainset$data %>% data.matrix())
    test_mat <- lapply(c(list(other_vars),unlist(cat_vars)),  
                       function(data, x){ 
                         data[,x]
                       }, data = val_split$testset$data %>% data.matrix())
    
    # Fit the model on the train data with the test/val data for early stopping 
    history <- model %>%
      fit(x = train_mat, 
          y = val_split$trainset$response %>% data.matrix(),
          sample_weight =  train_weights,
          epochs = epochs,
          batch_size = batch,
          callbacks = list(early_stop),
          validation_data = list(x_val = test_mat, 
                                 y_val = val_split$testset$response %>% data.matrix(),
                                 val_sample_weight = val_weights),
          verbose=0
      )
    
    # Calculate the loss on the validation set
    predict_val <- model %>% predict(test_mat)
    val_losses[match(f,folds),1] <- ifelse(loss=="gamma", 
                                           dev_gamma(val_split$testset$response %>% data.matrix(), predict_val, val_split$testset$weights %>% data.matrix()), 
                                           dev_poiss_2(val_split$testset$response %>% data.matrix(), predict_val))
    val_losses[match(f,folds),2] <- length(history$metrics$loss)
  }
  
  # Return the average validation loss, the run time for info, and the used parameters
  return(tibble("validation_error" = mean(val_losses[,1]),
                "cross_run_time" = difftime(time1 = Sys.time(), time2 = time, units = "min"),
                "epochs_used" = list(val_losses[,2]), 
                "dropout" = list(dropout), 
                "batch" = batch, 
                "hiddennodes" = list(hiddennodes), 
                "activation_h" = activation_h, 
                "activation_out" = activation_out, 
                "optimizer" = optimizer, 
                "epochs" = epochs, 
                "loss" = loss))
}

# Single NN run with AutoEncoder for the cat_vars provided
single_run_AE <- function(fold_data, flags_list, random_val_split = 0, 
                          autoencoder_trained,
                          cat_vars = list(), fold_var = "fold_nr", 
                          id_var = "id", early_stopping = TRUE, GLM_Bias_Regularization = FALSE,
                          output_modelinfo = FALSE){
  
  # For time elapsed tracking
  time <- Sys.time()
  
  dropout <- flags_list[["dropout"]] 
  batch <- flags_list[["batch"]]
  hiddennodes <- flags_list[["hiddennodes"]]
  activation_h <- flags_list[["activation_h"]]
  activation_out <- flags_list[["activation_out"]]
  optimizer <- flags_list[["optimizer"]]
  epochs <- flags_list[["epochs"]]
  loss <- flags_list[["loss"]]
  
  # If class hiddennodes is list, unlist (used for tuning dropout rates)
  if(class(hiddennodes)=="list"){hiddennodes <- unlist(hiddennodes)}
  # Error message if a different number of dropouts and layers are provided
  if((length(hiddennodes)!=length(dropout)) & length(dropout) != 1){
    stop(paste(" The number of dropouts provided (",
               paste(dropout, collapse=", "),
               ") is not equal to the hidden nodes provided (",
               paste(hiddennodes, collapse=", "),")"))}
  # If one dropoutrate is provided, apply it for all hidden layers
  if(length(dropout) == 1){dropout <- rep(dropout,length(hiddennodes))}
  
  # Remove the fold_nr and id variables if in the data
  fold_data$trainset$data <- fold_data$trainset$data %>% select(!any_of(c('id','fold_nr')))
  fold_data$testset$data <- fold_data$testset$data %>% select(!any_of(c('id','fold_nr')))
  
  # Convert the categorical variables to a one=hot matrices
  train_cat_data <- lapply(cat_vars,function(var){
    fold_data$trainset$data %>% 
      pull(var) %>% 
      as.data.table() %>% 
      one_hot(cols=".") %>% 
      data.matrix()
  })
  test_cat_data <- lapply(cat_vars,function(var){
    fold_data$testset$data %>% 
      pull(var) %>% 
      as.data.table() %>% 
      one_hot(cols=".") %>% 
      data.matrix()
  })
  
  # Concatenate the categorical matrices for model input
  train_cat_data_concat <- do.call(cbind,train_cat_data)
  test_cat_data_concat <- do.call(cbind,test_cat_data)
  
  # Model setup in Keras
  model <- keras_model_sequential()
  
  # Determine non-embedding variables
  other_vars <- setdiff(colnames(fold_data$trainset$data),cat_vars)
  # Make input layer for the non-embedding variables
  other_layer <- layer_input(shape=length(other_vars), dtype='float32',name="Design")
  
  # Autoencoder layer for the categorical variables
  # Initialize the weights from the trained autoencoder
  AE_input <- layer_input(shape = dim(train_cat_data_concat)[2])
  AE_layer <- AE_input %>% 
    layer_dense(units = ncol(autoencoder_trained[[1]]), weights = autoencoder_trained)
  
  # Concatenate the categorical autoencoded layer with the regular input layer, and add the hidden layers and output layer
  net = c(other_layer,AE_layer) %>% layer_concatenate()
  
  for(i in 1:length(hiddennodes)){
    net = net %>%
      layer_dense(units = hiddennodes[i], 
                  input_shape = ncol(fold_data$trainset$data), 
                  name = paste0("layer_",i), 
                  activation = activation_h)
    if(dropout[i]!=0){net = net %>% layer_dropout(rate=dropout[i])}
  }
  
  net = net %>% layer_dense(units = 1, activation = activation_out, name = "output_node") 
  
  # Make the model
  model <- keras_model(inputs = c(other_layer,AE_input), outputs = net)
  
  if(loss == "poisson"){
    model %>% compile(
      loss = "poisson",
      optimizer = optimizer,
      metrics = metric_poisson_metric
    )
    # Weights for the poisson do not matter, so we take 1 for everything. Does not influence the outcome
    train_weights <- rep(1,nrow(fold_data$trainset$data))
    val_weights <- rep(1,nrow(fold_data$testset$data))
  } else if(loss == "gamma"){
    model %>% compile(
      loss = custom_metric('metric',gamma_metric()),
      optimizer = optimizer,
      metrics = custom_metric('metric',gamma_metric()),
      weighted_metrics = custom_metric('weighted_metric', gamma_metric())
    )
    # Setting the weights for the Gamma to have weighted loss
    train_weights <- fold_data$trainset$weights %>% data.matrix()
    val_weights <- fold_data$testset$weights %>% data.matrix()
  }
  early_stop <- callback_early_stopping(monitor = "val_loss", min_delta = 0.0001, patience = 20) #The val_loss is weighted if weights are available
  
  train_mat <- list(fold_data$trainset$data %>% select(other_vars) %>% data.matrix(),
                    train_cat_data_concat)
  test_mat <- list(fold_data$testset$data %>% select(other_vars) %>% data.matrix(),
                   test_cat_data_concat)
  
  history <- model %>%
    fit(x = train_mat, 
        y = fold_data$trainset$response %>% data.matrix(),
        sample_weight =  train_weights,
        epochs = epochs,
        batch_size = batch,
        callbacks = if(early_stopping==TRUE){list(early_stop)},
        validation_split = random_val_split,
        validation_data = if(random_val_split == 0){list(x_val = test_mat, 
                                                         y_val = fold_data$testset$response %>% data.matrix(),
                                                         val_sample_weight = val_weights)},
        verbose=0)
  
  if(GLM_Bias_Regularization == TRUE){
    # Use the weights of the last hidden layer as inputs for a glm
    intermediate_model <- keras_model(inputs = model$input,
                                      outputs = get_layer(model,paste0("layer_",length(hiddennodes)))$output)
    intermediate_output <- predict(intermediate_model, train_mat)
    
    if(loss=="gamma"){
      model_bias_regulated <- glm(fold_data$trainset$response %>% data.matrix() ~ intermediate_output,  family= Gamma("log"))
    } else {
      model_bias_regulated <- glm(fold_data$trainset$response %>% data.matrix() ~ intermediate_output,  family= poisson("log"))
    } 
    
    # Calculate the loss on the validation set
    #intermediate_test_output <- predict(intermediate_model, test_mat)
    #predict_val <- predict(object=model_bias_regulated,newdata=list(intermediate_output=intermediate_test_output),type="response")
    
    predict_val <- predict_BRoption(list(intermediate_model,model_bias_regulated), test_mat, BR = TRUE, cann = FALSE)
    
  } else {
    # Calculate the loss on the validation set
    predict_val <- model %>% predict(test_mat)
  }
  
  if(loss=="gamma"){
    loss_val <-  dev_gamma(fold_data$testset$response %>% data.matrix(), predict_val, fold_data$testset$weights %>% data.matrix())
  } else {
    loss_val <-  dev_poiss_2(fold_data$testset$response %>% data.matrix(), predict_val)
  }
  
  # Return the parameters used, the validation error and the time elapsed
  results <- tibble(val_loss = loss_val, 
                    epochs_used = length(history$metrics$loss), 
                    time_elapsed = difftime(time1 = Sys.time(), time2 = time, units = "min"),
                    Portfolio_total_prediction = sum(predict_val),
                    Portfolio_total = sum(fold_data$testset$response),
                    Balance_ratio = Portfolio_total_prediction/Portfolio_total)
  # Also output model and variables used if asked (used for PDP)
  
  if(GLM_Bias_Regularization == TRUE){model_info <- list(intermediate_model, model_bias_regulated)} else {model_info <- model}
  output <- if(output_modelinfo == FALSE){results} else {
    list(results = results, model = model_info, 
         other_vars = other_vars, cat_vars = cat_vars, cann_variable = NULL)}
  return(output)
}

# Cross-validation run over all folds in data provided, with Autoencoded layer for the cat_vars provided
crossvalidation_run_AE <- function(fold_data, flags_list, autoencoder_trained,  
                                   cat_vars = list(), fold_var="fold_nr", id_var = "id"){
  
  # For time elapsed tracking
  time <- Sys.time()
  
  dropout <- flags_list[["dropout"]] 
  batch <- flags_list[["batch"]]
  hiddennodes <- flags_list[["hiddennodes"]]
  activation_h <- flags_list[["activation_h"]]
  activation_out <- flags_list[["activation_out"]]
  optimizer <- flags_list[["optimizer"]]
  epochs <- flags_list[["epochs"]]
  loss <- flags_list[["loss"]]
  
  # If class hiddennodes is list, unlist (used for tuning dropout rates)
  if(class(hiddennodes)=="list"){hiddennodes <- unlist(hiddennodes)}
  # Error message if a different number of dropouts and layers are provided
  if((length(hiddennodes)!=length(dropout)) & length(dropout) != 1){
    stop(paste(" The number of dropouts provided (",
               paste(dropout, collapse=", "),
               ") is not equal to the hidden nodes provided (",
               paste(hiddennodes, collapse=", "),")"))}
  # If one dropoutrate is provided, apply it for all hidden layers
  if(length(dropout) == 1){dropout <- rep(dropout,length(hiddennodes))}
  
  # Check which folds are in the training data
  folds <- unlist(unique(fold_data$trainset$data[,fold_var]))
  # Vector for validation set losses
  val_losses <- matrix(0, nrow = length(folds), ncol = 2)
  
  # Determine non-embedding variables
  other_vars <- setdiff(colnames(fold_data$trainset$data),c(cat_vars,id_var, fold_var))
  
  # Loop over all validation folds in the data, this can be replaced with a apply commando instead, but it seems to run very efficient as is
  for(f in folds){
    
    # Split the data into train and validation set
    val_split <- train_val(fold_data, f, fold_var)
    
    # Convert the categorical variables to a one=hot matrices
    train_cat_data <- lapply(cat_vars,function(var){
      val_split$trainset$data %>% 
        pull(var) %>% 
        as.data.table() %>% 
        one_hot(cols=".") %>% 
        data.matrix()
    })
    test_cat_data <- lapply(cat_vars,function(var){
      val_split$testset$data %>% 
        pull(var) %>% 
        as.data.table() %>% 
        one_hot(cols=".") %>% 
        data.matrix()
    })
    
    # Concatenate the categorical matrices for model input
    train_cat_data_concat <- do.call(cbind,train_cat_data)
    test_cat_data_concat <- do.call(cbind,test_cat_data)
    
    # Model setup in Keras
    model <- keras_model_sequential()
    
    # Make input layer for the non-embedding variables
    other_layer <- layer_input(shape=length(other_vars), dtype='float32',name="Design")
    
    # Autoencoder layer for the categorical variables
    # Initialize the weights from the trained autoencoder
    AE_input <- layer_input(shape = dim(train_cat_data_concat)[2])
    AE_layer <- AE_input %>% 
      layer_dense(units = ncol(autoencoder_trained[[1]]), weights = autoencoder_trained)
    
    # Concatenate the categorical autoencoded layer with the regular input layer
    net = c(other_layer,AE_layer) %>% layer_concatenate()
    
    # Add the hidden layers and output layer
    for(i in 1:length(hiddennodes)){
      net = net %>%
        layer_dense(units = hiddennodes[i], 
                    input_shape = ncol(val_split$trainset$data), 
                    kernel_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104), 
                    activation = activation_h)
      if(dropout[i]!=0){net = net %>% layer_dropout(rate=dropout[i])}
    }
    net = net %>% layer_dense(units = 1, activation = activation_out) 
    
    # Make the model
    model <- keras_model(inputs = c(other_layer, AE_input), outputs = net)
    
    if(loss == "poisson"){
      model %>% compile(
        loss = "poisson",
        optimizer = optimizer,
        metrics = metric_poisson_metric
      )
      # Weights for the poisson do not matter, so we take 1 for everything. Does not influence the outcome
      train_weights <- rep(1,nrow(val_split$trainset$data))
      val_weights <- rep(1,nrow(val_split$testset$data))
    } else if(loss == "gamma"){
      model %>% compile(
        loss = custom_metric('metric',gamma_metric()),
        optimizer = optimizer,
        metrics = custom_metric('metric',gamma_metric()),
        weighted_metrics = custom_metric('weighted_metric', gamma_metric())
      )
      # Setting the weights for the Gamma to have weighted loss
      train_weights <- val_split$trainset$weights %>% data.matrix()
      val_weights <- val_split$testset$weights %>% data.matrix()
    }
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20) #The val_loss is weighted if weights are available
    
    # Transform to list of matrices for all input layers
    train_mat <- list(val_split$trainset$data %>% select(other_vars) %>% data.matrix(),
                      train_cat_data_concat)
    test_mat <- list(val_split$testset$data %>% select(other_vars) %>% data.matrix(),
                     test_cat_data_concat)
    
    # Fit the model on the train data with the test/val data for early stopping 
    history <- model %>%
      fit(x = train_mat, 
          y = val_split$trainset$response %>% data.matrix(),
          sample_weight =  train_weights,
          epochs = epochs,
          batch_size = batch,
          callbacks = list(early_stop),
          validation_data = list(x_val = test_mat, 
                                 y_val = val_split$testset$response %>% data.matrix(),
                                 val_sample_weight = val_weights),
          verbose=0
      )
    
    # Calculate the loss on the validation set
    predict_val <- model %>% predict(test_mat)
    val_losses[match(f,folds),1] <- ifelse(loss=="gamma", 
                                           dev_gamma(val_split$testset$response %>% data.matrix(), predict_val, val_split$testset$weights %>% data.matrix()), 
                                           dev_poiss_2(val_split$testset$response %>% data.matrix(), predict_val))
    val_losses[match(f,folds),2] <- length(history$metrics$loss)
  }
  
  # Return the average validation loss, the run time for info, and the used parameters
  return(tibble("validation_error" = mean(val_losses[,1]),
                "cross_run_time" = difftime(time1 = Sys.time(), time2 = time, units = "min"),
                "epochs_used" = list(val_losses[,2]), 
                "dropout" = list(dropout), 
                "batch" = batch, 
                "hiddennodes" = list(hiddennodes), 
                "activation_h" = activation_h, 
                "activation_out" = activation_out, 
                "optimizer" = optimizer, 
                "epochs" = epochs, 
                "loss" = loss))
}

# Single CANN run with embedding layers for the cat_vars provided
single_CANN_run_Embedding <- function(fold_data, flags_list, random_val_split = 0, 
                                      embedding_output_dim = 1,  cat_vars = list(), cann_variable = "prediction", 
                                      trainable_output = FALSE, fold_var = "fold_nr", id_var = "id", 
                                      early_stopping = TRUE, GLM_Bias_Regularization = FALSE, output_modelinfo = FALSE){
  # For time elapsed tracking -----
  time <- Sys.time()
  
  dropout <- flags_list[["dropout"]] 
  batch <- flags_list[["batch"]]
  hiddennodes <- flags_list[["hiddennodes"]]
  activation_h <- flags_list[["activation_h"]]
  activation_out <- flags_list[["activation_out"]]
  optimizer <- flags_list[["optimizer"]]
  epochs <- flags_list[["epochs"]]
  loss <- flags_list[["loss"]]
  
  # If class hiddennodes is list, unlist (used for tuning dropout rates)
  if(class(hiddennodes)=="list"){hiddennodes <- unlist(hiddennodes)}
  # Error message if a different number of dropouts and layers are provided
  if((length(hiddennodes)!=length(dropout)) & length(dropout) != 1){
    stop(paste(" The number of dropouts provided (",
               paste(dropout, collapse=", "),
               ") is not equal to the hidden nodes provided (",
               paste(hiddennodes, collapse=", "),")"))}
  # If one dropoutrate is provided, apply it for all hidden layers
  if(length(dropout) == 1){dropout <- rep(dropout,length(hiddennodes))}
  
  # Remove the fold_nr and id variables if in the data
  fold_data$trainset$data <- fold_data$trainset$data %>% select(!any_of(c('id','fold_nr')))
  fold_data$testset$data <- fold_data$testset$data %>% select(!any_of(c('id','fold_nr')))
  
  # If embedding output size is length 1, make vector
  if(length(embedding_output_dim)==1){
    embedding_output_dim <- rep(embedding_output_dim,length(cat_vars))
  } else if (length(embedding_output_dim) != length(cat_vars)){
    stop(paste("Embedding output dimension vector is not the same as the number of variables to embed (dimensions =",
               length(embedding_output_dim), "but", length(cat_vars), "variables to embed provided"))}
  
  # Model setup in Keras -----
  model <- keras_model_sequential()
  
  # Determine non-embedding variables
  other_vars <- setdiff(colnames(fold_data$trainset$data),c(cat_vars,fold_var,cann_variable,id_var))
  # Make input layer for the non-embedding variables
  other_layer <- layer_input(shape=length(other_vars), dtype='float32',name="Design")
  # Input layer for the predictions from another model
  cann_input <- layer_input(shape=c(1),dtype='float32',name='cann_input')
  
  # Make the embedding layers for each embedding variable
  cat_input_layers <- lapply(cat_vars, function(x){
    layer_input(
      shape=c(1),
      dtype='float32',
      name = paste(x,"_input",sep = ""))})
  emb_layers <- lapply(cat_input_layers, function(x){
    x %>% layer_embedding(
      input_dim = nrow(unique(fold_data$trainset$data[,sub("\\_.*", "", x$name)]))+1, #WHY??? 
      output_dim = embedding_output_dim[match(sub("\\_.*", "", x$name), cat_vars)],
      input_length = 1 ) %>% 
      layer_flatten()})
  all_input_layers <- c(other_layer,unlist(emb_layers))
  
  # Concatenate the embedded layers with the regular input layers, and add the hidden layers and output layer
  net = all_input_layers %>% layer_concatenate()
  for(i in 1:length(hiddennodes)){
    net = net %>%
      layer_dense(units = hiddennodes[i], 
                  input_shape = ncol(fold_data$trainset$data), 
                  name = paste0("layer_",i), 
                  activation = activation_h)
    if(dropout[i]!=0){net = net %>% layer_dropout(rate=dropout[i])}
  }
  net = net %>% layer_dense(units = 1, activation = "linear", name = "pre_cann_layer") 
  
  # Add the combination layer
  response = list(net, cann_input) %>% 
    layer_concatenate(name = "cann_combination") %>% 
    layer_dense(units = 1L, 
                input_shape = 2L,
                trainable = trainable_output, 
                weights = list(matrix(c(1,1), nrow=2), array(c(0))),
                activation = activation_out,
                name = "cann_ouput")
  
  # Make the model
  model <- keras_model(inputs = c(other_layer, unlist(cat_input_layers), cann_input), outputs = response)
  if(loss == "poisson"){
    model %>% compile(
      loss = "poisson",
      optimizer = optimizer,
      metrics = metric_poisson_metric
    )
    # Weights for the poisson do not matter, so we take 1 for everything. Does not influence the outcome
    train_weights <- rep(1,nrow(fold_data$trainset$data))
    val_weights <- rep(1,nrow(fold_data$testset$data))
  } else if(loss == "gamma"){
    model %>% compile(
      loss = custom_metric('metric',gamma_metric()),
      optimizer = optimizer,
      metrics = custom_metric('metric',gamma_metric()),
      weighted_metrics = custom_metric('weighted_metric', gamma_metric())
    )
    # Setting the weights for the Gamma to have weighted loss
    train_weights <- fold_data$trainset$weights %>% data.matrix()
    val_weights <- fold_data$testset$weights %>% data.matrix()
  }
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20) #The val_loss is weighted if weights are avaliable
  
  # Transform CANN input -----
  fold_data$trainset$data <- fold_data$trainset$data %>%  mutate_at(cann_variable, log)
  fold_data$testset$data <- fold_data$testset$data %>%  mutate_at(cann_variable, log)
  
  # Transform to list of matrices for all input layers
  train_mat <- lapply(c(list(other_vars),unlist(cat_vars), list(cann_variable)),  
                      function(data, x){ 
                        data[,x]
                      }, data = fold_data$trainset$data %>% data.matrix())
  test_mat <- lapply(c(list(other_vars),unlist(cat_vars), list(cann_variable)),  
                     function(data, x){ 
                       data[,x]
                     }, data = fold_data$testset$data %>% data.matrix())
  
  # Fit the model
  history <- model %>%
    fit(x = train_mat, 
        y = fold_data$trainset$response %>% data.matrix(),
        sample_weight =  train_weights,
        epochs = epochs,
        batch_size = batch,
        callbacks = if(early_stopping==TRUE){list(early_stop)},
        validation_split = random_val_split,
        validation_data = if(random_val_split == 0){list(x_val = test_mat, 
                                                         y_val = fold_data$testset$response %>% data.matrix(),
                                                         val_sample_weight = val_weights)},
        verbose=0)
  # Bias Regularization Part -----
  if(GLM_Bias_Regularization == TRUE){
    # Use the weights of the last hidden layer as inputs for a glm
    intermediate_model <- keras_model(inputs = model$input,
                                      outputs = get_layer(model,paste0("layer_",length(hiddennodes)))$output)
    #outputs = get_layer(model,"pre_cann_layer")$output)
    intermediate_output <- predict(intermediate_model, train_mat)
    train_cann_input <- fold_data$trainset$data %>% select(cann_variable) %>% data.matrix()
    
    if(loss=="gamma"){
      model_bias_regulated <- glm((fold_data$trainset$response %>% data.matrix()) ~ intermediate_output + train_cann_input,  
                                  family= Gamma("log"))
    } else {
      if(cann_variable == "expo"){
        model_bias_regulated <- glm((fold_data$trainset$response %>% data.matrix()) ~ intermediate_output + offset(train_cann_input),  
                                    family= poisson("log"))
      } else {
        model_bias_regulated <- glm((fold_data$trainset$response %>% data.matrix()) ~ intermediate_output + train_cann_input,  
                                    family= poisson("log"))
      }
    }
    # Calculate the loss on the validation set
    intermediate_test_output <- predict(intermediate_model, test_mat)
    predict_val <- predict(object=model_bias_regulated,newdata=list(intermediate_output = intermediate_test_output, train_cann_input = fold_data$testset$data %>% select(cann_variable) %>% data.matrix()),type="response")
  } else {
    # Calculate the loss on the validation set
    predict_val <- model %>% predict(test_mat)
  }
  
  loss_val <- ifelse(loss=="gamma", 
                     dev_gamma(fold_data$testset$response %>% data.matrix(), predict_val, fold_data$testset$weights %>% data.matrix()), 
                     dev_poiss_2(fold_data$testset$response %>% data.matrix(), predict_val))
  
  # Return the parameters used, the validation error and the time elapsed
  results <- tibble(val_loss = loss_val, 
                    epochs_used = length(history$metrics$loss), 
                    time_elapsed = difftime(time1 = Sys.time(), time2 = time, units = "min"),
                    Portfolio_total_prediction = sum(predict_val),
                    Portfolio_total = sum(fold_data$testset$response),
                    Balance_ratio = Portfolio_total_prediction/Portfolio_total)
  # Also output model and variables used if asked (used for PDP)
  output <- if(output_modelinfo == FALSE){results} else {
    list(results = results, model = if(GLM_Bias_Regularization == TRUE){list(intermediate_model, model_bias_regulated)} else {model}, 
         other_vars = other_vars, cat_vars = cat_vars, cann_variable = cann_variable)}
  return(output)
}

# Cross-validation run over all folds in data provided, with Embedding layers for the cat_vars provided
crossvalidation_CANN_run_Embedding <- function(fold_data, flags_list, embedding_output_dim = 1,  cat_vars = list(), 
                                               cann_variable = "prediction", trainable_output = FALSE, fold_var="fold_nr", id_var = "id"){
  
  # For time elapsed tracking
  time <- Sys.time()
  
  dropout <- flags_list[["dropout"]] 
  batch <- flags_list[["batch"]]
  hiddennodes <- flags_list[["hiddennodes"]]
  activation_h <- flags_list[["activation_h"]]
  activation_out <- flags_list[["activation_out"]]
  optimizer <- flags_list[["optimizer"]]
  epochs <- flags_list[["epochs"]]
  loss <- flags_list[["loss"]]
  
  # If class hiddennodes is list, unlist (used for tuning dropout rates)
  if(class(hiddennodes)=="list"){hiddennodes <- unlist(hiddennodes)}
  # Error message if a different number of dropouts and layers are provided
  if((length(hiddennodes)!=length(dropout)) & length(dropout) != 1){
    stop(paste(" The number of dropouts provided (",
               paste(dropout, collapse=", "),
               ") is not equal to the hidden nodes provided (",
               paste(hiddennodes, collapse=", "),")"))}
  # If one dropoutrate is provided, apply it for all hidden layers
  if(length(dropout) == 1){dropout <- rep(dropout,length(hiddennodes))}
  
  # If embedding output size is length 1, make vector
  if(length(embedding_output_dim)==1){
    embedding_output_dim <- rep(1,length(cat_vars))
  } else if (length(embedding_output_dim) != length(cat_vars)){
    stop(paste("Embedding output dimension vector is not the same as the number of variables to embed (dimensions =",
               length(embedding_output_dim), "but", length(cat_vars), "variables to embed provided"))}
  
  # Check which folds are in the training data
  folds <- unlist(unique(fold_data$trainset$data[,fold_var]))
  # Vector for validation set losses
  val_losses <- matrix(0, nrow = length(folds), ncol = 2)
  
  # Determine non-embedding variables
  other_vars <- setdiff(colnames(fold_data$trainset$data),c(cat_variables,fold_var,cann_variable,id_var))
  # Make input layer for the non-embedding variables
  other_layer <- layer_input(shape=length(other_vars), dtype='float32',name="Design")
  # Input layer for the predictions from another model
  cann_input <- layer_input(shape=c(1),dtype='float32',name='cann_input')
  
  # Make the embedding layers for each embedding variable
  cat_input_layers <- lapply(cat_vars, function(x){
    layer_input(
      shape=c(1),
      dtype='float32',
      name = paste(x,"_input",sep = ""))})
  emb_layers <- lapply(cat_input_layers, function(x){
    x %>% layer_embedding(
      input_dim = nrow(unique(fold_data$trainset$data[,sub("\\_.*", "", x$name)]))+1, #WHY??? 
      output_dim = embedding_output_dim[match(sub("\\_.*", "", x$name), cat_vars)],
      input_length = 1 ) %>% 
      layer_flatten()})
  all_input_layers <- c(other_layer,unlist(emb_layers))
  
  # Loop over all validation folds in the data, this can be replaced with a apply commando instead, but it seems to run very efficient as is
  for(f in folds){
    
    # Split the data into train and validation set
    val_split <- train_val(fold_data, f, fold_var)
    
    # Concatenate the embedded layers with the regular input layers, and add the hidden layers and output layer
    net = all_input_layers %>% layer_concatenate()
    for(i in 1:length(hiddennodes)){
      net = net %>%
        layer_dense(units = hiddennodes[i], 
                    input_shape = ncol(val_split$trainset$data), 
                    kernel_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104), 
                    activation = activation_h)
      if(dropout[i]!=0){net = net %>% layer_dropout(rate=dropout[i])}
    }
    net = net %>% layer_dense(units = 1, activation = "linear") 
    
    # Add the combination layer
    response = list(net, cann_input) %>% 
      layer_add() %>% 
      layer_dense(units = 1,  
                  trainable = trainable_output, 
                  weights=list(array(1,dim=c(1,1)), array(0,dim=c(1))), 
                  activation = activation_out)
    
    # Make the model
    model <- keras_model(inputs = c(other_layer,unlist(cat_input_layers), cann_input), outputs = response)
    
    if(loss == "poisson"){
      model %>% compile(
        loss = "poisson",
        optimizer = optimizer,
        metrics = metric_poisson_metric
      )
      # Weights for the poisson do not matter, so we take 1 for everything. Does not influence the outcome
      train_weights <- rep(1,nrow(val_split$trainset$data))
      val_weights <- rep(1,nrow(val_split$testset$data))
    } else if(loss == "gamma"){
      model %>% compile(
        loss = custom_metric('metric',gamma_metric()),
        optimizer = optimizer,
        metrics = custom_metric('metric',gamma_metric()),
        weighted_metrics = custom_metric('weighted_metric', gamma_metric())
      )
      # Setting the weights for the Gamma to have weighted loss
      train_weights <- val_split$trainset$weights %>% data.matrix()
      val_weights <- val_split$testset$weights %>% data.matrix()
    }
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20) #The val_loss is weighted if weights are avaliable
    
    # Transform to list of matrices for all input layers
    train_mat <- lapply(c(list(other_vars),unlist(cat_vars), list(cann_variable)),  
                        function(data, x){ 
                          data[,x]
                        }, data = val_split$trainset$data %>% mutate_at(cann_variable, log) %>% data.matrix())
    test_mat <- lapply(c(list(other_vars),unlist(cat_vars), list(cann_variable)),  
                       function(data, x){ 
                         data[,x]
                       }, data = val_split$testset$data %>% mutate_at(cann_variable, log) %>% data.matrix())
    
    # Fit the model on the train data with the test/val data for early stopping 
    history <- model %>%
      fit(x = train_mat, 
          y = val_split$trainset$response %>% data.matrix(),
          sample_weight =  train_weights,
          epochs = epochs,
          batch_size = batch,
          callbacks = list(early_stop),
          validation_data = list(x_val = test_mat, 
                                 y_val = val_split$testset$response %>% data.matrix(),
                                 val_sample_weight = val_weights),
          verbose=0
      )
    
    # Calculate the loss on the validation set
    predict_val <- model %>% predict(test_mat)
    val_losses[match(f,folds),1] <- ifelse(loss=="gamma", 
                                           dev_gamma(val_split$testset$response %>% data.matrix(), predict_val, val_split$testset$weights %>% data.matrix()), 
                                           dev_poiss_2(val_split$testset$response %>% data.matrix(), predict_val))
    val_losses[match(f,folds),2] <- length(history$metrics$loss)
  }
  
  # Return the average validation loss, the run time for info, and the used parameters
  return(tibble("validation_error" = mean(val_losses[,1]),
                "cross_run_time" = difftime(time1 = Sys.time(), time2 = time, units = "min"),
                "epochs_used" = list(val_losses[,2]), 
                "dropout" = list(dropout), 
                "batch" = batch, 
                "hiddennodes" = list(hiddennodes), 
                "activation_h" = activation_h, 
                "activation_out" = "activation_out", 
                "optimizer" = optimizer, 
                "epochs" = epochs, 
                "loss" = loss))
}

# Single CANN run with AutoEncoder for the cat_vars provided
single_CANN_run_AE <- function(fold_data, flags_list, random_val_split = 0,
                               autoencoder_trained,
                               cat_vars = list(), cann_variable = "prediction", 
                               trainable_output = FALSE, fold_var = "fold_nr", id_var = "id", 
                               early_stopping = TRUE, GLM_Bias_Regularization = FALSE, output_modelinfo = FALSE){
  # For time elapsed tracking
  time <- Sys.time()
  
  dropout <- flags_list[["dropout"]] 
  batch <- flags_list[["batch"]]
  hiddennodes <- flags_list[["hiddennodes"]]
  activation_h <- flags_list[["activation_h"]]
  activation_out <- flags_list[["activation_out"]]
  optimizer <- flags_list[["optimizer"]]
  epochs <- flags_list[["epochs"]]
  loss <- flags_list[["loss"]]
  
  # If class hiddennodes is list, unlist (used for tuning dropout rates)
  if(class(hiddennodes)=="list"){hiddennodes <- unlist(hiddennodes)}
  # Error message if a different number of dropouts and layers are provided
  if((length(hiddennodes)!=length(dropout)) & length(dropout) != 1){
    stop(paste(" The number of dropouts provided (",
               paste(dropout, collapse=", "),
               ") is not equal to the hidden nodes provided (",
               paste(hiddennodes, collapse=", "),")"))}
  # If one dropoutrate is provided, apply it for all hidden layers
  if(length(dropout) == 1){dropout <- rep(dropout,length(hiddennodes))}
  
  # Remove the fold_nr and id variables if in the data
  fold_data$trainset$data <- fold_data$trainset$data %>% select(!any_of(c('id','fold_nr')))
  fold_data$testset$data <- fold_data$testset$data %>% select(!any_of(c('id','fold_nr')))

  # Convert the categorical variables to a one=hot matrices
  train_cat_data <- lapply(cat_vars,function(var){
    fold_data$trainset$data %>% 
      pull(var) %>% 
      as.data.table() %>% 
      one_hot(cols=".") %>% 
      data.matrix()
  })
  test_cat_data <- lapply(cat_vars,function(var){
    fold_data$testset$data %>% 
      pull(var) %>% 
      as.data.table() %>% 
      one_hot(cols=".") %>% 
      data.matrix()
  })
  
  # Concatenate the categorical matrices for model input
  train_cat_data_concat <- do.call(cbind,train_cat_data)
  test_cat_data_concat <- do.call(cbind,test_cat_data)

  # Model setup in Keras
  model <- keras_model_sequential()
  
  # Determine non-embedding variables
  other_vars <- setdiff(colnames(fold_data$trainset$data),c(cat_vars,fold_var,cann_variable,id_var))
  # Make input layer for the non-embedding variables
  other_layer <- layer_input(shape=length(other_vars), dtype='float32',name="Design")
  # Input layer for the predictions from another model
  cann_input <- layer_input(shape=c(1),dtype='float32',name='cann_input')
  
  # Autoencoder layer for the categorical variables
  # Initialize the weights from the trained autoencoder
  AE_input <- layer_input(shape = dim(train_cat_data_concat)[2])
  AE_layer <- AE_input %>% 
    layer_dense(units = ncol(autoencoder_trained[[1]]), weights = autoencoder_trained)
  
  # Concatenate the categorical autoencoded layer with the regular input layer, and add the hidden layers and output layer
  net = c(other_layer,AE_layer) %>% layer_concatenate()
  
  for(i in 1:length(hiddennodes)){
    net = net %>%
      layer_dense(units = hiddennodes[i], 
                  input_shape = ncol(fold_data$trainset$data), 
                  name = paste0("layer_",i), 
                  activation = activation_h)
    if(dropout[i]!=0){net = net %>% layer_dropout(rate=dropout[i])}
  }
  net = net %>% layer_dense(units = 1, activation = "linear", name = "pre_cann_layer") 
  
  # Add the combination layer
  response = list(net, cann_input) %>% 
    layer_concatenate(name = "cann_combination") %>% 
    layer_dense(units = 1L, 
                input_shape = 2L,
                trainable = trainable_output, 
                weights = list(matrix(c(1,1), nrow=2), array(c(0))),
                activation = activation_out,
                name = "cann_ouput")
  
  # Make the model
  model <- keras_model(inputs = c(other_layer, AE_input, cann_input), outputs = response)
  if(loss == "poisson"){
    model %>% compile(
      loss = "poisson",
      optimizer = optimizer,
      metrics = metric_poisson_metric
    )
    # Weights for the poisson do not matter, so we take 1 for everything. Does not influence the outcome
    train_weights <- rep(1,nrow(fold_data$trainset$data))
    val_weights <- rep(1,nrow(fold_data$testset$data))

  } else if(loss == "gamma"){
    model %>% compile(
      loss = custom_metric('metric',gamma_metric()),
      optimizer = optimizer,
      metrics = custom_metric('metric',gamma_metric()),
      weighted_metrics = custom_metric('weighted_metric', gamma_metric())
    )
    # Setting the weights for the Gamma to have weighted loss
    train_weights <- fold_data$trainset$weights %>% data.matrix()
    val_weights <- fold_data$testset$weights %>% data.matrix()
  }
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20) #The val_loss is weighted if weights are avaliable
  
  # Transform CANN input
  fold_data$trainset$data <- fold_data$trainset$data %>%  mutate_at(cann_variable, log)
  fold_data$testset$data <- fold_data$testset$data %>%  mutate_at(cann_variable, log)

  # Transform to list of matrices for all input layers
  train_mat <- list(fold_data$trainset$data %>% select(other_vars) %>% data.matrix(),
                    train_cat_data_concat,
                    fold_data$trainset$data %>% select(cann_variable) %>% data.matrix())
  test_mat <- list(fold_data$testset$data %>% select(other_vars) %>% data.matrix(),
                   test_cat_data_concat,
                   fold_data$testset$data %>% select(cann_variable) %>% data.matrix())
  
  # Fit the model
  history <- model %>%
    fit(x = train_mat, 
        y = fold_data$trainset$response %>% data.matrix(),
        sample_weight =  train_weights,
        epochs = epochs,
        batch_size = batch,
        callbacks = if(early_stopping==TRUE){list(early_stop)},
        validation_split = random_val_split,
        validation_data = if(random_val_split == 0){list(x_val = test_mat, 
                                                         y_val = fold_data$testset$response %>% data.matrix(),
                                                         val_sample_weight = val_weights)},
        verbose=0)
  
  # Bias Regularization Part
  if(GLM_Bias_Regularization == TRUE){
    # Use the weights of the last hidden layer as inputs for a glm
    intermediate_model <- keras_model(inputs = model$input,
                                      outputs = get_layer(model,paste0("layer_",length(hiddennodes)))$output)
    #outputs = get_layer(model,"pre_cann_layer")$output)
    intermediate_output <- predict(intermediate_model, train_mat)
    train_cann_input <- fold_data$trainset$data %>% select(cann_variable) %>% data.matrix()
    
    if(loss=="gamma"){
      model_bias_regulated <- glm((fold_data$trainset$response %>% data.matrix()) ~ intermediate_output + train_cann_input,  
                                  family= Gamma("log"))
    } else {
      if(cann_variable == "expo"){
        model_bias_regulated <- glm((fold_data$trainset$response %>% data.matrix()) ~ intermediate_output + offset(train_cann_input),  
                                    family= poisson("log"))
      } else {
        model_bias_regulated <- glm((fold_data$trainset$response %>% data.matrix()) ~ intermediate_output + train_cann_input,  
                                    family= poisson("log"))
      }
    }
    # Calculate the loss on the validation set
    intermediate_test_output <- predict(intermediate_model, test_mat)
    predict_val <- predict(object=model_bias_regulated,newdata=list(intermediate_output = intermediate_test_output, train_cann_input = fold_data$testset$data %>% select(cann_variable) %>% data.matrix()),type="response")
  } else {
    # Calculate the loss on the validation set
    predict_val <- model %>% predict(test_mat)
  }
  
  loss_val <- ifelse(loss=="gamma", 
                     dev_gamma(fold_data$testset$response %>% data.matrix(), predict_val, fold_data$testset$weights %>% data.matrix()), 
                     dev_poiss_2(fold_data$testset$response %>% data.matrix(), predict_val))
  
  # Return the parameters used, the validation error and the time elapsed
  results <- tibble(val_loss = loss_val, 
                    epochs_used = length(history$metrics$loss), 
                    time_elapsed = difftime(time1 = Sys.time(), time2 = time, units = "min"),
                    Portfolio_total_prediction = sum(predict_val),
                    Portfolio_total = sum(fold_data$testset$response),
                    Balance_ratio = Portfolio_total_prediction/Portfolio_total)
  # Also output model and variables used if asked (used for PDP)
  output <- if(output_modelinfo == FALSE){results} else {
    list(results = results, model = if(GLM_Bias_Regularization == TRUE){list(intermediate_model, model_bias_regulated)} else {model}, 
         other_vars = other_vars, cat_vars = cat_vars, cann_variable = cann_variable)}
  return(output)
}

# Cross-validation run over all folds in data provided, with Embedding layers for the cat_vars provided
crossvalidation_CANN_run_AE <- function(fold_data, flags_list, autoencoder_trained,  cat_vars = list(), 
                                        cann_variable = "prediction", trainable_output = FALSE, fold_var="fold_nr", id_var = "id"){
  
  # For time elapsed tracking
  time <- Sys.time()
  
  dropout <- flags_list[["dropout"]] 
  batch <- flags_list[["batch"]]
  hiddennodes <- flags_list[["hiddennodes"]]
  activation_h <- flags_list[["activation_h"]]
  activation_out <- flags_list[["activation_out"]]
  optimizer <- flags_list[["optimizer"]]
  epochs <- flags_list[["epochs"]]
  loss <- flags_list[["loss"]]
  
  # If class hiddennodes is list, unlist (used for tuning dropout rates)
  if(class(hiddennodes)=="list"){hiddennodes <- unlist(hiddennodes)}
  # Error message if a different number of dropouts and layers are provided
  if((length(hiddennodes)!=length(dropout)) & length(dropout) != 1){
    stop(paste(" The number of dropouts provided (",
               paste(dropout, collapse=", "),
               ") is not equal to the hidden nodes provided (",
               paste(hiddennodes, collapse=", "),")"))}
  # If one dropoutrate is provided, apply it for all hidden layers
  if(length(dropout) == 1){dropout <- rep(dropout,length(hiddennodes))}
  
  # Check which folds are in the training data
  folds <- unlist(unique(fold_data$trainset$data[,fold_var]))
  # Vector for validation set losses
  val_losses <- matrix(0, nrow = length(folds), ncol = 2)
  
  # Determine non-autoencoded variables
  other_vars <- setdiff(colnames(fold_data$trainset$data),c(cat_variables,fold_var,cann_variable,id_var))
  
  # Loop over all validation folds in the data, this can be replaced with a apply commando instead, but it seems to run very efficient as is
  for(f in folds){
    
    # Split the data into train and validation set
    val_split <- train_val(fold_data, f, fold_var)
    
    # Convert the categorical variables to a one=hot matrices
    train_cat_data <- lapply(cat_vars,function(var){
      val_split$trainset$data %>% 
        pull(var) %>% 
        as.data.table() %>% 
        one_hot(cols=".") %>% 
        data.matrix()
    })
    test_cat_data <- lapply(cat_vars,function(var){
      val_split$testset$data %>% 
        pull(var) %>% 
        as.data.table() %>% 
        one_hot(cols=".") %>% 
        data.matrix()
    })
    
    # Concatenate the categorical matrices for model input
    train_cat_data_concat <- do.call(cbind,train_cat_data)
    test_cat_data_concat <- do.call(cbind,test_cat_data)
    
    # Model setup in Keras
    model <- keras_model_sequential()
    
    # Make input layer for the non-embedding variables
    other_layer <- layer_input(shape=length(other_vars), dtype='float32',name="Design")
    # Input layer for the predictions from another model
    cann_input <- layer_input(shape=c(1),dtype='float32',name='cann_input')
    
    # Autoencoder layer for the categorical variables
    # Initialize the weights from the trained autoencoder
    AE_input <- layer_input(shape = dim(train_cat_data_concat)[2])
    AE_layer <- AE_input %>% 
      layer_dense(units = ncol(autoencoder_trained[[1]]), weights = autoencoder_trained)
    
    # Concatenate the categorical autoencoded layer with the regular input layer, and add the hidden layers and output layer
    net = c(other_layer,AE_layer) %>% layer_concatenate()
    
    # Concatenate the autoencoded layer with the regular input layers, and add the hidden layers and output layer
    for(i in 1:length(hiddennodes)){
      net = net %>%
        layer_dense(units = hiddennodes[i], 
                    input_shape = ncol(val_split$trainset$data), 
                    kernel_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 104), 
                    activation = activation_h)
      if(dropout[i]!=0){net = net %>% layer_dropout(rate=dropout[i])}
    }
    net = net %>% layer_dense(units = 1, activation = "linear") 
    
    # Add the combination layer
    response = list(net, cann_input) %>% 
      layer_add() %>% 
      layer_dense(units = 1,  
                  trainable = trainable_output, 
                  weights=list(array(1,dim=c(1,1)), array(0,dim=c(1))), 
                  activation = activation_out)
    
    # Make the model
    model <- keras_model(inputs = c(other_layer, AE_input, cann_input), outputs = response)
    
    if(loss == "poisson"){
      model %>% compile(
        loss = "poisson",
        optimizer = optimizer,
        metrics = metric_poisson_metric
      )
      # Weights for the poisson do not matter, so we take 1 for everything. Does not influence the outcome
      train_weights <- rep(1,nrow(val_split$trainset$data))
      val_weights <- rep(1,nrow(val_split$testset$data))
    } else if(loss == "gamma"){
      model %>% compile(
        loss = custom_metric('metric',gamma_metric()),
        optimizer = optimizer,
        metrics = custom_metric('metric',gamma_metric()),
        weighted_metrics = custom_metric('weighted_metric', gamma_metric())
      )
      # Setting the weights for the Gamma to have weighted loss
      train_weights <- val_split$trainset$weights %>% data.matrix()
      val_weights <- val_split$testset$weights %>% data.matrix()
    }
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20) #The val_loss is weighted if weights are available
    
    # Transform CANN input
    val_split$trainset$data <- val_split$trainset$data %>%  mutate_at(cann_variable, log)
    val_split$testset$data <- val_split$testset$data %>%  mutate_at(cann_variable, log)
    
    # Transform to list of matrices for all input layers
    train_mat <- list(val_split$trainset$data %>% select(other_vars) %>% data.matrix(),
                      train_cat_data_concat,
                      val_split$trainset$data %>% select(cann_variable) %>% data.matrix())
    test_mat <- list(val_split$testset$data %>% select(other_vars) %>% data.matrix(),
                     test_cat_data_concat,
                     val_split$testset$data %>% select(cann_variable) %>% data.matrix())
    
    # Fit the model on the train data with the test/val data for early stopping 
    history <- model %>%
      fit(x = train_mat, 
          y = val_split$trainset$response %>% data.matrix(),
          sample_weight =  train_weights,
          epochs = epochs,
          batch_size = batch,
          callbacks = list(early_stop),
          validation_data = list(x_val = test_mat, 
                                 y_val = val_split$testset$response %>% data.matrix(),
                                 val_sample_weight = val_weights),
          verbose=0
      )
    
    # Calculate the loss on the validation set
    predict_val <- model %>% predict(test_mat)
    val_losses[match(f,folds),1] <- ifelse(loss=="gamma", 
                                           dev_gamma(val_split$testset$response %>% data.matrix(), predict_val, val_split$testset$weights %>% data.matrix()), 
                                           dev_poiss_2(val_split$testset$response %>% data.matrix(), predict_val))
    val_losses[match(f,folds),2] <- length(history$metrics$loss)
  }
  
  # Return the average validation loss, the run time for info, and the used parameters
  return(tibble("validation_error" = mean(val_losses[,1]),
                "cross_run_time" = difftime(time1 = Sys.time(), time2 = time, units = "min"),
                "epochs_used" = list(val_losses[,2]), 
                "dropout" = list(dropout), 
                "batch" = batch, 
                "hiddennodes" = list(hiddennodes), 
                "activation_h" = activation_h, 
                "activation_out" = activation_out, 
                "optimizer" = optimizer, 
                "epochs" = epochs, 
                "loss" = loss))
}

# Run over al test fold, over a set of tuning parameter option (one at a time), in cross validation, option to use Embedding Layers
alltestfolds_tuningrun <- function(all_folds_data, flags_list, flags_are_grid = FALSE ,repeating_runs = 1, 
                                   embedding = FALSE, embedding_output_dim = 1, 
                                   autoencoders = FALSE, autoencoder_trained = list(),
                                   cat_vars = list(), 
                                   CANN = FALSE, cann_variable = "prediction", 
                                   trainable_output = FALSE, fold_var="fold_nr"){
  
  # Number of test folds in the data
  nbr_of_testfolds <- length(all_folds_data)
  print(paste("Number of test folds in the data =", nbr_of_testfolds))
  
  if(!flags_are_grid){
    
    # Which parameter will be tuned, stop if multiple have vectors assigned
    tuning_variable <- names(flags_list)[which(lapply(flags_list,length)!=1)]
    #if(length(tuning_variable)>1){
    #  stop(paste0("Multiple tuning parameters have a vector of values; ", paste(tuning_variable,collapse=", ")))}
    print(paste("The following parameters will be tuned:", toString(tuning_variable)))
    
    # Grid of test fold and tuning parameter options
    parameters <- expand.grid(c(list(1:nbr_of_testfolds), flags_list, list(1:repeating_runs)))
    colnames(parameters) <- c("folds", names(flags_list), "repeating_runs")
    print(paste("There will be",nrow(parameters),"different combinations tested in crossvalidation"))
    
  } else {
    
    # For tuning a grid of parameters
    tuning_variable <- names(flags_list)[sapply(flags_list, function(x) length(unique(x))>1)]
    print(paste("The following parameters will be tuned:", toString(tuning_variable)))
    parameters <- flags_list %>% 
      mutate(folds = list(1:nbr_of_testfolds)) %>% unnest(folds) %>% 
      mutate(repeating_runs = list(1:repeating_runs)) %>% unnest(repeating_runs) %>% 
      select(folds, optimizer, batch, activation_h,hiddennodes,dropout,activation_out, epochs, loss, repeating_runs) %>% 
      as.data.table
    
  }
  
  time <- Sys.time()
  # For each parameters row apply the cross validation function
  # This will run a 5-fold cross validation run for each parameter options, for each test fold
  if(embedding == FALSE & CANN == FALSE & autoencoders == FALSE){
    results <- apply(parameters, 1, function(x){
      print(paste("Now running testfold:", as.numeric(x[1]), ", and tuning parameter options:", toString(x[2:(length(x)-1)]), ", repeating option =", as.numeric(x[length(x)])))
      cross_run_results <- crossvalidation_run(all_folds_data[[as.numeric(x[1])]], 
                                               x[2:(length(x)-1)])
      return(cross_run_results %>% bind_cols("test_fold" = as.numeric(x[1]), "repeating_runs" = as.numeric(x[length(x)])))})
  } else if (embedding == TRUE & CANN == FALSE & autoencoders == FALSE) {
    results <- apply(parameters, 1, function(x){
      print(paste("Now running testfold:", as.numeric(x[1]), ", and tuning parameter options:", toString(x[2:(length(x)-1)]), ", repeating option =", as.numeric(x[length(x)])))
      cross_run_results <- crossvalidation_run_Embedding(all_folds_data[[as.numeric(x[1])]], 
                                                         x[2:(length(x)-1)],
                                                         embedding_output_dim = embedding_output_dim,
                                                         cat_vars = cat_vars)
      return(cross_run_results %>% bind_cols("test_fold" = as.numeric(x[1]), "repeating_runs" = as.numeric(x[length(x)])))})
  } else if (embedding == TRUE & CANN == TRUE & autoencoders == FALSE) {
    results <- apply(parameters, 1, function(x){
      print(paste("Now running testfold:", as.numeric(x[1]), ", and tuning parameter options:", toString(x[2:(length(x)-1)]), ", repeating option =", as.numeric(x[length(x)])))
      cross_run_results <- crossvalidation_CANN_run_Embedding(all_folds_data[[as.numeric(x[1])]], 
                                                              x[2:(length(x)-1)],
                                                              embedding_output_dim = embedding_output_dim,
                                                              cat_vars = cat_vars,
                                                              cann_variable = cann_variable,
                                                              trainable_output = trainable_output)
      return(cross_run_results %>% bind_cols("test_fold" = as.numeric(x[1]), "repeating_runs" = as.numeric(x[length(x)])))})
  } else if (embedding == FALSE & CANN == FALSE & autoencoders == TRUE) {
    results <- apply(parameters, 1, function(x){
      #print(paste("Now running testfold:", as.numeric(x[1]), ", and tuning parameter options:", toString(x[2:(length(x)-1)]), ", repeating option =", as.numeric(x[length(x)])))
      cross_run_results <- crossvalidation_run_AE(all_folds_data[[as.numeric(x[1])]], 
                                                  x[2:(length(x)-1)],
                                                  autoencoder_trained = autoencoder_trained[[as.numeric(x[1])]],
                                                  cat_vars = cat_vars)
      return(cross_run_results %>% bind_cols("test_fold" = as.numeric(x[1]), "repeating_runs" = as.numeric(x[length(x)])))})
  } else if (embedding == FALSE & CANN == TRUE & autoencoders == TRUE) {
    results <- apply(parameters, 1, function(x){
      #print(paste("Now running testfold:", as.numeric(x[1]), ", and tuning parameter options:", toString(x[2:(length(x)-1)]), ", repeating option =", as.numeric(x[length(x)])))
      cross_run_results <- crossvalidation_CANN_run_AE(all_folds_data[[as.numeric(x[1])]], 
                                                       x[2:(length(x)-1)],
                                                       autoencoder_trained = autoencoder_trained[[as.numeric(x[1])]],
                                                       cat_vars = cat_vars,
                                                       cann_variable = cann_variable,
                                                       trainable_output = trainable_output)
      return(cross_run_results %>% bind_cols("test_fold" = as.numeric(x[1]), "repeating_runs" = as.numeric(x[length(x)])))})
  }
  print(paste("Running the parameter options over all test fold took: ", difftime(time1 = Sys.time(), time2 = time, units = "min"), " minutes"))
  rbind_results <- tibble(rbindlist(results))
  # Sound to indicate tuning run has completed
  beep(5)
  return(rbind_results)
}

# Small function to readout optimal tuning parameter from the alltestfolds_tuningrun output
optimal_tuningParameter <- function(tuning_results, tuning_parameter){
  chosen_parameter <- tuning_results %>% 
    group_by(test_fold, repeating_runs) %>% 
    slice(which.min(validation_error)) %>%
    group_by_at(tuning_parameter) %>%
    summarise(N=n(), 
              Mean_error = mean(validation_error), 
              Mean_runTime = mean(cross_run_time),
              .groups = "drop") %>%
    arrange(desc(N),Mean_runTime) %>% slice(1) %>% select(tuning_parameter)
  print(paste(" Chosen parameter",tuning_parameter,"=",chosen_parameter))
  return(chosen_parameter)
}

# Better function for optimal tuning parameter; choses across all parameters, instead of old setup of one-by-one
opt_tuning <- function(tuning_result){
  tuning_result %>% 
    group_by(across(-c(repeating_runs, validation_error, cross_run_time, epochs_used))) %>% 
    summarise(bagging_val_error = mean(validation_error),
              mean_time = mean(cross_run_time)) %>% 
    group_by(test_fold) %>% 
    slice(which.min(bagging_val_error)) %>%
    group_by_at(c("dropout","batch","hiddennodes","activation_h","activation_out","optimizer","epochs","loss")) %>%
    summarise(N=n(), 
              bagging_val_error = mean(bagging_val_error), 
              mean_time = mean(mean_time),
              .groups = "drop") %>%
    arrange(desc(N),mean_time) %>% slice(1)
}


# Small function to turn a tuning result row into a useable flag list
tuning_to_flags <- function(result){
  out <- result %>%  
    select(optimizer, batch, activation_h,hiddennodes,dropout,activation_out, epochs, loss) %>% 
    mutate(dropout = unlist(dropout)[1]) %>% 
    as.data.table
  return(out)
}

# Function to determine test performance, can repeat runs for avoiding local minima
getTestErrors <- function(dataToUse, val_split_factor, flags, repeating_runs, embedding = FALSE, 
                          CANN = FALSE, cat_vars, autoencoder_trained = NULL, embedding_output_dim = 1, 
                          cann_variable = "prediction", trainable_output = FALSE, 
                          GLM_Bias_Regularization = FALSE){
  folds_in_data <- length(dataToUse)
  grid <- expand.grid(1:folds_in_data, 1:repeating_runs)
  if(is.null(autoencoder_trained)){autoencoders = FALSE} else {autoencoders = TRUE}

  print(autoencoders)
  # For all folds in the data the test performance is calculated
  # This is repeated for repeating_run number of times
  grid_results <- sapply(c(1:nrow(grid)), function(x){
    print(paste0("Now running testfold ",grid[x,1],", and repeating number ",grid[x,2]))
    rep_run <- if(embedding == FALSE & CANN == FALSE & autoencoders == FALSE){
      single_run(dataToUse[[grid[x,1]]], flags_list = flags, 
                 random_val_split = val_split_factor, GLM_Bias_Regularization)
    } else if (embedding == TRUE & CANN == FALSE & autoencoders == FALSE) {
      single_run_Embedding(dataToUse[[grid[x,1]]], flags_list = flags, 
                           random_val_split = val_split_factor, cat_vars = cat_vars, 
                           GLM_Bias_Regularization)
    } else if (embedding == TRUE & CANN == TRUE & autoencoders == FALSE) {
      single_CANN_run_Embedding(fold_data = dataToUse[[grid[x,1]]], 
                                flags_list = flags, 
                                random_val_split = val_split_factor, 
                                cat_vars = cat_vars, 
                                cann_variable = cann_variable, 
                                trainable_output = trainable_output, 
                                embedding_output_dim = embedding_output_dim, 
                                GLM_Bias_Regularization = GLM_Bias_Regularization)
    } else if (embedding == FALSE & CANN == TRUE & autoencoders == FALSE) {
      stop(paste("Not implemented"))
    } else if (embedding == FALSE & CANN == FALSE & autoencoders == TRUE) {
      print("we are here")
      single_run_AE(fold_data = dataToUse[[grid[x,1]]], 
                    flags_list = flags, 
                    random_val_split = val_split_factor, 
                    cat_vars = cat_vars, 
                    autoencoder_trained = autoencoder_trained[[grid[x,1]]]$Encoder,
                    GLM_Bias_Regularization = GLM_Bias_Regularization)
    } else if (embedding == FALSE & CANN == TRUE & autoencoders == TRUE) {
      single_CANN_run_AE(fold_data = dataToUse[[grid[x,1]]], 
                         flags_list = flags, 
                         random_val_split = val_split_factor, 
                         cat_vars = cat_vars, 
                         cann_variable = cann_variable, 
                         trainable_output = trainable_output, 
                         autoencoder_trained = autoencoder_trained[[grid[x,1]]]$Encoder,
                         GLM_Bias_Regularization = GLM_Bias_Regularization)
    }
    
    # We bind the test fold number and repeating run number to the results
    return(rep_run %>% bind_cols(Test_Fold = grid[x,1], Repeating_Run = grid[x,2]))
  })
  #  The results are transposed, and converted to a tibble
  return(grid_results %>% t() %>% as_tibble() %>% mutate_all(unlist))
}

# New predict function, with the option to use Bias Regularization directly
predict_BRoption <- function(model, data, BR, cann = FALSE){
  if(BR == FALSE){
    prediction <- model %>% predict(data, type = "response")
  } else if(BR == TRUE & cann == FALSE) {
    intermediate_output <- model[[1]] %>% predict(data)
    prediction <- predict(object = model[[2]], 
                          newdata = list("intermediate_output"= intermediate_output), 
                          type = "response")
  } else if(BR == TRUE & cann == TRUE) {
    intermediate_output <- model[[1]] %>% predict(data[1:(length(data)-1)])
    prediction <- predict(object = model[[2]], 
                          newdata = list("intermediate_output"= intermediate_output, 
                                         train_cann_input = (data[[length(data)]] %>% data.matrix())), 
                          type = "response")
  }
  return(prediction)
}

# Custom function to scale certain columns, with certain u and sd
scale_withPar <- function(data, scale_vars, scale_pars){
  bind_cols(
    data %>% 
      select(scale_vars) %>% 
      scale(., 
            center = (scale_pars  %>% pull(u)), 
            scale = (scale_pars %>% pull(sd))) %>%
      as_tibble,
    data %>% 
      select(!scale_vars)
  ) %>% select(colnames(data))
}

# Calculate PDP values
pdp_values <- function(variable, var_range, pdp_data, scale_param, scale_vars, run_models, select, output_var , 
                       BR = FALSE, fold_var="fold_nr", id_var = "id"){
  
  # For CANN gbm and glm models, the ClaimAmount models are model[[x+6]]
  if(output_var == "nclaims"){ offset <- 0} else {offset <- 6}
  
  # Calculate PDP values for each testfold in the data
  all_Models <- lapply(1:length(run_models), function(x){
    
    # If a CANN Glm model is used, read in levels of variable here (faster then doing it for each new_value)
    if(select == "cann_glm" & variable == "latlong") {
      binned_levels <- glm_data[[x]] %>% select(postcode,latlong) %>% unique
    } else if (select == "cann_glm" & variable %in% scale_vars) {
      # For continuous variables, determine the bins used in the glm modelling
      binned_levels <- glm_data[[x+offset]] %>% 
        right_join(data %>% select(id, variable), by = "id") %>% as_tibble() %>%
        select(paste0(variable,".y"), paste0(variable,".x")) %>%
        rename(real_var = paste0(variable,".y"), binned_var = paste0(variable,".x")) %>%
        arrange(real_var) %>% 
        distinct(real_var, .keep_all = TRUE) %>% 
        group_by(binned_var) %>%
        summarise(bin_lower_bound = min(real_var)) %>%
        select(bin_lower_bound, binned_var) %>% 
        arrange(bin_lower_bound)
    } else if(select == "cann_glm" & !(variable %in% scale_vars)){
      # for categorical variables, read the levels from glm data
      binned_levels <- glm_data[[x + offset]] %>% pull(!!variable) %>% levels() 
    }
    
    # Get relevant scale parameters for this variable and this testfold
    scale_info <- scale_param %>% filter(Testfold == x)
    
    var_range_scaled <- if(variable == "latlong"){
      var_range %>% 
        mutate(lat_scaled = scale(lat, 
                                  center = (scale_info %>% filter(Variable == "lat"))$u,
                                  scale = (scale_info %>% filter(Variable == "lat"))$sd)) %>%
        mutate(long_scaled = scale(long, 
                                   center = (scale_info %>% filter(Variable == "long"))$u,
                                   scale = (scale_info %>% filter(Variable == "long"))$sd)) 
    } else if(variable %in% scale_vars) {
      var_range %>% mutate(scaled = scale(!!as.name(variable), 
                                          center = (scale_info %>% filter(Variable == variable))$u,
                                          scale = (scale_info %>% filter(Variable == variable))$sd))
    } else {
      # Here add the categorical variables (need to go from string to numerical levels)
      stop()
    }
    
    # Calculate the mean prediction, with the changed variable values
    pdp_allTestFolds <- sapply(1:nrow(var_range),  function(nv){
      
      # Make new data table, with the new value for the selected variable
      new_data <- if(variable != "latlong"){
        pdp_data %>% scale_withPar(scale_vars = scale_vars, scale_pars = scale_info ) %>% 
          mutate(!!variable := var_range_scaled %>% slice(nv) %>% pull(scaled))
      } else {
        pdp_data %>% scale_withPar(scale_vars = scale_vars, scale_pars = scale_info ) %>% 
          mutate(lat = var_range_scaled %>% slice(nv) %>% pull(lat_scaled),
                 long = var_range_scaled %>% slice(nv) %>% pull(long_scaled))
      }
      
      if(select == "onehot"){
        # One Hot Encoding
        
        scaled_data <- new_data %>% 
          as.data.table() %>% 
          one_hot() %>%
          as_tibble()
        new_data_matrix <- lapply(c(list(run_models[[x]]$other_vars)),
                                  function(data, x){
                                    data[,x]
                                  }, data = scaled_data %>% data.matrix())
        
      } else if (select == "embedding") {
        # NN with Embedding layers
        
        new_data_matrix <- lapply(c(list(run_models[[x]]$other_vars), unlist(run_models[[x]]$cat_vars)),  
                                  function(data, x){ 
                                    data[,x]
                                  }, data = new_data %>% data.matrix())
        
      } else if (select == "cann_gbm") {
        # CANN model with GBM input. Make new gbm predictions based on the mutated data
        
        if(variable != "latlong"){
          new_gbm_data <- pdp_data %>% 
            filter(id %in% (new_data %>% pull(id))) %>% 
            mutate(!!variable := var_range_scaled %>% slice(nv) %>% pull(variable)) 
        } else {
          new_gbm_data <- pdp_data %>% 
            filter(id %in% (new_data %>% pull(id))) %>% 
            mutate(lat = var_range_scaled %>% slice(nv) %>% pull(lat), 
                   long = var_range_scaled %>% slice(nv) %>% pull(long)) 
        }
        
        new_gbm_data <- new_gbm_data %>%  
          mutate(prediction = exp(predict.gbm(gbm_fits[[x+offset]], 
                                              n.trees = gbm_fits[[x+offset]]$n.trees,
                                              new_gbm_data))) %>%
          select(id, prediction)
        
        new_data <- left_join(new_data, new_gbm_data, by = "id") %>% mutate(prediction = log(prediction))
        new_data_matrix <- lapply(c(list(run_models[[x]]$other_vars), unlist(run_models[[x]]$cat_vars), list(run_models[[x]]$cann_variable)),  
                                  function(data, x){ 
                                    data[,x]
                                  }, data = new_data %>% data.matrix())
        
      } else if (select == "cann_glm") {
        # CANN model with GLM inputs
        
        if(variable == "latlong"){
          new_glm_value <- binned_levels %>% filter(postcode <= (var_range %>% slice(nv) %>% pull(POSTCODE))) %>% slice(n()) %>% pull(latlong)
        } else if(variable %in% scale_vars){
          # Get bin which contains new variable
          new_glm_value <- binned_levels %>% filter(bin_lower_bound <= (var_range %>% slice(nv) %>% pull(variable))) %>% slice(n()) %>% pull(binned_var)
        } else {
          # Categorical variable, Character in GLM
          new_glm_value <- binned_levels[var_range %>% slice(nv) %>% pull(variable)] # new_value is the index of the levels of the factor in the glm data
        }
        
        # Adjust glm_data with new value
        glm_new_data <- glm_data[[x + offset]] %>% 
          filter(id %in% (new_data %>% pull(id))) %>% 
          mutate(!!variable := new_glm_value) %>% mutate_at(variable, as.factor) %>% as_tibble()
        # Make new glm prediction based on mutated glm data
        glm_new_prediction <- bind_cols(glm_new_data, prediction = predict(glm_fits[[x + offset]], glm_new_data, "response")) %>% 
          mutate(prediction = log(prediction)) %>%
          select(c("id","prediction")) 
        # Add new glm prediction to new_data set
        new_data <- left_join(new_data, glm_new_prediction, by = "id")
        
        new_data_matrix <- lapply(c(list(run_models[[x]]$other_vars), unlist(run_models[[x]]$cat_vars), list(run_models[[x]]$cann_variable)),  
                                  function(data, x){ 
                                    data[,x]
                                  }, data = new_data %>% data.matrix())
        
      } else if (select == "cann_expo") {
        new_data <- new_data %>% mutate(expo = log(expo))
        new_data_matrix <- lapply(c(list(run_models[[x]]$other_vars), unlist(run_models[[x]]$cat_vars), list(run_models[[x]]$cann_variable)),  
                                  function(data, x){ 
                                    data[,x]
                                  }, data = new_data %>% data.matrix())
      }
      
      # Predict on the new data matrix
      pdp_output <- predict_BRoption(run_models[[x]]$model, new_data_matrix, BR, grepl("cann", select)) %>% mean()
      
      return(pdp_output)
    })  %>% as_tibble() %>% bind_cols(var_range, testfold = x)
    
    return(pdp_allTestFolds)
  })
  return(all_Models %>% bind_rows())
}

# Make a plot of PDP values (from the pdp_values function)
plot_pdp <- function(pdp_values, var_name, xlabel = NULL, ylabel = NULL, coloruse = NULL, 
                     belgium_shape = NULL, map_testfold = 1, model_name = "", legend_name = "Prediction Effect"){
  if(var_name != "latlong"){
    # Pdp plot of a continuous or categorical variable
    plot <- pdp_values %>% 
      mutate(testfold = factor(testfold)) %>% 
      ggplot(aes(x = !!as.name(var_name), y = value, group = testfold)) +
      geom_line(aes(color = testfold)) + 
      scale_color_manual(name = "Model Trained without Test Fold", values = coloruse) + 
      theme_bw() + 
      labs(x = xlabel, y = ylabel)
  } else {
    # Group PDP values per postalcode, and select a test fold
    pdp_values_forMap <- pdp_values %>% 
      filter(testfold == map_testfold) %>% 
      group_by(POSTCODE) %>% 
      summarise(pdp_value = mean(value))
    
    # Join pdp info with the Belgian Shape file
    belgium_shape_postcode <- left_join(belgium_shape, pdp_values_forMap, by = "POSTCODE")
    
    # Plot the PDP effect of latlong over a map of Belgium
    plot <- tm_shape(belgium_shape_postcode) + 
      tm_borders(col = "black") + 
      tm_fill(col = "pdp_value", title = legend_name,textNA = "No Policyholders", style = "cont", palette = "Blues", colorNA = "white") +
      tm_layout(frame = FALSE, legend.title.size=1.5, legend.text.size = 1, 
                main.title = "PDP: Effect of Postalcode on Predicted Number of Claims", main.title.position = "left", 
                title = model_name, title.size = 1)
  }
  return(plot)
}

# Calculate VIP values
vip_function <- function(vipdata, scale_vars, scale_param, run_models, vars_with_label, select, output_var, 
                         BR = FALSE, fold_var="fold_nr", id_var = "id"){
  
  # For CANN gbm and glm models, the ClaimAmount models are model[[x+6]]
  if(output_var == "nclaims"){ offset <- 0} else {offset <- 6}
  
  vip_values <- lapply(1:6, function(fold){
    
    run <- run_models[[fold]]
    scale_info <- scale_param %>% filter(Testfold == x)
    
    # Construct regular scaled data, with cann inputs when needed, and make baseline predictions
    if(select == "onehot"){
      scaled_data <- vipdata %>% 
        as.data.table() %>% 
        one_hot() %>%
        as_tibble() %>% 
        scale_withPar(scale_vars = scale_vars, scale_pars = scale_info )
      reg_data_matrix <- lapply(c(list(run$other_vars)),
                                function(data, x){
                                  data[,x]
                                }, data = scaled_data %>% data.matrix())
      
    } else if(select == "embedding"){
      scaled_data <- vipdata %>% scale_withPar(scale_vars = scale_vars, scale_pars = scale_info )
      reg_data_matrix <- lapply(c(list(run$other_vars), unlist(run$cat_vars)),
                                function(data, x){
                                  data[,x]
                                }, data = scaled_data %>% data.matrix())
      
    } else if(select == "cann_gbm"){
      scaled_data <- vipdata  %>% 
        mutate(prediction = exp(predict.gbm(gbm_fits[[fold+offset]], 
                                            n.trees = gbm_fits[[fold+offset]]$n.trees,
                                            vipdata))) %>% 
        mutate(prediction = log(prediction)) %>%
        scale_withPar(scale_vars = scale_vars, scale_pars = scale_info )
      reg_data_matrix <- lapply(c(list(run$other_vars), unlist(run$cat_vars), list(run$cann_variable)),
                                function(data, x){
                                  data[,x]
                                }, data = scaled_data %>% data.matrix())
      
    } else if(select == "cann_glm"){
      # Adjust glm_data with new value
      glm_new_data <- glm_data[[fold + offset]] %>% 
        filter(id %in% (vipdata %>% pull(id)))
      glm_new_prediction <- bind_cols(glm_new_data, prediction = predict(glm_fits[[fold + offset]], glm_new_data, "response")) %>% 
        mutate(prediction = log(prediction)) %>%
        select(c("id","prediction")) 
      scaled_data <- left_join(vipdata %>% scale_withPar(scale_vars = scale_vars, scale_pars = scale_info ), glm_new_prediction, by = "id")
      reg_data_matrix <- lapply(c(list(run$other_vars), unlist(run$cat_vars), list(run$cann_variable)),
                                function(data, x){
                                  data[,x]
                                }, data = scaled_data %>% data.matrix())
      
    } else if((select == "cann_expo")){
      scaled_data <- vipdata %>% scale_withPar(scale_vars = scale_vars, scale_pars = scale_info ) %>% mutate(expo = log(expo))
      reg_data_matrix <- lapply(c(list(run$other_vars), unlist(run$cat_vars), list(run$cann_variable)),
                                function(data, x){
                                  data[,x]
                                }, data = scaled_data %>% data.matrix())
      
    }
    # Make the prediction for the unpermutated data set:
    reg_prediction <- predict_BRoption(run$model, reg_data_matrix, BR, grepl("cann", select))
    
    # Mutate each variable, and get the change in prediction
    all_vars <- sapply((vars_with_label %>% filter(Variable != "long" & Variable != "lat") %>% bind_rows(tibble(Variable = "latlong", xlabels = "Long-Lat Combination")) %>% pull(Variable)), function(var){
      
      # permutate indices, use it to permutate variable
      indices <- c(1:nrow(vipdata)) %>% sample()
      if(var =="latlong"){
        mut_data<- vipdata %>% mutate(long = (vipdata %>% slice(indices) %>% pull(long)), lat = (vipdata %>% slice(indices) %>% pull(lat)))
      } else {
        mut_data <- vipdata %>% mutate(!!var := (vipdata %>% slice(indices) %>% pull(var)))
      }
      perm_id <- bind_cols(origin = vipdata %>% pull("id"), permutate = vipdata %>% slice(indices) %>% pull("id"))
      
      # Prepare permutated data, based on which model is selected
      if(select == "onehot"){
        mut_data <- mut_data %>% 
          as.data.table() %>% 
          one_hot() %>%
          as_tibble()
        mut_data_matrix <- lapply(c(list(run$other_vars)),
                                  function(data, x){
                                    data[,x]
                                  }, data = mut_data %>% scale_withPar(scale_vars = scale_vars, scale_pars = scale_info ) %>% data.matrix())
        
      } else if(select == "embedding"){
        mut_data_matrix <- lapply(c(list(run$other_vars), unlist(run$cat_vars)),
                                  function(data, x){
                                    data[,x]
                                  }, data = mut_data %>% scale_withPar(scale_vars = scale_vars, scale_pars = scale_info ) %>% data.matrix())
        
      } else if(select == "cann_gbm"){
        # Add new gbm prediction, based on permutated data
        mut_data_cann <- mut_data  %>% 
          mutate(prediction = exp(predict.gbm(gbm_fits[[fold+offset]], 
                                              n.trees = gbm_fits[[fold+offset]]$n.trees,
                                              mut_data))) %>% 
          mutate(prediction = log(prediction))
        # With permutated data and new gbm prediction, predict
        mut_data_matrix <- lapply(c(list(run$other_vars), unlist(run$cat_vars), list(run$cann_variable)),
                                  function(data, x){
                                    data[,x]
                                  }, data = mut_data_cann %>% scale_withPar(scale_vars = scale_vars, scale_pars = scale_info ) %>% data.matrix())
        
      } else if(select == "cann_glm"){
        # Apply the same permutation in the glm_data
        if(var =="latlong" & offset == 6){
          # the CA glm data has long and lat seperatly
          glm_perm_data <- glm_data[[fold + offset]] %>% 
            slice(perm_id$origin) %>%
            mutate(long = (glm_data[[fold + offset]] %>% slice(perm_id$permutate) %>% pull(long)), 
                   lat = (glm_data[[fold + offset]] %>% slice(perm_id$permutate) %>% pull(lat)))
        } else {
          glm_perm_data <- glm_data[[fold + offset]] %>% 
            slice(perm_id$origin) %>%
            mutate(!!var := glm_data[[fold + offset]] %>% slice(perm_id$permutate) %>% pull(var))
        }
        # Predict with the new permutated glm data
        glm_perm_prediction <- bind_cols(glm_perm_data, prediction = predict(glm_fits[[fold + offset]], glm_perm_data, "response")) %>% 
          mutate(prediction = log(prediction)) %>%
          select(c("id","prediction")) 
        # Join permutated data and permutated glm prediction back together, and predict
        mut_data_cann <- left_join(mut_data, glm_perm_prediction, by = "id")
        mut_data_matrix <- lapply(c(list(run$other_vars), unlist(run$cat_vars), list(run$cann_variable)),
                                  function(data, x){
                                    data[,x]
                                  }, data = mut_data_cann %>% scale_withPar(scale_vars = scale_vars, scale_pars = scale_info ) %>% data.matrix())
        
      } else if((select == "cann_expo")){
        mut_data_matrix <- lapply(c(list(run$other_vars), unlist(run$cat_vars), list(run$cann_variable)),
                                  function(data, x){
                                    data[,x]
                                  }, data = mut_data %>% scale_withPar(scale_vars = scale_vars, scale_pars = scale_info ) %>% mutate(expo = log(expo)) %>% data.matrix())
        
      }
      
      # make predictions on the permutated dataset
      mut_prediction <- predict_BRoption(run$model, mut_data_matrix, BR, grepl("cann", select))
      return(sum(abs(reg_prediction - mut_prediction)))
      
    }) %>% 
      as_tibble(rownames = "Variable") %>% 
      bind_cols(Testfold = fold) %>% 
      mutate(scaled_vip = value / sum(value))
    
  }) %>% bind_rows()
  return(vip_values)
}

# Make a plot of VIP values (from the vip_values function)
plot_vip_ordered <- function(vip_values, vars_with_label, modellabel, xlabel = "Importance", ylabel = "Variable", coloruse){
  left_join(vip_values, vars_with_label, by = "Variable") %>% 
    ggplot(aes(y = reorder(xlabels, scaled_vip, sum))) +   
    geom_col(aes(x = scaled_vip, fill = as.factor(Testfold)), position="dodge") + 
    theme_bw() + 
    labs(title = "Variable Importance Plot", subtitle = modellabel) +
    xlab(xlabel) + ylab(ylabel) + 
    scale_fill_manual(name = "Model Trained without Test Fold", values = testfold_colors) +
    theme(legend.position="bottom", legend.direction="horizontal") + 
    guides(fill = guide_legend(ncol=6))  
}

# Make a plot of VIP values (from the vip_values function)
plot_vip <- function(vip_values, vars_with_label, modellabel, xlabel = "Importance", ylabel = "Variable", coloruse){
  left_join(vip_values, vars_with_label, by = "Variable") %>% 
    ggplot(aes(y = xlabels)) +   
    geom_col(aes(x = scaled_vip, fill = as.factor(Testfold)), position="dodge") + 
    theme_bw() + 
    labs(title = "Variable Importance Plot", subtitle = modellabel) +
    xlab(xlabel) + ylab(ylabel) + 
    scale_fill_manual(name = "Model Trained without Test Fold", values = testfold_colors) +
    theme(legend.position="bottom", legend.direction="horizontal") + 
    guides(fill = guide_legend(ncol=6))  
}

# Calculate relativity ratio's and make a lift plot
loss_ratio_lift <- function(data, bench_name, comp_name, bins){
  
  # Gather the variables of interest
  comp <- data.table(base = data %>% 
                       mutate(pi_bench = !!as.name(paste0(bench_name,"_NC")) * !!as.name(paste0(bench_name,"_CA"))) %>% 
                       pull(pi_bench),
                     pred = data %>% 
                       mutate(pi_comp = !!as.name(paste0(comp_name,"_NC")) * !!as.name(paste0(comp_name,"_CA"))) %>% 
                       pull(pi_comp),
                     amount = data %>% replace_na(list(average = 0)) %>% mutate(amount = average * nclaims) %>% pull(amount),
                     expo = data %>% pull(expo))
  # Calculate the relativity and order the data on this value
  comp[, rel := pred/base]
  comp <- setorder(comp,rel)
  
  # Calculate the loss ratios to determine the model lift
  lift <- comp[, list(LR = sum(amount)/sum(base),
                      expo = sum(expo),
                      min.rel = min(rel),
                      max.rel = max(rel)),
               keyby = cut(cumsum(expo),sum(expo)*(0:bins)/bins)]
  
  lift[, label := paste0('[',round(min.rel,2),',',round(max.rel,2),']')]
  
  lift[, min.rel := round(min.rel,2)]
  lift[, max.rel := round(max.rel,2)]
  lift$min.rel[1]<- 0
  lift$max.rel[bins] <- "+Inf"
  
  lift[, label2 := paste0('[',min.rel,',',max.rel,']')]
  
  
  # Create the list plot
  lift_plot <- ggplot(lift, aes(label2,LR)) + 
    geom_bar(stat = 'identity', fill='#99CCFF', color='#003366') +
    geom_text(aes(label = round(LR, digits = 3)), vjust = -0.15) +
    theme_bw() + 
    labs(x = 'Relativity bin', y = 'Loss ratio') + 
    ylim(0,1.6)
  
  # Return list of lift data and plot
  list(comp_data = comp, lifts = lift, lifts_plot = lift_plot)
}

# Calculate relativity ratio's and make a double lift plot
double_lift <- function(data, bench_name, comp_name, bins, plotit = TRUE, perc.err = FALSE){
  
  # Gather the variables of interest
  comp <- data.table(base = data %>% 
                       mutate(pi_bench = !!as.name(paste0(bench_name,"_NC")) * !!as.name(paste0(bench_name,"_CA"))) %>% 
                       pull(pi_bench),
                     pred = data %>% 
                       mutate(pi_comp = !!as.name(paste0(comp_name,"_NC")) * !!as.name(paste0(comp_name,"_CA"))) %>% 
                       pull(pi_comp),
                     amount = data %>% replace_na(list(average = 0)) %>% mutate(amount = average * nclaims) %>% pull(amount),
                     expo = data %>% pull(expo))
  # Calculate the relativity and order the data on this value
  comp[, rel := pred/base]
  comp <- setorder(comp,rel)
  
  # Calculate the means to determine the model lift
  if(!perc.err){
    lift <- comp[, list(true = mean(amount),
                        comp = mean(pred),
                        bench = mean(base),
                        expo = sum(expo),
                        min.rel = min(rel),
                        max.rel = max(rel)),
                 keyby = cut(cumsum(expo),sum(expo)*(0:bins)/bins)]
    lift[, c('true','comp','bench')] <- lift[, lapply(.SD, function(x) x/mean(x)), .SDcols = c('true','comp','bench')]
  } else{
    lift <- comp[, list(comp = mean(pred)/mean(amount) - 1,
                        bench = mean(base)/mean(amount) - 1,
                        expo = sum(expo),
                        min.rel = min(rel),
                        max.rel = max(rel)),
                 keyby = cut(cumsum(expo),sum(expo)*(0:bins)/bins)]
  }
  
  lift[, label := paste0('[',round(min.rel,2),',',round(max.rel,2),']')]
  
  lift[, min.rel := round(min.rel,2)]
  lift[, max.rel := round(max.rel,2)]
  lift$min.rel[1]<- 0
  lift$max.rel[bins] <- "+Inf"
  
  lift[, label2 := paste0('[',min.rel,',',max.rel,']')]
  
  if(!plotit) return(lift)
  
  lift <- melt(lift, id.vars = c('label2'), measure.vars = intersect(c('true','comp','bench'),names(lift)), variable.name = 'tariff')
  return(ggplot(lift, aes(label2, value, group = tariff, colour = tariff)) + geom_line(aes(linetype = tariff)) + geom_point() + theme_bw() + labs(x = 'Relativity bin', y = ifelse(perc.err,'Percentage error','Double lift')))
}

# Calculate and plot the Lorenz Curve of selected models versus benchmark
ordered_lorenz_curve <- function(preds, mclass, base, profit_ind){
  
  # Compute Gini indices
  gini_index <- gini(loss = 'amount',
                     score  = mclass,
                     base = base,
                     data = preds)
  
  # Get the points of the Lorenz curves
  lorenz <- as.data.table(slot(gini_index,'lorenz')[[1]])
  names(lorenz) <- gsub('_prem:poissgamma','',names(lorenz))
  
  # Melt the data in a long format
  lorenz_melt <- melt(lorenz, id.vars = '.P.', variable.name = 'method', value.name = 'prem')
  
  # Plot the ordered Lorenz curves and the line of equality
  ggplt <- ggplot() + theme_bw() + geom_line(data = lorenz, aes(x = .P., y = .P.), lwd = 0.5) + labs(x = 'Premiums', y = 'Losses') +
    geom_line(data = lorenz_melt, aes(x = .P., y = prem, group = method, colour = method), lwd = 0.75)
  
  # Add the optimal profit point if needed
  if(profit_ind) {
    opt_x <- lorenz_melt[which.min(prem - .P.), .P.]
    opt_y <- lorenz_melt[which.min(prem - .P.), prem]
    ggplt <- ggplt + geom_segment(aes(x = opt_x, y = opt_y, xend = opt_x, yend = 0), linetype = 3) +
      geom_segment(aes(x = opt_x, y = opt_y, xend = 0, yend = opt_y), linetype = 3) +
      annotate('text', x = opt_x + 12.5, y = opt_y, label = paste0('(',round(opt_x),',',round(opt_y),')'), size = 6)
  }
  
  # Return the plot
  return(ggplt)
}

# -----
# -----
# ----- THE END ----
# -----
# -----





