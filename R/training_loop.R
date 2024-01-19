training_loop <- function(model,
                          batch_gen_train,
                          batch_gen_valid,
                          noise_channels,
                          num_samples,
                          steps_per_checkpoint,
                          batch_size,
                          log_folder = paste0(getwd(), "/log_folder"),
                          continue = FALSE,
                          mode = "GAN"
                          ){
  num_checkpoints <- round(num_samples/(steps_per_checkpoint * batch_size))
  checkpoint      <- 1

  # create log folder and model save/load subfolder if they don't exist
  if(!file.exists(log_folder)){
    dir.create(log_folder)
  }

  model_weights_root <- paste0(log_folder, "models/")
  if(!file.exists(model_weights_root)){
    dir.create(model_weights_root)
  }

  # plots_root <- paste0(log_folder, "plots/")
  # if(!file.exists(plots_root)){
  #   dir.create(plots_root)
  # }

  if(continue){
    ## IF continue training, load most recent model weights
     load(model, filenames_from_root(model, model_weights_root))
  }

  # initialize run status
  training_samples <- 0
  log_file = paste0(log_folder, "log.txt")

  if(continue){
    # set current iteration and increase total number
     training_samples <- as.numeric(substr(dir(model_weights_root)[length(dir(model_weights_root))], 13, 19))
     num_samples <- training_samples + num_samples
  }

  while (training_samples < num_samples){ # main training loop

    cat("\n")
    print(paste0("Checkpoint ", checkpoint, "/", num_checkpoints))

    # train for some number of batches
    loss_log = train_model(model=model,
                           mode=mode,
                           batch_gen_train=batch_gen_train,
                           batch_gen_valid=batch_gen_valid,
                           noise_channels=noise_channels,
                           checkpoint=checkpoint,
                           steps_per_checkpoint=steps_per_checkpoint)

    training_samples <- training_samples + steps_per_checkpoint * batch_size

    if(checkpoint == 1){
      # set up log DataFrame based on loss_log entries
      col_names  <- c("training_samples", names(loss_log))
      log        <- data.frame(matrix(ncol = length(col_names), nrow = 0))
      names(log) <- col_names
      log        <- tibble(log)
    }

    checkpoint <- checkpoint + 1

    # save results
    save(model, model_weights_root)

    run_status = data.frame(training_samples = training_samples)
    exportJSON <- toJSON(run_status)
    f <- paste0(log_folder, "run_status.json")
    write(exportJSON, f)

    #  fromJSON(f) # to read it again

    data <- data.frame(training_samples = training_samples)
    for(foo in names(loss_log)){
      data[[foo]] <- loss_log[[foo]]
    }

    if(!file.exists(log_file)){
      write.table(data, file = log_file, row.names = FALSE)
    }else{
      write.table(data, file = log_file, append = TRUE, col.names = FALSE, row.names = FALSE)
    }

    # Save model weights each checkpoint
    gen_weights_file <- paste0(model_weights_root, "gen_weights-", formatC(training_samples, digits = 6, flag = 0), ".h5")
    model$gen$save_weights(gen_weights_file)
    disc_weights_file <- paste0(model_weights_root, "disc_weights-", formatC(training_samples, digits = 6, flag = 0), ".h5")
    model$disc$save_weights(disc_weights_file)
  }
}
