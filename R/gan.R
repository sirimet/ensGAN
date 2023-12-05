#' Initialize gan
#'
#' @param gen
#' @param disc
#' @param mode
#' @param gradient_penalty_weight
#' @param lr_disc
#' @param lr_gen
#' @param avg_seed
#' @param ensemble_size
#' @param ensemble_members
#' @param content_loss_weight
#'
#' @return
#' @export
#'
#' @examples
initialize_ensgan <- function(gen, disc, mode = mode,  arch = arch, gradient_penalty_weight = 10,
                              lr_disc = 0.0001, lr_gen = 0.0001, avg_seed = NULL,
                              ensemble_size = NULL, ensemble_members = NULL, content_loss_weight = NULL){
  object <- NULL
  object$gen = gen
  object$disc = disc
  object$mode = mode
  object$arch = arch
  object$gradient_penalty_weight = gradient_penalty_weight
  object$lr_disc = lr_disc
  object$lr_gen = lr_gen
  object$ensemble_size = ensemble_size
  object$content_loss_weight = content_loss_weight
  object <- object %>% build_ens_gan()

  return(object)
}

#' Title
#'
#' @param self
#' @param root
#'
#' @return
#' @export
#'
#' @examples
filenames_from_root <- function(self, root){
  fn <- list(
    gen_weights = paste0(root, "-gen_weights.h5"),
    disc_weights = paste0(root, "-disc_weights.h5"),
    gen_opt_weights = paste0(root, "-gen_opt_weights.h5"),
    disc_opt_weights = paste0(root, "-disc_opt_weights.h5")
  )
  return(fn)
}

#' Title
#'
#' @param self
#' @param load_files
#'
#' @return
#' @export
#'
#' @examples
load <- function(self, load_files){
  self$gen$load_weights(load_files[["gen_weights"]])
  self$disc$load_weights(load_files[["disc_weights"]])

  if(!self$disc$trainable){
    self$gen_trainer <- self$gen$make_train_function()
    load_opt_weights(self$gen_trainer,
                     load_files[["gen_opt_weights"]])
  }
  if(!self$gen$trainable){
    self$disc_trainer <- self$disc$make_train_function()
    load_opt_weights(self$disc_trainer,
                     load_files[["disc_opt_weights"]])
  }
}

#' Title
#'
#' @param self
#' @param save_fn_root
#'
#' @return
#' @export
#'
#' @examples
save <- function(self, save_fn_root){
  paths <- filenames_from_root(self, save_fn_root)
  self$gen$save_weights(paths[["gen_weights"]], overwrite = TRUE)
  self$disc$save_weights(paths[["disc_weights"]], overwrite = TRUE)
  save_opt_weights(self$disc_trainer, paths[["disc_opt_weights"]])
  save_opt_weights(self$gen_trainer, paths[["gen_opt_weights"]])
}


#' Title
#'
#' @param self
#'
#' @return
#' @export
#'
#' @examples
build_ens_gan <- function(self){
  # find shapes for inputs
  cond_shapes  = input_shapes(self$gen, "lo_res_inputs")
  const_shapes = input_shapes(self$gen, "hi_res_inputs")
  noise_shapes = input_shapes(self$gen, "noise_input")
  sample_shapes = input_shapes(self$disc, "output")

  # Create generator training network
  self$disc$trainable <- FALSE
  cond_in  = c(layer_input(shape =  cond_shapes[[1]]))
  const_in = c(layer_input(shape = const_shapes[[1]]))

  if(is.null(self$ensemble_size)){
    noise_in = c(layer_input(shape=noise_shapes[[1]]))
  }else{
    noise_in = list()
    for(ii in 1:self$ensemble_size){
      noise_in[[ii]] <- layer_input(shape=noise_shapes[[1]]) # c() ?
    }
  }
  gen_in = c(cond_in, const_in, noise_in)

  gen_out = self$gen(gen_in[1:3])  # only use cond/const/noise
  gen_out = ensure_list(gen_out)
  disc_in_gen = c(cond_in, const_in, gen_out)
  disc_out_gen = self$disc(disc_in_gen)
  full_gen_out = list(disc_out_gen)
  if(!is.null(self$ensemble_size)){
    # generate ensemble of predictions and add mean to gen_trainer output
    preds <- list()
    for(ii in 1:(self$ensemble_size)){
      preds[[ii]] <- self$gen(list(gen_in[[1]], gen_in[[2]], gen_in[[2+ii]]))
    }
    preds <- tf$stack(preds)
    pred_mean <- tf$reduce_mean(preds, axis=0L)
    full_gen_out <- c(full_gen_out, preds_mean)

  }
  self$gen_trainer <- keras_model(inputs=gen_in,
                                  outputs=full_gen_out)

  self$disc$trainable <- TRUE
  # Create discriminator training network
  self$gen$trainable <- FALSE
  cond_in   <- lapply(cond_shapes,   function(s) layer_input(shape = s, name = "lo_res_inputs"))
  const_in  <- lapply(const_shapes,  function(s) layer_input(shape = s, name = "hi_res_inputs"))
  noise_in  <- lapply(noise_shapes,  function(s) layer_input(shape = s, name = "noise_input"))
  sample_in <- lapply(sample_shapes, function(s) layer_input(shape = s, name = "output"))

  gen_in = c(cond_in, const_in, noise_in)

  disc_in_real = sample_in[[1]]
  disc_in_fake = self$gen(gen_in)

  disc_in_avg   <- random_weighted_average(disc_in_real, disc_in_fake)
  disc_out_real <- self$disc(c(cond_in, const_in, disc_in_real))
  disc_out_fake <- self$disc(c(cond_in, const_in, disc_in_fake))
  disc_out_avg  <- self$disc(c(cond_in, const_in, disc_in_avg))

  self$disc_trainer <- keras_model(inputs = c(cond_in, const_in, noise_in, sample_in),
                                   outputs=list(disc_out_real, disc_out_fake),
                                   name='disc_trainer')
  self$gen$trainable <- TRUE
  self <- self %>% compile()
  return(self)
}

#' Title
#'
#' @param self
#' @param opt_disc
#' @param opt_gen
#'
#' @return
#' @export
#'
#' @examples
compile <- function(self, opt_disc = NULL, opt_gen = NULL){
  #create optimizers
  if(is.null(opt_disc)){
    opt_disc = optimizer_adam(self$lr_disc, beta_1 = 0.5, beta_2 = 0.9, clipnorm = .9) ## TRY SOME GRADIENT CLIPPING, SEE IF IT HELPS--SHOULD IMPLEMENT GRADIENT PENALTY
  }
  self$opt_disc = opt_disc
  if(is.null(opt_gen)){
    opt_gen = optimizer_adam(self$lr_gen, beta_1 = 0.5, beta_2 = 0.9, clipnorm = .9) ## TRY SOME GRADIENT CLIPPING, SEE IF IT HELPS--SHOULD IMPLEMENT GRADIENT PENALTY
  }
  self$opt_gen = opt_gen

  self$disc$trainable <- FALSE
  if(!is.null(self$ensemble_size)){
    losses <- list(wasserstein_loss, 'mse')
    loss_weights <- list(1.0, self$content_loss_weight)
  }else{
    losses <- list(custom_loss)
    loss_weights <- list(1.0)
  }
  self$gen_trainer$compile(loss = losses,
                           loss_weights = loss_weights,
                           optimizer = self$opt_gen)

  self$disc$trainable <- TRUE
  self$gen$trainable <- FALSE
  if(!is.null(self$ensemble_size)){
    losses_d <- list(wasserstein_loss, wasserstein_loss)
  }else{
    losses_d <- list(custom_loss, custom_loss)
  }
  self$disc_trainer$compile(
    loss = losses_d,
    loss_weights = list(1.0, 1.0),
    optimizer = self$opt_disc
  )
  self$gen$trainable <- TRUE
  return(self)
}

#' Train
#'
#' @param self
#' @param batch_gen
#' @param noise_gen
#' @param num_gen_batches
#' @param training_ratio
#' @param show_progress
#'
#' @return
#' @export
#'
#' @examples
train <- function(self, batch_gen, noise_gen, num_gen_batches = 1,
                  training_ratio = 1, show_progress = TRUE){

  ############################################################################
  ##### Get batch size from data
  disc_target_real <- NULL

  dataset_iterator <- batch_gen$take(1L)$as_numpy_iterator()
  tmp_batch        <- dataset_iterator %>% iter_next()
  if(length(tmp_batch) == 2) batch_size       <- dim(tmp_batch[[1]][[1]])[1]
  if(length(tmp_batch) == 3) batch_size       <- dim(tmp_batch[[1]])[1]
  rm(tmp_batch, dataset_iterator)
  ############################################################################


  if(show_progress){
    # Initialize progbar and batch counter
    progbar <- progress_bar$new(total = num_gen_batches)
  }
  disc_target_real <- tf$Variable(initial_value = array(1, dim = c(batch_size, 1)))
  disc_target_fake <- -disc_target_real
  gen_target       <- disc_target_real
  disc_target      <- list(disc_target_real, disc_target_fake)

  batch_gen_iter   <- batch_gen %>% as_iterator()

  for(k in 1:num_gen_batches){
    # train discriminator
    disc_loss   <- NULL
    disc_loss_n <- 0
    for(rep in 1:training_ratio){
      #        cat(".")
      # generate some real samples
      workaround <- batch_gen_iter %>% iter_next()
      if(length(workaround) == 2){
        cond   <- workaround[[1]][[1]]
        const  <- workaround[[1]][[2]]
        sample <- workaround[[2]][[1]]
      }else{
        if(length(workaround) == 3){
          cond   <- workaround[[1]]
          const  <- workaround[[2]]
          sample <- workaround[[3]]
        }else{
          stop("There's something wiggy about the data here.")
        }
      }

      if(self$arch == "ctrl"){
        cond   <- cond[ , , , 1, ]
      }
      if(self$arch == "ens"){
        cond <- cond[ , , , 1:ensemble_members, , drop = FALSE]
      }

      #cat("About to train step 1\n")

      self$gen$trainable <- FALSE
      dl <- self$disc_trainer$train_on_batch(
        list(cond, const, noise_gen(), sample), disc_target, return_dict = T)
      self$gen$trainable <- TRUE

      dl <- unlist(dl)

      if(is.null(disc_loss)){
        disc_loss <- array(dl)
      }else{
        disc_loss <- disc_loss + array(dl)
      }
      disc_loss_n <- disc_loss_n + 1

      rm(sample, cond, const)
    }


    disc_loss <- disc_loss/disc_loss_n

    self$disc$trainable <- FALSE

    workaround <- batch_gen_iter$get_next()
    if(length(workaround) == 2){
      cond   <- workaround[[1]][[1]]
      const  <- workaround[[1]][[2]]
      sample <- workaround[[2]][[1]]
    }else{
      if(length(workaround) == 3){
        cond   <- workaround[[1]]
        const  <- workaround[[2]]
        sample <- workaround[[3]]
      }else{
        stop("There's something wiggy about the data here.")
      }
    }

    if(self$arch == "ctrl"){
      cond   <- cond[ , , , 1, ]
    }
    if(self$arch == "ens"){
      cond <- cond[ , , , 1:ensemble_members, , drop = FALSE]
    }

    condconst <- list(cond, const)
    if(is.null(self$ensemble_size)){
      gt_outputs <- list(gen_target)
      noise_list <- list(noise_gen())
    }else{
      noise_list = lapply(1:self$ensemble_size, function(x) noise_gen())
      gt_outputs = c(gen_target, sample)
    }
    gt_inputs <- c(condconst, noise_list)

    #cat("About to train step 2\n")

    if(self$mode == 'GAN'){
      gen_loss = self$gen_trainer$train_on_batch(
        gt_inputs, gt_outputs)
    }

    if(!is.list(gen_loss)) gen_loss <- as.list(gen_loss)
    rm(sample, cond, const)
    self$disc$trainable <- TRUE

    if(show_progress){
      progbar$tick()
    }

    loss_log = list()
    if(self$mode == "det"){
      stop("Doctor, what are you doing here? You're supposed to be on Gallifrey")
    }
    if(self$mode == "GAN"){
      loss_log["disc_loss"]      <- disc_loss[1]
      loss_log["disc_loss_real"] <- disc_loss[2]
      loss_log["disc_loss_fake"] <- disc_loss[3]
      loss_log["gen_loss_total"] <- gen_loss[[1]]
      if(!is.null(self$ensemble_size)){
        loss_log["gen_loss_disc"] <- gen_loss[[2]]
        loss_log["gen_loss_ct"]   <- gen_loss[[3]]
      }
    }
    gc()

    ## This is a bit hard coded, but not super important. Will change if time.
    if(k %in% seq(200, 3200, 200)){
      data <- data.frame(training_samples = training_samples + batch_size * k)
      for(foo in names(loss_log)){
        data[[foo]] <- loss_log[[foo]]
      }
      if(!file.exists(log_file)){
        write.table(data, file = log_file, row.names = FALSE)
      }else{
        write.table(data, file = log_file, append = TRUE, col.names = FALSE, row.names = FALSE)
      }
    }

  }
  return(loss_log)
}

