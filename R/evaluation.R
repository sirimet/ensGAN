#' Denormalise precipitation
#'
#' @param x
#'
#' @return
#' @export
#'
#' @examples
denormalise <- function(x){
  return(pmin(10**x - 1, 500))
}

#' Setup Inputs
#'
#' @param ...
#' @param mode
#' @param arch
#' @param val_years
#' @param downsample
#' @param weights
#' @param input_channels
#' @param batch_size
#' @param num_batches
#' @param filters_gen
#' @param filters_disc
#' @param noise_channels
#' @param latent_variables
#' @param ensemble_members
#' @param padding
#' @param load_full_image
#'
#' @return
#' @export
#'
#' @examples
setup_inputs <- function(...,
                         mode,
                         arch,
                         val_years,
                         downsample,
                         weights,
                         input_channels,
                         batch_size,
                         num_batches,
                         filters_gen,
                         filters_disc,
                         noise_channels,
                         latent_variables,
                         ensemble_members = ensemble_members,
                         padding,
                         load_full_image){
  if(load_full_image){
    # small batch size to prevent memory issues
    batch_size  <- 1
  }else{
    batch_size  <- batch_size
    num_batches <- num_batches
  }

  # initialise model
  model <- setup_model(mode=mode,
                       arch=arch,
                       input_channels=input_channels,
                       filters_gen=filters_gen,
                       filters_disc=filters_disc,
                       noise_channels=noise_channels,
                       latent_variables=latent_variables,
                       ensemble_members=ensemble_members,
                       padding=padding)

  gen   <- model$gen

  if(load_full_image){
    cat('\nLoading full sized image dataset\n')
    # load full size image
    c(batch_gen_train, batch_gen_valid) %<-% setup_data(
      load_full_image = TRUE,
      val_years       = val_years,
      batch_size      = batch_size,
      downsample      = downsample)
  }else{
    cat('\nEvaluating with smaller image dataset\n')
    c(batch_gen_train, batch_gen_valid) %<-% setup_data(
      load_full_image = FALSE,
      val_years       = val_years,
      val_size        = val_size,
      downsample      = downsample,
      weight          = weights,
      batch_size      = batch_size)
  }
  return(list(gen = gen, batch_gen_valid = batch_gen_valid))
}


#' Title
#'
#' @param x
#' @param rnd_mean
#' @param rnd_range
#'
#' @return
#' @export
#'
#' @examples
randomize_nans <- function(x, rnd_mean, rnd_range){
  nan_mask  = is.nan(x)
  nan_shape = sum(nan_mask)
  x[nan_mask] = rnd_mean + (runif(nan_shape)-0.5)*rnd_range
  # return(x) # ???
}

#' Title
#'
#' @param ...
#' @param mode
#' @param gen
#' @param batch_gen
#' @param noise_channels
#' @param latent_variables
#' @param batch_size
#' @param num_batches
#' @param noise_offset
#' @param noise_mul
#' @param denormalise_data
#' @param add_noise
#' @param rank_samples
#' @param noise_factor
#' @param normalize_ranks
#' @param load_full_image
#' @param max_pooling
#' @param avg_pooling
#' @param show_progress
#' @param total_samples
#' @param eval_on
#'
#' @return
#' @export
#'
#' @examples
ensemble_ranks <- function(...,
                           mode,
                           gen,
                           batch_gen,
                           noise_channels,
                           latent_variables,
                           batch_size,
                           num_batches,
                           noise_offset = 0.0,
                           noise_mul = 1.0,
                           denormalise_data = TRUE,
                           add_noise = TRUE,
                           rank_samples = 100,
                           noise_factor = NULL,
                           normalize_ranks = TRUE,
                           load_full_image = FALSE,
                           max_pooling = FALSE,
                           avg_pooling = FALSE,
                           show_progress = TRUE,
                           total_samples = 90,
                           eval_on = eval_on){

  cat("Evaluating on", eval_on, "\n")
  ranks_ens  = list()
  ranks_full = list()
  crps_scores_ens <- c() # list()
  crps_scores_full = c()

  batch_gen_iter <- batch_gen %>% as_iterator()

  if(show_progress){
    # Initialize progbar and batch counter
    progbar <- progress_bar$new(total = num_batches)
  }

  pooling_methods = list('no_pooling')
  if(max_pooling){
    pooling_methods <- c(pooling_methods, 'max_4')
    pooling_methods <- c(pooling_methods, 'max_16')
    pooling_methods <- c(pooling_methods, 'max_8_no_overlap')
  }
  if(avg_pooling){
    pooling_methods <- c(pooling_methods, 'avg_4')
    pooling_methods <- c(pooling_methods, 'avg_16')
    pooling_methods <- c(pooling_methods, 'avg_8_no_overlap')
  }

  for(k in 1:floor(total_samples / batch_size)){
    cat(k, "\n")
    # load truth images
    if(data_type == "mixed"){ ## changed from mod_type to data_type
      workaround <- batch_gen_iter %>% iter_next()
      cond <- workaround[[1]][[1]]
      const <- workaround[[1]][[2]]
      sample <- workaround[[2]][[1]]
      sample_truth <- sample$numpy()
      raw    <- cond[ , , , 3, drop = FALSE] # precipitation_amount_acc
      raw    <- denormalise(raw$numpy())
      raw    <- raw %>% keras::layer_upsampling_2d(c(8,8))
      if(!denormalise_data) raw <- log10(raw + 1)
    }else{
      workaround <- batch_gen_iter %>% iter_next()
      ens <- workaround[[1]]
      if(arch %in% c("ctrl", "mixed")){
        cond <- ens[ , , , 1, ]
      }else{
        cond <- ens[ , , , 1:ensemble_members, ]
      }
      const  <- workaround[[2]] #[['hi_res_inputs']]
      sample <- workaround[[3]] #[['output']]
      sample_truth = sample$numpy()
      raw    <- ens[ , , , , 3] # precipitation_amount_acc
      raw    <- denormalise(raw$numpy())
      raw    <- raw %>% keras::layer_upsampling_2d(c(8,8))
      if(!denormalise_data) raw <- log10(raw + 1)
    }
    if(denormalise_data){
      sample_truth = denormalise(sample_truth)
    }
    if(add_noise){
      c(noise_dim_1, noise_dim_2) %<-% dim(sample_truth)[-c(1, length(dim(sample_truth)))]
      noise        <- array(runif(batch_size * noise_dim_1 * noise_dim_2 * 1), dim = c(batch_size, noise_dim_1, noise_dim_2, 1))
      if(!is.null(noise_factor)) noise <- noise * noise_factor
      sample_truth <- sample_truth + noise
    }

    # generate predictions
    gen_ens  <- list()
    gen_ens0 <- list()

    if(arch %in% c("ctrl", "mixed")){
      noise_shape <- c(dim(cond)[-c(1, length(dim(cond)))], noise_channels)
      noise_gen = noise_generator(noise_shape, batch_size = batch_size)

      cond0 <- ens[ , , , 1, ]
      for(em in 1:dim(ens)[4]){
        #cat("Ensemble member", em, "\n")
        samples_gen <- list()
        samples_gen0 <- list()
        cond <- ens[ , , , em, ]
        for(i in 1:rank_samples){
          nn = noise_gen()
          nn = nn * noise_mul
          nn = nn - noise_offset
          if("full" %in% eval_on){
            sample_gen = gen$predict(list(cond, const, nn))
            for (imgur in 1:batch_size){
              sample_gen[imgur, , , 1] <- pmin(sample_gen[imgur, , , 1], log_maxes)
            }
            samples_gen <- c(samples_gen, tf$cast(sample_gen, "float32"))
          }
          if("ens0" %in% eval_on){
            sample_gen0 <- gen$predict(list(cond0, const, nn))
            for(imgur in 1:batch_size){
              sample_gen0[imgur, , , 1] <- pmin(sample_gen0[imgur, , , 1], log_maxes)
            }
            samples_gen0 <- c(samples_gen0, tf$cast(sample_gen0, "float32"))
          }
        }

        for(ii in 1:rank_samples){
          if("full" %in% eval_on) sample_gen <- abind::adrop(as.array(samples_gen[[ii]]), drop = length(dim(samples_gen[[ii]])))
          if("ens0" %in% eval_on) sample_gen0 <- abind::adrop(as.array(samples_gen0[[ii]]), drop = length(dim(samples_gen0[[ii]])))
          if(denormalise_data){
            if("full" %in% eval_on) sample_gen = denormalise(sample_gen)
            if("ens0" %in% eval_on) sample_gen0 = denormalise(sample_gen0)
          }
          if(add_noise){
            if("full" %in% eval_on) {
              c(noise_dim_1, noise_dim_2) %<-% dim(sample_gen)[-1]
            }else{
              c(noise_dim_1, noise_dim_2) %<-% dim(sample_gen0)[-1]
            }
            noise      <- array(runif(batch_size * noise_dim_1 * noise_dim_2), dim = c(batch_size, noise_dim_1, noise_dim_2))
            if(!is.null(noise_factor)) noise <- noise * noise_factor
            if("full" %in% eval_on) sample_gen <- sample_gen + noise
            if("ens0" %in% eval_on) sample_gen0 <- sample_gen0 + noise
          }
          if("full" %in% eval_on) samples_gen[[ii]] <- sample_gen
          if("ens0" %in% eval_on) samples_gen0[[ii]] <- sample_gen0
        }

        if("full" %in% eval_on) {
          samples_gen <- abind::abind(samples_gen, along = length(dim(samples_gen[[1]]))+1)
          gen_ens[[em]] <- samples_gen
        }
        if("ens0" %in% eval_on) {
          samples_gen0 <- abind::abind(samples_gen0, along = length(dim(samples_gen0[[1]]))+1)
          gen_ens0[[em]] <- samples_gen0
        }
      }
    }

    if(arch == "ens"){
      noise_shape <- c(dim(cond)[-c(1, (length(dim(cond))-1):length(dim(cond)))], noise_channels)
      noise_gen = noise_generator(noise_shape, batch_size = batch_size)

      samples_gen <- list()
      samples_gen0 <- list()
      for(i in 1:rank_samples){
        nn = noise_gen()
        nn = nn * noise_mul
        nn = nn - noise_offset
        if("full" %in% eval_on){
          sample_gen = gen$predict(list(cond, const, nn))
          for (imgur in 1:batch_size){
            sample_gen[imgur, , , 1] <- pmin(sample_gen[imgur, , , 1], log_maxes)
          }
          samples_gen <- c(samples_gen, tf$cast(sample_gen, "float32"))
        }
      }

      for(ii in 1:rank_samples){ # length(samples_gen0)
        # sample_gen = np.squeeze(samples_gen[ii], axis=-1) # squeeze out trival dim
        if("full" %in% eval_on) sample_gen <- abind::adrop(as.array(samples_gen[[ii]]), drop = length(dim(samples_gen[[ii]])))
        if(denormalise_data){
          if("full" %in% eval_on) sample_gen = denormalise(sample_gen)
        }
        if(add_noise){
          if("full" %in% eval_on) {
            c(noise_dim_1, noise_dim_2) %<-% dim(sample_gen)[-1]
          }else{
            c(noise_dim_1, noise_dim_2) %<-% dim(sample_gen0)[-1]
          }
          noise      <- array(runif(batch_size * noise_dim_1 * noise_dim_2), dim = c(batch_size, noise_dim_1, noise_dim_2))
          if(!is.null(noise_factor)) noise <- noise * noise_factor
          if("full" %in% eval_on) sample_gen <- sample_gen + noise
          #                      if("ens0" %in% eval_on) sample_gen0 <- sample_gen0 + noise
        }
        if("full" %in% eval_on) samples_gen[[ii]] <- sample_gen
        #              if("ens0" %in% eval_on) samples_gen0[[ii]] <- sample_gen0
      }

      if("full" %in% eval_on) {
        samples_gen <- abind::abind(samples_gen, along = length(dim(samples_gen[[1]]))+1)
        gen_ens[[1]] <- samples_gen
      }
    }

    if("full" %in% eval_on) full_sample_ens <- abind::abind(gen_ens, along = 4)
    if("ens0" %in% eval_on) ens0_sample_ens <- abind::abind(gen_ens0, along = 4)

    # calculate ranks
    # currently ranks only calculated without pooling
    # probably fine but may want to threshold in the future, e.g. <1mm, >5mm
    sample_truth_ranks <- array(sample_truth, dim = c(length(sample_truth), 1)) # unwrap into one long array, then unwrap samples_gen in same format

    if("ens0" %in% eval_on){
      samples_gen_ranks  <- array(ens0_sample_ens, dim = c(prod(dim(ens0_sample_ens))/(rank_samples*length(gen_ens0)), (rank_samples*length(gen_ens0)))) # rank_samples = 100 by default??
      rank <- rowSums(apply(samples_gen_ranks, 2, function(x) sample_truth_ranks >= x))
      ranks_ens <- c(ranks_ens, rank)
      rm(samples_gen_ranks, rank)
      gc()
    }else{
      cat("Skipping ens\n")
    }

    if("full" %in% eval_on){
      full_ranks  <- array(full_sample_ens, dim = c(prod(dim(full_sample_ens))/(rank_samples*length(gen_ens)), (rank_samples*length(gen_ens)))) # rank_samples = 100 by default??
      rank_full <- rowSums(apply(full_ranks, 2, function(x) sample_truth_ranks >= x))
      ranks_full <- c(ranks_full, rank_full)
    }else{
      cat("Skipping full\n")
    }

    if("ens0" %in% eval_on){
      # Calculate CRPS scores for each ensemble memeber
      for(em in 1){
        samples_gen <- ens0_sample_ens

        # calculate CRPS scores for different pooling methods
        for(method in pooling_methods){
          #cat(method, "\n")
          if(method == 'no_pooling'){
            sample_truth_pooled = sample_truth
            samples_gen_pooled = samples_gen
          }else{
            sample_truth_pooled = pool(sample_truth, method)[[1]]
            samples_gen_pooled = pool(samples_gen, method)[[1]]
          }
          # USE QUICKER CRPS FUNCTION IN C++ (from JBBremnes)
          samples_gen_pooled <- matrix(samples_gen_pooled, ncol = dim(samples_gen)[4])
          sample_truth_pooled <- as.numeric(sample_truth_pooled)
          if(crps_fast){
            crps_score <- mean(unlist(mclapply(1:nrow(samples_gen_pooled), function(x) crps_ensemble_fast(samples_gen_pooled[x, , drop = FALSE], sample_truth_pooled[x]), mc.cores = detectCores())))
          }else{
            crps_score <- mean(unlist(mclapply(1:nrow(samples_gen_pooled), function(x) crps_ensemble(samples_gen_pooled[x, , drop = FALSE], sample_truth_pooled[x]), mc.cores = detectCores())))
          }
          rm(sample_truth_pooled, samples_gen_pooled)
          gc()

          if(!method %in% names(crps_scores_ens)){
            crps_scores_ens[[method]] <- crps_score
          }else{
            crps_scores_ens[[method]] <- c(crps_scores_ens[[method]], crps_score)
          }
        }
      }
    }else{
      cat("Skipping ens\n")
    }

    if("full" %in% eval_on){
      for(method in pooling_methods){
        #            cat(method, "\n")
        if(method == 'no_pooling'){
          sample_truth_pooled = sample_truth
          samples_gen_pooled = full_sample_ens
        }else{
          sample_truth_pooled = pool(sample_truth, method)[[1]]
          samples_gen_pooled = pool(full_sample_ens, method)[[1]]
        }
        # USE QUICKER CRPS FUNCTION IN C++ (from JBBremnes)
        samples_gen_pooled <- matrix(samples_gen_pooled, ncol = dim(full_sample_ens)[4])
        sample_truth_pooled <- as.numeric(sample_truth_pooled)
        if(crps_fast){
          crps_score <- mean(unlist(mclapply(1:nrow(samples_gen_pooled), function(x) crps_ensemble_fast(samples_gen_pooled[x, , drop = FALSE], sample_truth_pooled[x]), mc.cores = detectCores())))
        }else{
          crps_score <- mean(unlist(mclapply(1:nrow(samples_gen_pooled), function(x) crps_ensemble(samples_gen_pooled[x, , drop = FALSE], sample_truth_pooled[x]), mc.cores = detectCores())))
        }
        rm(sample_truth_pooled, samples_gen_pooled)
        gc()
        crps_scores_full[[method]] <- c(crps_scores_full[[method]], crps_score)
      }
    }else{
      cat("Skipping full\n")
    }


    if(show_progress){
      progbar$tick()
    }
  }

  if("ens0" %in% eval_on) ranks_ens  <- lapply(ranks_ens, unlist)
  if("full" %in% eval_on) ranks_full <- lapply(ranks_full, unlist)
  gc()

  if(normalize_ranks){
    if("ens0" %in% eval_on) ranks_ens  <- lapply(ranks_ens,  function(x) x / (rank_samples*length(gen_ens0)))
    if("full" %in% eval_on) ranks_full <- lapply(ranks_full, function(x) x / (rank_samples*length(gen_ens)))
    gc()
  }

  return(list(ranks_full = ranks_full, crps_scores_full = crps_scores_full, ranks_ens = ranks_ens, crps_scores_ens = crps_scores_ens))
}



#' Title
#'
#' @param norm_ranks
#' @param num_ranks
#'
#' @return
#' @export
#'
#' @examples
rank_KS <- function(norm_ranks, num_ranks = 100){

  ## python's zero indexing might or might not be a problem here. will need to come back and check.

  # (h, b) = np.histogram(norm_ranks, num_ranks+1)
  tmp <- hist(norm_ranks, breaks = seq(min(norm_ranks), max(norm_ranks), length.out = num_ranks + 2), right = FALSE, plot = FALSE)
  h   <- tmp$counts
  b   <- tmp$breaks

  # h = h / h.sum()
  h   <- h / sum(h)

  # ch = np.cumsum(h)
  ch  <- cumsum(h)

  # cb = b[1:]
  cb  <- b[-1]
  return(max(abs(ch-cb)))
}


#' Title
#'
#' @param norm_ranks
#' @param num_ranks
#'
#' @return
#' @export
#'
#' @examples
rank_CvM <- function(norm_ranks, num_ranks = 100){
  # (h, b) = np.histogram(norm_ranks, num_ranks+1)
  tmp <- hist(norm_ranks, breaks = seq(min(norm_ranks), max(norm_ranks), length.out = num_ranks+2), right = FALSE, plot = FALSE)
  h   <- tmp$counts
  b   <- tmp$breaks

  h   <- h / sum(h)
  ch  <- cumsum(h)
  cb  <- b[-1]
  db  <- diff(b)

  return(sqrt(sum((ch-cb)**2 * db)))
}


#' Title
#'
#' @param norm_ranks
#' @param num_ranks
#'
#' @return
#' @export
#'
#' @examples
rank_DKL <- function(norm_ranks, num_ranks = 100){
  tmp <- hist(norm_ranks, breaks = seq(min(norm_ranks), max(norm_ranks), length.out = num_ranks+2), right = FALSE, plot = FALSE)
  h   <- tmp$counts
  b   <- tmp$breaks

  q   <- h / sum(h)
  p   <- 1/length(h)

  return(p*sum(log(p/q)))
}


#' Title
#'
#' @param norm_ranks
#' @param num_ranks
#'
#' @return
#' @export
#'
#' @examples
rank_OP <- function(norm_ranks, num_ranks = 100){
  op <- sum(norm_ranks %in% c(0,1))
  op <- op/length(norm_ranks)
  return(op)
}


#' Title
#'
#' @param log_fname
#' @param line
#'
#' @return
#' @export
#'
#' @examples
log_line <- function(log_fname, line){
  if(!file.exists(log_fname)){
    sink(log_fname)
    cat(line, "\n")
    sink()
  }else{
    sink(log_fname, append = TRUE)
    cat(line, "\n")
    sink()
  }
}


#' Title
#'
#' @param ...
#' @param mode
#' @param arch
#' @param val_years
#' @param log_fname
#' @param weights_dir
#' @param downsample
#' @param weights
#' @param add_noise
#' @param noise_factor
#' @param load_full_image
#' @param model_numbers
#' @param ranks_to_save
#' @param batch_size
#' @param num_batches
#' @param filters_gen
#' @param filters_disc
#' @param input_channels
#' @param latent_variables
#' @param noise_channels
#' @param padding
#' @param rank_samples
#' @param max_pooling
#' @param avg_pooling
#' @param total_samples
#' @param ensemble_members
#' @param eval_on
#'
#' @return
#' @export
#'
#' @examples
rank_metrics_by_time <- function(...,
                                 mode,
                                 arch,
                                 val_years,
                                 log_fname,
                                 weights_dir,
                                 downsample = FALSE,
                                 weights = NULL,
                                 add_noise = TRUE,
                                 noise_factor = NULL,
                                 load_full_image = FALSE,
                                 model_numbers = NULL,
                                 ranks_to_save = NULL,
                                 batch_size = NULL,
                                 num_batches = NULL,
                                 filters_gen = NULL,
                                 filters_disc = NULL,
                                 input_channels = NULL,
                                 latent_variables = NULL,
                                 noise_channels = NULL,
                                 padding = NULL,
                                 rank_samples = NULL,
                                 max_pooling = FALSE,
                                 avg_pooling = FALSE,
                                 total_samples = 90,
                                 ensemble_members = NULL,
                                 eval_on = c("ens0", "full")){

  c(gen, batch_gen_valid) %<-% setup_inputs(mode = mode,
                                            arch = arch,
                                            val_years = val_years,
                                            downsample = downsample,
                                            weights = weights,
                                            input_channels = input_channels,
                                            batch_size = batch_size,
                                            num_batches = num_batches,
                                            filters_gen = filters_gen,
                                            filters_disc = filters_disc,
                                            noise_channels = noise_channels,
                                            latent_variables = latent_variables,
                                            ensemble_members = ensemble_members,
                                            padding = padding,
                                            load_full_image = load_full_image)

  for(mod in eval_on){
    log_fname_mod <- paste0(dirname(log_fname), "/", mod, "_", basename(log_fname))
    log_line(log_fname_mod, "N,KS,CvM,DKL,OP,CRPS,CRPS_max_4,CRPS_max_16,CRPS_max_8_no_overlap,CRPS_avg_4,CRPS_avg_16,CRPS_avg_8_no_overlap,mean,std")
  }

  if(!substr(weights_dir, nchar(weights_dir), nchar(weights_dir)) == "/") weights_dir <- paste0(weights_dir, "/")

  for(model_number in model_numbers){
    gen_weights_file = paste0(weights_dir, "gen_weights-", formatC(model_number, digits = 6, flag = 0), ".h5")
    if(!file.exists(gen_weights_file)){
      print(paste0(gen_weights_file, " not found, skipping"))
      next
    }

    print(gen_weights_file)
    gen$load_weights(gen_weights_file)
    c(ranks_full, crps_scores_full, ranks_ens, crps_scores_ens) %<-%
      ensemble_ranks(mode = mode,
                     gen = gen,
                     batch_gen = batch_gen_valid,
                     noise_channels = noise_channels,
                     latent_variables = latent_variables,
                     batch_size = batch_size,
                     num_batches = num_batches,
                     add_noise = add_noise,
                     rank_samples = rank_samples,
                     noise_factor = noise_factor,
                     load_full_image = load_full_image,
                     max_pooling = max_pooling,
                     avg_pooling = avg_pooling,
                     total_samples = total_samples,
                     eval_on = eval_on)

    ranks_list <- list(full = ranks_full,
                       ens0 = ranks_ens)

    crps_scores_list <- list(full = crps_scores_full,
                             ens0 = crps_scores_ens)

    for(mod in eval_on){
      log_fname_mod <- paste0(dirname(log_fname), "/", mod, "_", basename(log_fname))
      ranks <- unlist(ranks_list[[mod]])
      crps_scores <- crps_scores_list[[mod]]
      KS  <- rank_KS(ranks)
      CvM <- rank_CvM(ranks)
      DKL <- rank_DKL(ranks)
      OP  <- rank_OP(ranks)
      CRPS_no_pool <- mean(as.array(crps_scores[['no_pooling']]))
      if(max_pooling){
        CRPS_max_4  = mean(as.array(crps_scores[['max_4']]))
        CRPS_max_16 = mean(as.array(crps_scores[['max_16']]))
        CRPS_max_8_no_overlap = mean(as.array(crps_scores[['max_8_no_overlap']]))
      }else{
        CRPS_max_4  = NaN
        CRPS_max_16 = NaN
        CRPS_max_8_no_overlap = NaN
      }
      if(avg_pooling){
        CRPS_avg_4  = mean(as.array(crps_scores[['avg_4']]))
        CRPS_avg_16 = mean(as.array(crps_scores[['avg_16']]))
        CRPS_avg_8_no_overlap = mean(as.array(crps_scores[['avg_8_no_overlap']]))
      }else{
        CRPS_avg_4  = NaN
        CRPS_avg_16 = NaN
        CRPS_avg_8_no_overlap = NaN
      }
      mean <- mean(ranks)
      std  <- sd(ranks)

      log_line(log_fname_mod, paste0(model_number,",", paste0(signif(c(KS, CvM, DKL, OP, CRPS_no_pool, CRPS_max_4, CRPS_max_16, CRPS_max_8_no_overlap,
                                                                       CRPS_avg_4, CRPS_avg_16, CRPS_avg_8_no_overlap, mean, std), digits = 6), collapse = ",")))

      # save one directory up from model weights, in same dir as logfile
      ranks_folder = dirname(log_fname)

      # This is super python specific, so I guess I will make it equally R specific
      if(model_number %in% ranks_to_save){
        if(!add_noise & !load_full_image){
          # fname = 'ranks-small_image-{}.npz'.format(model_number)
          fname = paste0(mod, "_ranks-small_image-", model_number, ".RData")
        }else if(add_noise & !load_full_image){
          # fname = 'ranks-small_image-noise-{}.npz'.format(model_number)
          fname = paste0(mod, "_ranks-small_image-noise-", model_number, ".RData")
        } else if(!add_noise & load_full_image){
          # fname = 'ranks-full_image-{}.npz'.format(model_number)
          fname = paste0(mod, "_ranks-full_image-", model_number, ".RData")
        } else if(add_noise & load_full_image){
          # fname = 'ranks-full_image-noise-{}.npz'.format(model_number)
          fname = paste0(mod, "_ranks-full_image-noise-", model_number, ".RData")
        }
        # np.savez(os.path.join(ranks_folder, fname), ranks)
        base::save(ranks, file = paste(ranks_folder, fname, sep = "/"))
      }
    }
  }
}


#' Title
#'
#' @param img1
#' @param img2
#'
#' @return
#' @export
#'
#' @examples
log_spectral_distance <- function(img1, img2){
  power_spectrum_dB <- function(img){
    fx <- fft(img)
    # fx = fx[:img.shape[0]//2, :img.shape[1]//2]
    fx <- fx[1:(dim(img)[1]%/%2), 1:(dim(img)[2]%/%2)]
    px <- abs(fx)**2
    return(10 * log10(px))
  }

  d = (power_spectrum_dB(img1)-power_spectrum_dB(img2))**2

  # d[~np.isfinite(d)] = np.nan
  d[is.infinite(d)] = NaN
  return(sqrt(mean(d, na.rm = T)))
}


#' Title
#'
#' @param batch1
#' @param batch2
#'
#' @return
#' @export
#'
#' @examples
log_spectral_distance_batch <- function(batch1, batch2){
  lsd_batch = list()
  for(i in 1:nrow(batch1)){
    lsd = log_spectral_distance(
      batch1[i, , , ], batch2[i, , , ]
    )
    lsd_batch <- c(lsd_batch, lsd)
  }

  return(lsd_batch) # original code had return np.array(lsd_batch) so perhaps list is not right? We'll see
}

#' Title
#'
#' @param truth
#' @param pred
#'
#' @return
#' @export
#'
#' @examples
calculate_rapsd_rmse <- function(truth, pred){
  ## avoid producing inf values by removing RAPSD calc for images
  ## that are mostly zeroes (mean pixel value < 0.01)
  if (mean(truth) < 0.002 | mean(pred) < 0.002){
    return(NaN)
  }

  ### HEYYYYY an entire new script to translate. Magnificent! Will get back to that then.
  fft_freq_truth = rapsd(truth, fft_method = fft)
  fft_freq_pred  = rapsd(pred,  fft_method = fft)
  truth <- 10 * log10(fft_freq_truth)
  pred  <- 10 * log10(fft_freq_pred)
  rmse  <- sqrt(mean((truth-pred)**2, na.rm = TRUE))
  return(rmse)
}

#' Title
#'
#' @param batch1
#' @param batch2
#'
#' @return
#' @export
#'
#' @examples
rapsd_batch <- function(batch1, batch2){
  # radially averaged power spectral density
  ## squeeze out final dimension (channels)
  if(length(dim(batch1)) == 4){
    batch1 = drop(batch1)
  }
  if(length(dim(batch2)) == 4){
    batch2 = drop(batch2)
  }
  rapsd_batch = list()
  for(i in 1:nrow(batch1)){
    rapsd_score = calculate_rapsd_rmse(
      batch1[i, , ], batch2[i, , ]) ## Should accommodate all shapes, I suppose batch1[i,...], batch2[i,...]. Oh well.
    if(!is.nan(rapsd_score)){
      rapsd_batch <- c(rapsd_batch, rapsd_score)
    }
  }
  return(rapsd_batch) # again, possibly not a list? np.array(rapsd_batch)
}

#' Title
#'
#' @param ...
#' @param mode
#' @param gen
#' @param batch_gen
#' @param noise_channels
#' @param latent_variables
#' @param batch_size
#' @param num_instances
#' @param num_batches
#' @param load_full_image
#' @param denormalise_data
#' @param show_progress
#'
#' @return
#' @export
#'
#' @examples
image_quality <- function(...,
                          mode,
                          gen,
                          batch_gen,
                          noise_channels,
                          latent_variables,
                          batch_size,
                          num_instances = 100,
                          num_batches = 100,
                          load_full_image = FALSE,
                          denormalise_data = TRUE,
                          show_progress = TRUE){
  batch_gen_iter = batch_gen %>% as_iterator()

  num_batches <- floor(90/batch_size)

  scores <- list()
  for(i in 1:11){
    scores[[i]] = list(mae_all   = c(), #list()
                       mse_all   = c(),
                       ssim_all  = c(),
                       lsd_all   = c(),
                       rapsd_all = c())
  }

  if(show_progress){
    progbar <- progress_bar$new(total = num_batches)
  }

  for(k in 1:num_batches){
    cat(k, "out of", num_batches, "\n")
    if(load_full_image){
      workaround <- batch_gen_iter %>% iter_next()
      ens <- workaround[[1]]
      if(length(dim(ens) == 5)){
        cond <- ens[ , , , 1, ]
      }else{
        cond <- ens
      }

      const  <- workaround[[2]] #[['hi_res_inputs']]
      sample <- workaround[[3]] #[['output']]
      sample <- sample$numpy()
      raw    <- ens[ , , , , 3]
      raw    <- denormalise(raw$numpy())
      raw    <- raw %>% keras::layer_upsampling_2d(c(8,8))
      if(!denormalise_data) raw <- log10(raw + 1)
    }else{
      workaround <- batch_gen_iter %>% iter_next()
      ens <- workaround[[1]]
      if(length(dim(ens)) == 5){
        cond <- ens[ , , , 1, ]
      }else{
        cond <- ens
      }
      const  <- workaround[[2]] #[['hi_res_inputs']]
      sample <- workaround[[3]] #[['output']]
      sample <- sample$numpy()
      raw    <- ens[ , , , , 3]
      raw    <- denormalise(raw$numpy())
      raw    <- raw %>% keras::layer_upsampling_2d(c(8,8))
      if(!denormalise_data) raw <- log10(raw + 1)
    }
    if(denormalise_data){
      sample = denormalise(sample)
    }

    noise_shape <- c(dim(as.array(cond[1, , , 1])), noise_channels)
    noise_gen   <- noise_generator(noise_shape, batch_size = batch_size) # * num_batches) # going a bit off script here, by replacing batch_size with batch_size * num_batches.

    for(em in 1:11){
      cond <- ens[ , , , , em]

      for(i in 1:num_instances){
        img_gen = gen$predict(list(cond, const, noise_gen()))

        if(denormalise_data){
          img_gen = denormalise(img_gen)
        }

        mae   <- apply((abs(sample - img_gen)), 1, mean)
        mse   <- apply((sample - img_gen)**2, 1, mean)

        ### gaaaaaahh, another script. I was kidding before. It's actually quite miserable to discover one of these
        ssim  <- MultiScaleSSIM(sample, img_gen, 1.0)
        lsd   <- log_spectral_distance_batch(sample, img_gen)
        rapsd1 <- rapsd_batch(sample, img_gen)

        scores[[em]]$mae_all   <- c(scores[[em]]$mae_all, mae)
        scores[[em]]$mse_all   <- c(scores[[em]]$mse_all, mse)
        scores[[em]]$ssim_all  <- c(scores[[em]]$ssim_all, ssim)
        scores[[em]]$lsd_all   <- c(scores[[em]]$lsd_all, lsd)
        scores[[em]]$rapsd_all <- c(scores[[em]]$rapsd_all, rapsd1)
      }
    }

    if(show_progress){
      progbar$tick()
    }
  }

  return(scores)
}




#' Title
#'
#' @param ...
#' @param mode
#' @param arch
#' @param val_years
#' @param log_fname
#' @param weights_dir
#' @param downsample
#' @param weights
#' @param load_full_image
#' @param model_numbers
#' @param batch_size
#' @param num_batches
#' @param filters_gen
#' @param filters_disc
#' @param input_channels
#' @param latent_variables
#' @param noise_channels
#' @param padding
#'
#' @return
#' @export
#'
#' @examples
quality_metrics_by_time <- function(...,
                                    mode,
                                    arch,
                                    val_years,
                                    log_fname,
                                    weights_dir,
                                    downsample = FALSE,
                                    weights = NULL,
                                    load_full_image = FALSE,
                                    model_numbers = NULL,
                                    batch_size = NULL,
                                    num_batches = NULL,
                                    filters_gen = NULL,
                                    filters_disc = NULL,
                                    input_channels = NULL,
                                    latent_variables = NULL,
                                    noise_channels = NULL,
                                    padding = NULL){

  c(gen, batch_gen_valid) %<-% setup_inputs(mode = mode,
                                            arch = arch,
                                            val_years = val_years,
                                            downsample = downsample,
                                            weights = weights,
                                            input_channels = input_channels,
                                            batch_size = batch_size,
                                            num_batches = num_batches,
                                            filters_gen = filters_gen,
                                            filters_disc = filters_disc,
                                            noise_channels = noise_channels,
                                            latent_variables = latent_variables,
                                            padding = padding,
                                            load_full_image = load_full_image)

  num_instances = 100  # samples per image
  for(em in 1:11){
    log_fname_em <- paste0(dirname(log_fname), "/ensemble_member_", em, "_", basename(log_fname))
    log_line(log_fname_em, paste0("Samples per image: ", num_instances))
    log_line(log_fname_em, "N,RMSE,MSSSIM,LSD,RAPSD,MAE")
  }

  if(!substr(weights_dir, nchar(weights_dir), nchar(weights_dir)) == "/") weights_dir <- paste0(weights_dir, "/")
  for(model_number in model_numbers){
    gen_weights_file = paste0(weights_dir, "gen_weights-", formatC(model_number, digits = 6, flag = 0), ".h5")

    if(!file.exists(gen_weights_file)){
      print(paste0(gen_weights_file, " not found, skipping"))
      next
    }

    cat("\n", gen_weights_file, "\n")
    gen$load_weights(gen_weights_file)
    scores %<-% image_quality(mode = mode,
                              gen = gen,
                              batch_gen = batch_gen_valid,
                              noise_channels = noise_channels,
                              latent_variables = latent_variables,
                              batch_size = batch_size,
                              num_instances = num_instances,
                              num_batches = num_batches,
                              load_full_image = load_full_image)

    for(em in 1:11){
      log_fname_em <- paste0(dirname(log_fname), "/ensemble_member_", em, "_", basename(log_fname))
      mse <- scores[[em]]$mse_all
      ssim <- scores[[em]]$ssim_all
      lsd <- scores[[em]]$lsd_all
      rapsd1 <- scores[[em]]$rapsd_all
      mae <- scores[[em]]$mae_all

      log_line(log_fname_em, paste0(model_number, ",", paste0(signif(c(sqrt(mean(unlist(mse))),
                                                                       mean(unlist(ssim)),
                                                                       mean(unlist(lsd), na.rm = TRUE),
                                                                       mean(unlist(rapsd1), na.rm = TRUE),
                                                                       mean(unlist(mae))), digits = 6), collapse = ",")))
    }
  }
}

