#' Title
#'
#' @param model
#' @param filepath
#'
#' @return
#' @export
#'
#' @examples
save_opt_weights <- function(model, filepath){
  f <- filepath
  if(!file.exists(filepath)){
    h5createFile(filepath)
    h5createGroup(f, "optimizer_weights")
  }
  # Save optimizer weights.
  symbolic_weights <- model$optimizer$weights
  if(!is.null(symbolic_weights)){
    weight_values = k_batch_get_value(symbolic_weights)
    weight_names  = list()
    for(i in 1:length(symbolic_weights)){
      w <- symbolic_weights[[i]]
      if(k_backend() == "theano"){
        if(!is.null(w$name) & !(w$name == "/variable")){
          name <- w$name
        }else{
          name <- paste0("param_", i)
        }
      }else{
        if(!is.null(w$name)){
          name <- w$name
        }else{
          name <- paste0("param_", i)
        }
      }
      weight_names[[i]] <- utf8_encode(name)
    }

    for(i in 1:length(symbolic_weights)){
      name <- weight_names[[i]]
      name <- gsub("/", " ", name)
      val  <- weight_values[[i]]
      suppressMessages(h5write(val, f, paste0("optimizer_weights/", i, "_splithere_", name), encoding = "UTF-8", write.attributes=TRUE))
    }
  }
  h5closeAll()
}

#' Title
#'
#' @param model
#' @param filepath
#'
#' @return
#' @export
#'
#' @examples
load_opt_weights <- function(model, filepath){
  f <- H5Fopen(filepath)
  optimizer_weights_group <- f$optimizer_weights
  ## undo the thing I did to keep track of order:
  names(optimizer_weights_group) <- gsub(" ", "/", names(optimizer_weights_group))
  optimizer_weight_names  <- names(optimizer_weights_group)
  index <- strsplit(optimizer_weight_names, "_splithere_")
  optimizer_weight_names <- unlist(lapply(index, function(x) x[2]))
  index <- as.numeric(unlist(lapply(index, function(x) x[1])))
  optimizer_weight_names <- optimizer_weight_names[order(index)]
  optimizer_weights_group <- optimizer_weights_group[order(index)]
  optimizer_weights_group[[1]] <- optimizer_weights_group[[1]][1]
  optimizer_weight_values <- lapply(1:length(optimizer_weights_group), function(i) tf$Variable(optimizer_weights_group[[i]], dtype = "float64"))
  optimizer_weight_values <- lapply(optimizer_weight_values, function(x) tf$cast(x, "int64"))
  model$optimizer$set_weights(optimizer_weight_values)
}

#' Title
#'
#' @param x
#'
#' @return
#' @export
#'
#' @examples
ensure_list <- function(x){
  if(!"list" %in% class(x))
    x = list(x)
  return(x)
}

#' Title
#'
#' @param model
#' @param prefix
#'
#' @return
#' @export
#'
#' @examples
input_shapes <- function(model, prefix){
  model.inputs <- unlist(lapply(model$inputs, as.character))
  model.inputs <- grep(prefix, model.inputs)
  shapes <- list()
  for(il in 1:length(model.inputs)){
    shapes[[il]] <- model$input_shape[[model.inputs[il]]][-1]
  }
  shapes <- shapes
  return(shapes)
}
