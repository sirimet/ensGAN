library(keras)


#' Random Weighted Average
#'
#' @param x
#' @param y
#'
#' @return
#' @export
#'
#' @examples
random_weighted_average <- function(x, y){
  shape = k_shape(x)
  weights = tf$compat$v1$random_uniform(shape[1, drop = FALSE],0,1)
  for(i in 1:(length(k_int_shape(x))-1)){
    weights = k_expand_dims(weights,-1)
  }
  weights <- tf$cast(weights, "float32")
  return(x*weights + y*(1-weights))
}

#' Title
#'
#' @param x
#' @param pad_by
#'
#' @return
#' @export
#'
#' @examples
layer_reflection_padding_2d <- function(x, pad_by){
  padding <- pad_by
  i_pad <- padding[1]
  j_pad <- padding[2]

  x <- tensorflow::tf$pad(x, matrix(c(0, 0, i_pad, i_pad, j_pad, j_pad, 0, 0), nrow = 4, byrow = TRUE), "REFLECT")
  return(x)
}

#' Title
#'
#' @param x
#' @param pad_by
#'
#' @return
#' @export
#'
#' @examples
layer_symmetric_padding_2d <- function(x, pad_by){
  padding <- pad_by

  i_pad <- padding[1]
  j_pad <- padding[2]

  return(tf$pad(x, matrix(c(0, 0, i_pad, i_pad, j_pad, j_pad, 0, 0), nrow = 4, byrow = TRUE), "SYMMETRIC"))
}

#' Title
#'
#' @param object
#' @param target_shape
#' @param input_shape
#' @param batch_input_shape
#' @param batch_size
#' @param dtype
#' @param name
#' @param trainable
#' @param weights
#'
#' @return
#' @export
#'
#' @examples
layer_reshape_custom <- function (object, target_shape, input_shape = NULL, batch_input_shape = NULL,
                                  batch_size = NULL, dtype = NULL, name = NULL, trainable = NULL,
                                  weights = NULL) {

  create_layer(keras$layers$Reshape, object, list(target_shape = normalize_shape(target_shape),
                                                  input_shape = normalize_shape(input_shape), batch_input_shape = normalize_shape(batch_input_shape),
                                                  batch_size = as_nullable_integer(batch_size), dtype = dtype,
                                                  name = name, trainable = trainable, weights = weights))
}

#' Title
#'
#' @param shape
#'
#' @return
#' @export
#'
#' @examples
normalize_shape <- function(shape) {
  ## Kidnapped this function off the internet:
  # Helper function to coerce shape arguments to tuple
  # tf$reshape()/k_reshape() doesn't accept a tf.TensorShape object

  # reflect NULL back
  if (is.null(shape))
    return(shape)

  # if it's a list or a numeric vector then convert to integer
  # NA's in are accepted as NULL
  # also accept c(NA), as if it was a numeric
  if (is.list(shape) || is.numeric(shape) ||
      (is.logical(shape) && all(is.na(shape)))) {

    shape <- lapply(shape, function(value) {
      # Pass through python objects unmodified, only coerce R objects
      # supplied shapes, e.g., to tf$random$normal, can be a list that's a mix
      # of scalar integer tensors and regular integers
      if (inherits(value, "python.builtin.object"))
        return(value)

      # accept NA,NA_integer_,NA_real_ as NULL
      if ((is_scalar(value) && is.na(value)))
        return(NULL)

      if (!is.null(value))
        as.integer(value)
      else
        NULL
    })
  }

  if (inherits(shape, "tensorflow.python.framework.tensor_shape.TensorShape"))
    shape <- as.list(shape$as_list()) # unpack for tuple()

  # coerce to tuple so it's iterable
  tuple(shape)
}


#' Title
#'
#' @param x
#'
#' @return
#' @export
#'
#' @examples
as_shape <- function(x) {
  lapply(x, function(d) {
    if (is.null(d))
      NULL
    else
      as.integer(d)
  })
}

## And this one, I guess:

#' Title
#'
#' @param x
#'
#' @return
#' @export
#'
#' @examples
as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}

