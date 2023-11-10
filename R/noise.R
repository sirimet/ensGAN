#' Noise Generator
#'
#' @param noise_shapes
#' @param batch_size
#' @param random_seed
#' @param mean
#' @param std
#'
#' @return
#' @export
#'
#' @examples
noise_generator <- function(noise_shapes, batch_size = 32, random_seed = NULL, mean = 0, std = 1){
  if(!is.null(random_seed)) set.seed(random_seed)

  noise_shapes <- as.numeric(noise_shapes)
  shape <- as.numeric(c(batch_size, noise_shapes))
  n <- function() array(rnorm(prod(shape), mean = mean, sd = std), dim = shape)

  return(n)
}
