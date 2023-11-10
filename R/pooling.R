#' Pooling
#'
#' @param x
#' @param pool_type
#' @param data_format
#'
#' @return
#' @export
#'
#' @examples
pool <- function(x, pool_type, data_format='channels_last'){
  # """Apply pooling operation (via Tensorflow) to input Numpy array x.
  #   x should be 4-dimensional: N x W x H x C ('channels_last') or N x C x W x H ('channels_first')
  #   Pooling is applied on W and H dimensions.
  #
  #   """
  pool_op = list(
    max_4 = layer_max_pooling_2d(pool_size = c(4, 4), strides = c(2, 2), data_format = data_format),
    max_16 = layer_max_pooling_2d(pool_size = c(16, 16), strides = c(4, 4), data_format = data_format),
    max_8_no_overlap = layer_max_pooling_2d(pool_size = c(8, 8), strides = c(8, 8), data_format = data_format),
    max_10_no_overlap = layer_max_pooling_2d(pool_size = c(10, 10), strides = c(10, 10), data_format = data_format),
    avg_4 = layer_average_pooling_2d(pool_size = c(4, 4), strides = c(2, 2), data_format = data_format),
    avg_16 = layer_average_pooling_2d(pool_size = c(16, 16), strides = c(4, 4), data_format = data_format),
    avg_8_no_overlap = layer_average_pooling_2d(pool_size = c(8, 8), strides = c(8, 8), data_format = data_format),
    avg_10_no_overlap = layer_average_pooling_2d(pool_size = c(10, 10), strides = c(10, 10), data_format = data_format)
  )[pool_type]

  return(lapply(pool_op, function(f) f(x)$numpy()))
}
