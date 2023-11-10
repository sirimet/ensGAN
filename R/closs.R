#' Custom Loss
#'
#' @param y_true
#' @param y_pred
#' @param threshold
#'
#' @return
#' @export
#'
#' @examples
custom_loss <- function(y_true, y_pred, threshold = 2.087243){
  # calculate the normal loss using mean squared error
  normal_loss <- k_mean((y_true - y_pred)^2, axis = -1)
  threshold   <- array(threshold, dim = dim(y_pred))
  # calculate the penalty for values above the threshold
  err <- y_pred - threshold

  abs_err <- tf$abs(err)
  double_err <- tf$add(err, abs_err)
  penalty <- tf$divide(double_err, 2)

  # calculate the final loss as the sum of the normal loss and penalty
  loss <- normal_loss + penalty
  return(loss)
}
