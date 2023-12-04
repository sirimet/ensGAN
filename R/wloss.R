# wloss.py

wasserstein_loss <- function(y_true, y_pred){
  return(k_mean(y_true * y_pred, axis=-1))
}
