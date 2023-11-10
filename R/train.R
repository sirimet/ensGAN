#' Train Model
#'
#' @param ...
#' @param model
#' @param mode
#' @param batch_gen_train
#' @param batch_gen_valid
#' @param noise_channels
#' @param checkpoint
#' @param steps_per_checkpoint
#'
#' @return
#' @export
#'
#' @examples
train_model <- function(...,
                        model = NULL,
                        mode = NULL,
                        batch_gen_train = NULL,
                        batch_gen_valid = NULL,
                        noise_channels = NULL,
                        checkpoint = NULL,
                        steps_per_checkpoint = NULL){

  dataset_iterator <- batch_gen_train$take(tf$cast(1, "int64")) %>% as_iterator()
  cond <- iter_next(dataset_iterator)
  if(length(cond) == 2){
    img_shape  = cond[[1]][[1]]$shape[2:(length(cond[[1]][[1]]$shape)-1)]
  } else{
    img_shape  = cond[[1]]$shape[2:(length(cond[[1]]$shape)-1)]
  }
  if(length(cond) == 2){
    batch_size = cond[[1]][[1]]$shape[1]
  } else{
    batch_size = cond[[1]]$shape[1]
  }
  rm(cond, dataset_iterator)

  noise_shape = c(img_shape[1], img_shape[2], noise_channels)
  noise_gen   = noise_generator(noise_shape, batch_size = batch_size)
  loss_log    = train(model, batch_gen_train, noise_gen,
                      steps_per_checkpoint, training_ratio = 1) #5)

  return(loss_log)
}
