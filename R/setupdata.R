#' Title
#'
#' @param train_years
#' @param val_years
#' @param batch_size
#' @param val_size
#' @param weight
#' @param val_fixed
#' @param data_type
#'
#' @return
#' @export
#'
#' @examples
setup_batch_gen <- function(train_years,
                            val_years,
                            batch_size = 64,
                            val_size = NULL,
                            records_folder = records_folder,
                            weight = NULL,
                            val_fixed = TRUE,
                            data_type = data_type){

  return_dic <- FALSE

  if(is.null(train_years)){
    train <- NULL
  }else{
    train <- data_generator(train_years, batch_size = batch_size,
                            records_folder = records_folder,
                            weight = weight)
  }

  # note -- using create_fixed_dataset with a batch size not divisible by 16 will cause problems [is this true?]
  # create_fixed_dataset will not take a list
  if(!is.null(val_size)){
    # assume that val_size is small enough that we can just use one batch
    val = create_fixed_dataset(val_years, batch_size = val_size, records_folder = records_folder)# , downsample = downsample)
    val = val$take(1L)
    if(val_fixed){
      val = val$cache()
    }
  }else{
    val <- create_fixed_dataset(val_years, batch_size = batch_size, records_folder = records_folder) #, downsample = downsample)
  }
  return(list(train, val))

}



#' Title
#'
#' @param train_years
#' @param val_years
#' @param val_size
#' @param weight
#' @param batch_size
#' @param load_full_image
#' @param data_type
#'
#' @return
#' @export
#'
#' @examples
setup_data <- function(train_years = NULL,
                       val_years = NULL,
                       val_size = NULL,
                       records_folder = getwd(),
                       weight = NULL,
                       batch_size = NULL,
                       data_type = "ctrl"){

  # if(load_full_image){
  #   if(is.null(train_years)){
  #     batch_gen_train <- NULL
  #   }else{
  #     batch_gen_train <- setup_full_image_dataset(train_years,
  #                                                 batch_size = batch_size,
  #                                                 records_folder = records_folder)
  #   }
  #   if(is.null(val_years)){
  #     batch_gen_valid <- NULL
  #   }else{
  #     batch_gen_valid <- setup_full_image_dataset(val_years,
  #                                                 batch_size = batch_size,
  #                                                 records_folder = records_folder)
  #   }
  # }else{
    c(batch_gen_train, batch_gen_valid) %<-% setup_batch_gen(
      train_years = train_years,
      val_years   = val_years,
      batch_size  = batch_size,
      val_size    = val_size,
      records_folder = records_folder,
      weight      = weight,
      data_type   = data_type)
  # }
  gc()
  return(list(batch_gen_train, batch_gen_valid))
}

