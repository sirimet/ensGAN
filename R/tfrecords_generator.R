#' Title
#'
#' @param year
#' @param batch_size
#' @param repeated
#' @param weight
#'
#' @return
#' @export
#'
#' @examples
data_generator <- function(year, batch_size, repeated = TRUE, weight = NULL, records_folder = records_folder){
  return(create_mixed_dataset(year, batch_size, repeated = repeated, weight = weight, records_folder = records_folder))
}

#' Title
#'
#' @param year
#' @param batch_size
#' @param era_shape
#' @param con_shape
#' @param out_shape
#' @param repeated
#' @param folder
#' @param shuffle_size
#' @param weight
#'
#' @return
#' @export
#'
#' @examples
create_mixed_dataset <- function(year, batch_size, era_shape = c(16,16,9), con_shape = c(128,128,3),
                                 out_shape = c(128,128,1), repeated = TRUE,
                                 records_folder = records_folder, shuffle_size = 1024,
                                 weight=NULL){
  return_dic <- FALSE
  batch_size <- as.integer(batch_size)
  shuffle_size <- as.integer(shuffle_size)

  classes <- 4
  if(is.null(weight)){
    weight <- as.list(rep(1/classes, classes))
  }

  datasets <- lapply(1:classes, function(i) create_dataset(year, i, records_folder = records_folder, shuffle_size = shuffle_size,
                                                           repeated = repeated))

  sampled_ds = tf$data$Dataset$sample_from_datasets(datasets, weights = weight)$batch(batch_size)

  sampled_ds = sampled_ds$prefetch(tf$cast(2, "int64"))
  return(sampled_ds)

}


#' Title
#'
#' @param inputs
#' @param outputs
#'
#' @return
#' @export
#'
#' @examples
dataset_downsampler <- function(inputs, outputs){
  image <- outputs[['output']]
  kernel_tf <- tf$constant(0.01, shape = c(8, 8, 1, 1), dtype = tf$float32)
  image <- tf$nn$conv2d(image, filters = kernel_tf, strides = c(1, 8, 8, 1), padding = 'VALID',
                        name = 'conv_debug', data_format = 'NHWC')
  inputs[['lo_res_inputs']] <- image
  return(list(inputs, outputs))
}


#' Title
#'
#' @param record_batch
#' @param insize
#' @param consize
#' @param outsize
#'
#' @return
#' @export
#'
#' @examples
parse_batch <- function(record_batch, insize = c(128,128,1), consize = c(128,128,3),
                        outsize = c(16,16,9)){
  return_dic <- FALSE
  # Create a description of the features
  feature_description = list(
    hi_res_shape = tf$io$FixedLenFeature(shape(), tf$string),
    hi_res_image = tf$io$FixedLenFeature(shape(), tf$string),

    const_shape = tf$io$FixedLenFeature(shape(), tf$string),
    const_image = tf$io$FixedLenFeature(shape(), tf$string),

    lo_res_shape = tf$io$FixedLenFeature(shape(), tf$string),
    lo_res_image = tf$io$FixedLenFeature(shape(), tf$string)
  )

  # Parse the input `tf.Example` proto using the dictionary above
  example = tf$io$parse_example(record_batch, feature_description)

  hi_res_shape = example[['hi_res_shape']]
  const_shape  = example[['const_shape']]
  lo_res_shape = example[['lo_res_shape']]

  hi_res_image = example[['hi_res_image']]
  const_image  = example[['const_image']]
  lo_res_image = example[['lo_res_image']]

  #get our 'feature'-- our image -- and reshape it appropriately
  hi_res_shape = tf$io$parse_tensor(hi_res_shape, out_type=tf$int64)
  feature1 = tf$io$parse_tensor(hi_res_image, out_type=tf$float32)
  feature1 = tf$reshape(feature1, shape = hi_res_shape)

  const_shape = tf$io$parse_tensor(const_shape, out_type=tf$int64)
  feature2 = tf$io$parse_tensor(const_image, out_type=tf$float32)
  feature2 = tf$reshape(feature2, shape = const_shape)

  lo_res_shape = tf$io$parse_tensor(lo_res_shape, out_type=tf$int64)
  feature3 = tf$io$parse_tensor(lo_res_image, out_type = tf$float32)
  feature3 = tf$reshape(feature3, shape = lo_res_shape)


  if(return_dic){
    return (list(list(lo_res_inputs = feature3,
                      hi_res_inputs = feature2),
                 list(output = feature1)))
  }else{
    return(list(feature3, feature2, feature1))
  }
}

#' Title
#'
#' @param record_batch
#' @param insize
#' @param consize
#' @param outsize
#' @param return_dic
#'
#' @return
#' @export
#'
#' @examples
parse_batch_ens <- function(record_batch, insize = c(512,512,1), consize = c(512,512,2),
                            outsize = c(64,64,9), return_dic = FALSE){
  # Create a description of the features
  feature_description = list(
    h1 = tf$io$FixedLenFeature(shape(), tf$int64),
    w1 = tf$io$FixedLenFeature(shape(), tf$int64),
    d1 = tf$io$FixedLenFeature(shape(), tf$int64),
    hi_res_image = tf$io$FixedLenFeature(shape(), tf$string),

    h2 = tf$io$FixedLenFeature(shape(), tf$int64),
    w2 = tf$io$FixedLenFeature(shape(), tf$int64),
    d2 = tf$io$FixedLenFeature(shape(), tf$int64),
    const_image = tf$io$FixedLenFeature(shape(), tf$string),

    h3 = tf$io$FixedLenFeature(shape(), tf$int64),
    w3 = tf$io$FixedLenFeature(shape(), tf$int64),
    d3 = tf$io$FixedLenFeature(shape(), tf$int64),
    ensmem = tf$io$FixedLenFeature(shape(), tf$int64),
    lo_res_image = tf$io$FixedLenFeature(shape(), tf$string)
  )

  # Parse the input `tf.Example` proto using the dictionary above
  example = tf$io$parse_example(record_batch, feature_description)

  h1 = example[['h1']]
  w1 = example[['w1']]
  d1 = example[['d1']]

  h2 = example[['h2']]
  w2 = example[['w2']]
  d2 = example[['d2']]

  h3 = example[['h3']]
  w3 = example[['w3']]
  d3 = example[['d3']]
  ensmem = example[['ensmem']]

  hi_res_image = example[['hi_res_image']]
  const_image  = example[['const_image']]
  lo_res_image = example[['lo_res_image']]

  feature1 = tf$io$parse_tensor(hi_res_image, out_type=tf$float32)
  feature1 = tf$reshape(feature1, shape = list(h1, w1, d1))

  feature2 = tf$io$parse_tensor(const_image, out_type=tf$float32)
  feature2 = tf$reshape(feature2, shape = list(h2, w2, d2))

  feature3 = tf$io$parse_tensor(lo_res_image, out_type = tf$float32)
  feature3 = tf$reshape(feature3, shape = list(h3, w3, d3, ensmem))

  if(return_dic){
    return (list(list(lo_res_inputs = feature3,
                      hi_res_inputs = feature2),
                 list(output = feature1)))
  }else{
    return(list(feature3, feature2, feature1))
  }
}

#' Title
#'
#' @param year
#' @param clss
#' @param era_shape
#' @param con_shape
#' @param out_shape
#' @param folder
#' @param shuffle_size
#' @param repeated
#'
#' @return
#' @export
#'
#' @examples
create_dataset <- function(year, clss, era_shape = c(16,16,11,9), con_shape = c(128,128,3), out_shape = c(128,128,1),
                           records_folder = records_folder, shuffle_size = 1024, repeated = TRUE){
  shuffle_size <- tf$cast(shuffle_size, "int64")
  AUTOTUNE  <- tf$data$experimental$AUTOTUNE
  # I did it - myyyyy way.
  fl       <- unlist(lapply(year, function(x) dir(path = records_folder, pattern = paste0(x, "_", clss, ".tfrecords"), full.names = TRUE)))
  files_ds <- tf$data$Dataset$list_files(fl)
  ds       <- tf$data$TFRecordDataset(files_ds,
                                      num_parallel_reads = AUTOTUNE)
  ds       <- ds$shuffle(shuffle_size)
  if(!data_type == "mixed"){
    ds       <- ds$map(function(x) parse_batch_ens(x, insize = out_shape, consize = con_shape,
                                                   outsize = era_shape))
  } else{
    ds       <- ds$map(function(x) parse_batch(x, insize = out_shape, consize = con_shape,
                                               outsize = era_shape))
  }
  if(repeated){
    return(ds[["repeat"]]())
  }
  else{
    return(ds)
  }
}

#' Title
#'
#' @param year
#' @param mode
#' @param batch_size
#' @param era_shape
#' @param con_shape
#' @param out_shape
#' @param name
#' @param folder
#'
#' @return
#' @export
#'
#' @examples
create_fixed_dataset <- function(year = NULL, mode = 'validation', batch_size = 16,
                                era_shape = c(16,16,9), con_shape = c(128,128,3),
                                out_shape = c(128,128,1), name = NULL, records_folder = records_folder){
  batch_size = tf$cast(batch_size, "int64")
  return_dic <- FALSE

  if(is.null(year) & is.null(name)){
    stop("Must specify year or file name")
  } else{
    if(is.null(name)){
      name <- paste0(year, ".tfrecords")
    }
  }

  fl        <- unlist(lapply(name, function(x) dir(path = records_folder, pattern = x, full.names = TRUE)))
  files_ds <- tf$data$Dataset$list_files(fl)
  ds       <- tf$data$TFRecordDataset(files_ds,
                                      num_parallel_reads = as.integer(1))
  if(!data_type == "mixed"){
    ds       <- ds$map(function(x) parse_batch_ens(x, insize = era_shape, consize = con_shape,
                                                   outsize = out_shape))
  } else{
    ds       <- ds$map(function(x) parse_batch(x, insize = era_shape, consize = con_shape,
                                               outsize = out_shape))
  }
  ds       <- ds$batch(batch_size)

  return(ds)
}

