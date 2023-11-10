#' Title
#'
#' @param value
#'
#' @return
#' @export
#'
#' @examples
bytes_feature <- function(value){
  ### Returns a bytes_list from a string / byte.
  value <- value$numpy()
  return(tf$train$Feature(bytes_list = tf$train$BytesList(value = list(value))))
}


#' Title
#'
#' @param list_of_floats
#'
#' @return
#' @export
#'
#' @examples
float_feature <- function(list_of_floats){  # float32
  return(tf$train$Feature(float_list = tf$train$FloatList(value = list_of_floats)))
}

#' Title
#'
#' @param list_of_int64s
#'
#' @return
#' @export
#'
#' @examples
int64_feature <- function(list_of_int64s){  # float32
  return(tf$train$Feature(int64_list = tf$train$Int64List(value = list_of_int64s)))
}

#' Title
#'
#' @param array
#'
#' @return
#' @export
#'
#' @examples
serialize_array <- function(array){
  array <- tf$io$serialize_tensor(array)
  return(array)
}

#' Title
#'
#' @param hi_res_input
#' @param const
#' @param lo_res_input
#'
#' @return
#' @export
#'
#' @examples
parse_single_instance <- function(hi_res_input, const, lo_res_input){

  #define the dictionary -- the structure -- of our single example
  data = list(
    h1 = int64_feature(list(hi_res_input$shape[[2]])),
    w1 = int64_feature(list(hi_res_input$shape[[3]])),
    d1 = int64_feature(list(hi_res_input$shape[[4]])),
    hi_res_image = bytes_feature(serialize_array(hi_res_input)),

    h2 = int64_feature(list(const$shape[[2]])),
    w2 = int64_feature(list(const$shape[[3]])),
    d2 = int64_feature(list(const$shape[[4]])),
    const_image = bytes_feature(serialize_array(const)),

    h3 = int64_feature(list(lo_res_input$shape[[2]])),
    w3 = int64_feature(list(lo_res_input$shape[[3]])),
    d3 = int64_feature(list(lo_res_input$shape[[4]])),
    lo_res_image = bytes_feature(serialize_array(lo_res_input))

  )
  #create an Example, wrapping the single features
  out = tf$train$Example(features=tf$train$Features(feature=data))

  return(out)

}

#' Title
#'
#' @param hi_res_input
#' @param const
#' @param lo_res_input
#'
#' @return
#' @export
#'
#' @examples
parse_single_instance2 <- function(hi_res_input, const, lo_res_input){

  #define the dictionary -- the structure -- of our single example
  data = list(
    hi_res_shape = bytes_feature(serialize_array(tf$cast(array(c(hi_res_input$shape[[2]], hi_res_input$shape[[3]], hi_res_input$shape[[4]]), dim = c(3)), "int64"))),
    hi_res_image = bytes_feature(serialize_array(hi_res_input)),

    const_shape = bytes_feature(serialize_array(tf$cast(array(c(const$shape[[2]], const$shape[[3]], const$shape[[4]]), dim = c(3)), "int64"))),
    const_image = bytes_feature(serialize_array(const)),

    lo_res_shape = bytes_feature(serialize_array(tf$cast(array(c(lo_res_input$shape[[2]], lo_res_input$shape[[3]], lo_res_input$shape[[4]]), dim = c(3)), "int64"))),
    lo_res_image = bytes_feature(serialize_array(lo_res_input))

  )
  #create an Example, wrapping the single features
  out = tf$train$Example(features=tf$train$Features(feature=data))

  return(out)

}

#' Title
#'
#' @param hi_res_input
#' @param const
#' @param lo_res_input
#'
#' @return
#' @export
#'
#' @examples
parse_single_instance_ens <- function(hi_res_input, const, lo_res_input){

  #define the dictionary -- the structure -- of our single example
  data = list(
    h1 = int64_feature(list(hi_res_input$shape[[2]])),
    w1 = int64_feature(list(hi_res_input$shape[[3]])),
    d1 = int64_feature(list(hi_res_input$shape[[4]])),
    hi_res_image = bytes_feature(serialize_array(hi_res_input)),

    h2 = int64_feature(list(const$shape[[2]])),
    w2 = int64_feature(list(const$shape[[3]])),
    d2 = int64_feature(list(const$shape[[4]])),
    const_image = bytes_feature(serialize_array(const)),

    h3 = int64_feature(list(lo_res_input$shape[[2]])),
    w3 = int64_feature(list(lo_res_input$shape[[3]])),
    d3 = int64_feature(list(lo_res_input$shape[[4]])),
    ensmem = int64_feature(list(lo_res_input$shape[[5]])),
    lo_res_image = bytes_feature(serialize_array(lo_res_input))

  )
  #create an Example, wrapping the single features
  out = tf$train$Example(features=tf$train$Features(feature=data))

  return(out)

}

#' Title
#'
#' @param hi_res_inputs
#' @param consts
#' @param lo_res_inputs
#' @param filename
#' @param records_folder
#'
#' @return
#' @export
#'
#' @examples
write_images_to_tfr_short <- function(hi_res_inputs, consts, lo_res_inputs, filename = "images",
                                      records_folder = getwd()){
  ## Write files, no classes

  filename <- paste0(records_folder, "/", filename, ".tfrecords")
  writer   <- tf$io$TFRecordWriter(filename) #create a writer that'll store our data to disk
  count    <- 0

  for(index in 1:dim(hi_res_inputs)[1]){
    #get the data we want to write
    current_hi_res_input = hi_res_inputs[index, , , , drop = FALSE]
    current_const = consts[index, , , , drop = FALSE]
    current_lo_res_input = lo_res_inputs[index, , , , drop = FALSE]

    out = parse_single_instance(hi_res_input = current_hi_res_input, const = current_const, lo_res_input = current_lo_res_input)

    writer$write(out$SerializeToString())
    count <- count + 1
  }

  writer$close()
  cat(paste0("Wrote ", count, " elements to TFRecord"))
  return(count)
}

#' Title
#'
#' @param hi_res_inputs
#' @param consts
#' @param lo_res_inputs
#' @param filename
#' @param num_class
#' @param records_folder
#'
#' @return
#' @export
#'
#' @examples
write_images_to_tfr_short2 <- function(hi_res_inputs, consts, lo_res_inputs, filename = "images", num_class = 4,
                                       records_folder = getwd()){
  # Write files with class divisions
  fle_hdles <- list()
  for(fh in 1:num_class){
    flename = paste0(records_folder, "/", filename, "_", fh, ".tfrecords")
    fle_hdles[[fh]] <- tf$io$TFRecordWriter(flename)
  }

  count    <- 0

  for(index in 1:dim(hi_res_inputs)[1]){
    #get the data we want to write
    current_hi_res_input = hi_res_inputs[index, , , , drop = FALSE]
    current_const = consts[index, , , , drop = FALSE]
    current_lo_res_input = lo_res_inputs[index, , , , drop = FALSE]

    out = parse_single_instance2(hi_res_input = current_hi_res_input, const = current_const, lo_res_input = current_lo_res_input)

    clss = pmin(floor(1 + (mean(current_hi_res_input$numpy() > 0.1)*num_class)), num_class)  # all class binning is here!
    fle_hdles[[clss]]$write(out$SerializeToString())

    count <- count + 1
  }

  for(fh in fle_hdles){
    fh$close()
  }
  cat(paste0("Wrote ", count, " elements to TFRecord"))
  return(count)
}


#' Title
#'
#' @param hi_res_inputs
#' @param consts
#' @param lo_res_inputs
#' @param filename
#' @param records_folder
#'
#' @return
#' @export
#'
#' @examples
write_images_to_tfr_short_ens <- function(hi_res_inputs, consts, lo_res_inputs, filename = "images",
                                          records_folder = getwd()){
  # Write ensemble data, no classes

  filename <- paste0(records_folder, "/", filename, ".tfrecords")
  writer   <- tf$io$TFRecordWriter(filename) #create a writer that'll store our data to disk
  count    <- 0

  for(index in 1:dim(hi_res_inputs)[1]){
    #get the data we want to write
    current_hi_res_input = hi_res_inputs[index, , , , drop = FALSE]
    current_const = consts[index, , , , drop = FALSE]
    current_lo_res_input = lo_res_inputs[index, , , , , drop = FALSE]

    out = parse_single_instance_ens(hi_res_input = current_hi_res_input, const = current_const, lo_res_input = current_lo_res_input)

    writer$write(out$SerializeToString())
    count <- count + 1
  }

  writer$close()
  cat(paste0("Wrote ", count, " elements to TFRecord"))
  return(count)
}



#' Title
#'
#' @param hi_res_inputs
#' @param consts
#' @param lo_res_inputs
#' @param filename
#' @param num_class
#' @param records_folder
#'
#' @return
#' @export
#'
#' @examples
write_images_to_tfr_short_ens2 <- function(hi_res_inputs, consts, lo_res_inputs, filename = "images", num_class = 4,
                                           records_folder = getwd()){

  # Write ensemble data, classes
  fle_hdles <- list()
  for(fh in 1:num_class){
    flename = paste0(records_folder, "/", filename, "_", fh, ".tfrecords")
    fle_hdles[[fh]] <- tf$io$TFRecordWriter(flename)
  }

  count    <- 0

  for(index in 1:dim(hi_res_inputs)[1]){
    #get the data we want to write
    current_hi_res_input = hi_res_inputs[index, , , , drop = FALSE]
    current_const = consts[index, , , , drop = FALSE]
    current_lo_res_input = lo_res_inputs[index, , , , drop = FALSE]

    out = parse_single_instance_ens(hi_res_input = current_hi_res_input, const = current_const, lo_res_input = current_lo_res_input)

    clss = pmin(floor(1 + (mean(current_hi_res_input$numpy() > 0.1)*num_class)), num_class)  # all class binning is here!
    fle_hdles[[clss]]$write(out$SerializeToString())

    count <- count + 1
  }

  # writer$close()
  for(fh in fle_hdles){
    fh$close()
  }
  cat(paste0("Wrote ", count, " elements to TFRecord"))
  return(count)
}



#' Title
#'
#' @param element
#'
#' @return
#' @export
#'
#' @examples
parse_tfr_element <- function(element){
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = list(
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
    lo_res_image = tf$io$FixedLenFeature(shape(), tf$string)
  )

  content = tf$io$parse_single_example(element, data)

  h1 = content[['h1']]
  w1 = content[['w1']]
  d1 = content[['d1']]

  h2 = content[['h2']]
  w2 = content[['w2']]
  d2 = content[['d2']]

  h3 = content[['h3']]
  w3 = content[['w3']]
  d3 = content[['d3']]

  hi_res_image = content[['hi_res_image']]
  const_image = content[['const_image']]
  lo_res_image = content[['lo_res_image']]

  #get our 'feature'-- our image -- and reshape it appropriately
  feature1 = tf$io$parse_tensor(hi_res_image, out_type=tf$float32)
  feature1 = tf$reshape(feature1, shape = list(h1, w1, d1))

  feature2 = tf$io$parse_tensor(const_image, out_type=tf$float32)
  feature2 = tf$reshape(feature2, shape = list(h2, w2, d2))

  feature3 = tf$io$parse_tensor(lo_res_image, out_type = tf$float32)
  feature3 = tf$reshape(feature3, shape = list(h3, w3, d3))

  return (list(feature1, feature2, feature3))

}


#' Title
#'
#' @param element
#'
#' @return
#' @export
#'
#' @examples
parse_tfr_element2 <- function(element){
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = list(
    hi_res_shape = tf$io$FixedLenFeature(shape(), tf$string),
    hi_res_image = tf$io$FixedLenFeature(shape(), tf$string),

    const_shape = tf$io$FixedLenFeature(shape(), tf$string),
    const_image = tf$io$FixedLenFeature(shape(), tf$string),

    lo_res_shape = tf$io$FixedLenFeature(shape(), tf$string),
    lo_res_image = tf$io$FixedLenFeature(shape(), tf$string)
  )

  content = tf$io$parse_single_example(element, data)

  hi_res_shape = content[['hi_res_shape']]
  hi_res_shape = tf$io$parse_tensor(hi_res_shape, out_type=tf$int64)
  const_shape  = content[['const_shape']]
  const_shape = tf$io$parse_tensor(const_shape, out_type=tf$int64)
  lo_res_shape = content[['lo_res_shape']]
  lo_res_shape = tf$io$parse_tensor(lo_res_shape, out_type=tf$int64)

  hi_res_image = content[['hi_res_image']]
  const_image  = content[['const_image']]
  lo_res_image = content[['lo_res_image']]

  #get our 'feature'-- our image -- and reshape it appropriately
  feature1 = tf$io$parse_tensor(hi_res_image, out_type=tf$float32)
  feature1 = tf$reshape(feature1, shape = hi_res_shape) #tf$cast(hi_res_shape, "int64")

  feature2 = tf$io$parse_tensor(const_image, out_type=tf$float32)
  feature2 = tf$reshape(feature2, shape = const_shape)

  feature3 = tf$io$parse_tensor(lo_res_image, out_type = tf$float32)
  feature3 = tf$reshape(feature3, shape = lo_res_shape)

  return (list(feature1, feature2, feature3))

}


#' Title
#'
#' @param element
#'
#' @return
#' @export
#'
#' @examples
parse_tfr_element_ens <- function(element){
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = list(
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

  content = tf$io$parse_single_example(element, data)

  h1 = content[['h1']]
  w1 = content[['w1']]
  d1 = content[['d1']]

  h2 = content[['h2']]
  w2 = content[['w2']]
  d2 = content[['d2']]

  h3 = content[['h3']]
  w3 = content[['w3']]
  d3 = content[['d3']]
  ensmem = content[['ensmem']]

  hi_res_image = content[['hi_res_image']]
  const_image = content[['const_image']]
  lo_res_image = content[['lo_res_image']]

  #get our 'feature'-- our image -- and reshape it appropriately
  feature1 = tf$io$parse_tensor(hi_res_image, out_type=tf$float32)
  feature1 = tf$reshape(feature1, shape = list(h1, w1, d1))

  feature2 = tf$io$parse_tensor(const_image, out_type=tf$float32)
  feature2 = tf$reshape(feature2, shape = list(h2, w2, d2))

  feature3 = tf$io$parse_tensor(lo_res_image, out_type = tf$float32)
  feature3 = tf$reshape(feature3, shape = list(h3, w3, d3, ensmem))

  return (list(feature1, feature2, feature3))

}


#' Title
#'
#' @param filename
#'
#' @return
#' @export
#'
#' @examples
get_dataset_small <- function(filename){
  #create the dataset
  dataset <- tfdatasets::tfrecord_dataset(filename)

  #pass every single feature through our mapping function
  dataset <- dataset %>% tfdatasets::dataset_map(
    parse_tfr_element2
  ) %>%
    dataset_shuffle(5) %>%
    dataset_batch(batch_size, drop_remainder = TRUE) %>%
    dataset_prefetch(1)

  return(dataset)

}

#' Title
#'
#' @param filename
#'
#' @return
#' @export
#'
#' @examples
get_dataset_small_ens <- function(filename){
  #create the dataset
  dataset <- tfdatasets::tfrecord_dataset(filename)

  #pass every single feature through our mapping function
  dataset <- dataset %>% tfdatasets::dataset_map(
    parse_tfr_element_ens
  ) %>%
    # dataset_repeat() %>%
    dataset_shuffle(5) %>%
    dataset_batch(batch_size, drop_remainder = TRUE) %>%
    dataset_prefetch(1)

  return(dataset)

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
parse_batch <- function(record_batch, insize = c(512,512,1), consize = c(512,512,2),
                        outsize = c(64,64,9), return_dic = FALSE){
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


