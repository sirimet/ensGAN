#' Title
#'
#' @param x
#' @param filters
#' @param kernel_size
#' @param padding
#'
#' @return
#' @export
#'
#' @examples
layer_conv_2d_padding <- function(x, filters, kernel_size = c(3, 3), padding = "same"){
  if(length(kernel_size) == 1) kernel_size <- rep(kernel_size, 2)
  if(length(kernel_size)  > 2) kernel_size <- kernel_size[1:2]
  # cat("This function has been called!\n")
  if(padding %in% c("reflect", "symmetric")){
    # cat("The first if has been entered.\n")
    pad <- sapply(kernel_size, function(s) (s-1) %/% 2) # only works if s is odd!
    print(pad)
    if(padding == "reflect"){
      # cat("The second if has been entered.\n")
      x <- x %>%
        layer_reflection_padding_2d(pad_by = pad) %>%
        layer_conv_2d(filters, kernel_size, padding = "valid")
      return(x)
    }
    if(padding == "symmetric"){
      x <- x %>%
        layer_symmetric_padding_2d(pad_by = pad) %>%
        layer_conv_2d(filters, kernel_size, padding = "valid")
      return(x)
    }
  }else{
    x <- x %>%
      layer_conv_2d(filters, kernel_size, padding = padding)
    return(x)
  }
}

#' Title
#'
#' @param x
#' @param filters
#' @param conv_size
#' @param stride
#' @param relu_alpha
#' @param norm
#' @param dropout_rate
#' @param padding
#' @param force_1d_conv
#'
#' @return
#' @export
#'
#' @examples
layer_residual_block <- function(x, filters, conv_size=c(3, 3), stride=1, relu_alpha=0.2, norm = NULL
                                 , dropout_rate = NULL, padding = NULL, force_1d_conv=FALSE){
  # This next line is not ideal:
  in_channels <- as.integer(grep("[[:digit:]]", unlist(strsplit(gsub("[[:punct:]]", "", gsub("None",
                                                                                             "NULL", x$shape)), " ")), value = T))
  in_channels <- in_channels[length(in_channels)]

  x_in <- x %>%
    layer_average_pooling_2d(pool_size=c(stride, stride))

  if(force_1d_conv | (filters != in_channels)){
    x_in <- x_in %>%
      layer_conv_2d(filters=filters, kernel_size=c(1, 1))
  }

  # first block of activation and 3x3 convolution
  x <- x %>%
    layer_activation_leaky_relu(relu_alpha) %>%
    layer_conv_2d_padding(filters=filters, kernel_size=conv_size) #, padding=padding)
  if(is.null(norm)){
    # cat("Nothing to see here.")
  }else{
    if(norm == "batch"){
      x <- x %>%
        layer_batch_normalization()
    }
    # cat("Something is not implemented.")
  }


  if(!is.null(dropout_rate)){
    x <- x %>%
      layer_dropout(dropout_rate)
    print(paste0("Dropout rate is ", dropout_rate))
  }


  # second block of activation and 3x3 convolution
  x <- x %>%
    layer_activation_leaky_relu(relu_alpha) %>%
    layer_conv_2d_padding(filters=filters, kernel_size=conv_size) #, padding=padding)
  if(is.null(norm)){
    # cat("Nothing to see here.")
  }else{
    if(norm == "batch"){
      x <- x %>%
        layer_batch_normalization()
    }
    # cat("Something is not implemented.")
  }

  if(!is.null(dropout_rate)){
    x <- x %>%
      layer_dropout(dropout_rate)
    print(paste0("Dropout rate is ", dropout_rate))
  }

  # skip connection
  x <- layer_add(c(x, x_in))

  return(x)
}


#' Title
#'
#' @param const_input
#' @param filters
#'
#' @return
#' @export
#'
#' @examples
layer_const_upscale_block <- function(const_input, filters){
  # Map (n x 250 x 250 x 2) to (n x 10 x 10 x f)
  const_output <- const_input %>%
    layer_conv_2d(filters = filters, kernel_size = c(6, 6), strides=4, padding="valid") %>%
    layer_activation_relu() %>%
    #    SwarmActivation(base_activation = "relu") %>%
    layer_conv_2d(filters = filters, kernel_size = c(2, 2), strides=3, padding="valid") %>%
    layer_activation_relu() %>%
    #    SwarmActivation(base_activation = "relu") %>%
    layer_conv_2d(filters = filters, kernel_size = c(3, 3), strides=2, padding="valid") %>%
    layer_activation_relu()
  #    SwarmActivation(base_activation = "relu")
  return(const_output)
}

#' Title
#'
#' @param const_input
#' @param filters
#'
#' @return
#' @export
#'
#' @examples
layer_const_upscale_block_512 <- function(const_input, filters){
  # Map (n x 512 x 512 x 2) to (n x 64 x 64 x f)
  const_output <- const_input %>%
    layer_conv_2d(filters = filters, kernel_size = c(2, 2), strides=4, padding="valid") %>%
    layer_activation_relu() %>%
    #    SwarmActivation(base_activation = "relu") %>%
    layer_conv_2d(filters = filters, kernel_size = c(2, 2), strides=2, padding="valid") %>%
    layer_activation_relu()
  #    SwarmActivation(base_activation = "relu")
  return(const_output)
}

#' Title
#'
#' @param const_input
#' @param filters
#'
#' @return
#' @export
#'
#' @examples
layer_const_upscale_block_100 <- function(const_input, filters){
  # Map (n x 100 x 100 x 2) to (n x 10 x 10 x f)
  const_output <- const_input %>%
    layer_conv_2d(filters=filters, kernel_size=c(5, 5), strides=5, padding="valid") %>%
    layer_activation_relu() %>%
    layer_conv_2d(filters=filters, kernel_size=c(2, 2), strides=2, padding="valid") %>%
    layer_activation_relu()
  return(const_output)
}
