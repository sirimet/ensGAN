#' Title
#'
#' @param mode
#' @param arch
#' @param input_channels
#' @param ensemble_members
#' @param latent_variables
#' @param noise_channels
#' @param filters_gen
#' @param img_shape
#' @param constant_fields
#' @param conv_size
#' @param padding
#' @param stride
#' @param relu_alpha
#' @param norm
#' @param dropout_rate
#'
#' @return
#' @export
#'
#' @examples
generator_ens <- function(mode,
                      arch,
                      input_channels = 9,
                      ensemble_members = 2,
                      latent_variables = 1,
                      noise_channels = 8,
                      filters_gen = 64,
                      img_shape = c(128, 128),
                      constant_fields = 3,
                      conv_size = c(3, 3),
                      padding = NULL,
                      stride = 1,
                      relu_alpha = 0.2,
                      norm = NULL,
                      dropout_rate = NULL){

  if(arch == "forceconv"){
    forceconv <- TRUE
  }else{
    forceconv <- FALSE
  }

  generator_input <- layer_input(shape=list(NULL, NULL, ensemble_members, input_channels), name="lo_res_inputs")

  ## Learning some linear relationships?
  reshaped_generator_input <- generator_input %>%
    layer_conv_3d(filters = input_channels, kernel_size=c(3, 3, 1), strides = 1, padding = "same") %>%
    layer_conv_3d(filters = input_channels, kernel_size=c(3, 3, 1), strides = 1, padding = "same") %>%
    layer_average_pooling_3d(pool_size = c(1, 1, ensemble_members))

  reshaped_generator_input <- tf$squeeze(reshaped_generator_input, 3)

  # constant fields
  const_input <- layer_input(shape=list(NULL, NULL, constant_fields), name="hi_res_inputs")

  # Convolve constant fields down to match other input dimensions
  upscaled_const_input <- layer_const_upscale_block_512(const_input, filters=filters_gen)

  noise_input <- layer_input(shape=list(NULL, NULL, noise_channels), name="noise_input")

  # Concatenate all inputs together
  generator_output <- layer_concatenate(list(reshaped_generator_input, upscaled_const_input, noise_input)) %>%
    # Pass through 3 residual blocks
    layer_residual_block(filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv) %>%
    layer_residual_block(filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv) %>%
    layer_residual_block(filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)

  # Upsampling from (10,10) to (100,100) with alternating residual blocks
  block_channels <- c(2*filters_gen, filters_gen)
  generator_output <- generator_output %>%
    layer_upsampling_2d(size=c(4, 4), interpolation='bilinear') %>%
    layer_residual_block(filters=block_channels[1], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv) %>%
    layer_upsampling_2d(size=c(2, 2), interpolation='bilinear') %>%
    layer_residual_block(filters=block_channels[2], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)

  # Concatenate with original size constants field
  generator_output = layer_concatenate(c(generator_output, const_input))

  # Pass through 3 residual blocks
  generator_output <- generator_output %>%
    layer_residual_block(filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv) %>%
    layer_residual_block(filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv) %>%
    layer_residual_block(filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv) %>%
    # Output layer
    layer_conv_2d(filters=1, kernel_size=c(1, 1), activation='softplus', name="output")

    model <- keras_model(inputs = c(generator_input, const_input, noise_input), outputs = generator_output)

  return(model)
}

#' Title
#'
#' @param arch
#' @param input_channels
#' @param ensemble_members
#' @param constant_fields
#' @param filters_disc
#' @param conv_size
#' @param padding
#' @param stride
#' @param relu_alpha
#' @param norm
#' @param dropout_rate
#'
#' @return
#' @export
#'
#' @examples
discriminator_ens <- function(arch,
                          input_channels = 9,
                          ensemble_members = 2,
                          constant_fields = 3,
                          filters_disc = 64,
                          conv_size = c(3, 3),
                          padding = "valid",
                          stride=1,
                          relu_alpha=0.2,
                          norm=NULL,
                          dropout_rate=NULL){

  if(arch == "forceconv"){
    forceconv <- TRUE
  }else{
    forceconv <- FALSE
  }

  # Network inputs
  # low resolution condition
  generator_input <- layer_input(shape = list(NULL, NULL, ensemble_members, input_channels), name = "lo_res_inputs")

  ## Learning some linear relationships?
  reshaped_generator_input <- generator_input %>%
    layer_conv_3d(filters = input_channels, kernel_size=c(3, 3, 1), strides = 1, padding = "same") %>%
    layer_conv_3d(filters = input_channels, kernel_size=c(3, 3, 1), strides = 1, padding = "same") %>%
    layer_average_pooling_3d(pool_size = c(1, 1, ensemble_members)) ## dim out: c(NULL, NULL, 1, channels)

  reshaped_generator_input <- tf$squeeze(reshaped_generator_input, 3) ## dim out: c(NULL, NULL, ensemble_members)

  # constant fields
  const_input <- layer_input(shape = list(NULL, NULL, constant_fields), name = "hi_res_inputs")
  # target image
  generator_output <- layer_input(shape = list(NULL, NULL, 1), name = "output")

  # convolve down constant fields to match ERA
  lo_res_const_input <- layer_const_upscale_block_512(const_input, filters=filters_disc)

  # concatenate constants to lo-res input
  lo_res_input <- layer_concatenate(c(reshaped_generator_input, lo_res_const_input))

  # concatenate constants to hi-res input
  hi_res_input <- layer_concatenate(c(generator_output, const_input))

  # encode inputs using residual blocks
  block_channels <- c(filters_disc, 2*filters_disc)
  # run through one set of RBs
  lo_res_input <- lo_res_input %>%
    layer_residual_block(filters=block_channels[1], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)

  hi_res_input <- hi_res_input %>%
    layer_conv_2d(filters = block_channels[1], kernel_size = c(4, 4), strides = 4, padding="valid", activation="relu") %>%
    layer_residual_block(filters=block_channels[1], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)

  # run through second set of RBs
  lo_res_input <- lo_res_input %>%
    layer_residual_block(filters=block_channels[2], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)

  hi_res_input <- hi_res_input %>%
    layer_conv_2d(filters = block_channels[2], kernel_size = c(2, 2), strides = 2, padding = "valid", activation = "relu") %>%
    layer_residual_block(filters = block_channels[2], conv_size = conv_size, stride = stride, relu_alpha = relu_alpha, norm = norm, dropout_rate = dropout_rate, padding = padding, force_1d_conv = forceconv)

  # concatenate hi- and lo-res inputs channel-wise before passing through discriminator
  disc_input <- layer_concatenate(c(lo_res_input, hi_res_input)) %>%
    # encode in residual blocks
    layer_residual_block(filters=filters_disc, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv) %>%
    layer_residual_block(filters=filters_disc, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)

  # discriminator output
  disc_output <- disc_input %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(64, activation='relu') %>%
    layer_dense(1, name="disc_output")


  disc = keras_model(inputs = c(generator_input, const_input, generator_output), outputs = disc_output)

  return(disc)
}
