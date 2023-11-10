#' Title
#'
#' @param ...
#' @param mode
#' @param arch
#' @param input_channels
#' @param filters_gen
#' @param filters_disc
#' @param noise_channels
#' @param latent_variables
#' @param padding
#' @param kl_weight
#' @param ensemble_size
#' @param ensemble_members
#' @param content_loss_weight
#' @param lr_disc
#' @param lr_gen
#'
#' @return
#' @export
#'
#' @examples
setup_model <- function(...,
                        mode = mode,
                        arch = NULL,
                        input_channels = NULL,
                        filters_gen = NULL,
                        filters_disc = NULL,
                        noise_channels = NULL,
                        latent_variables = NULL,
                        padding = NULL,
                        kl_weight = NULL,
                        ensemble_size = NULL,
                        ensemble_members = NULL,
                        content_loss_weight = NULL,
                        lr_disc = NULL,
                        lr_gen = NULL){

  gen_to_use = list(ctrl  = generator,
                    mixed = generator,
                    ens   = generator_ens)[[arch]]
  disc_to_use = list(ctrl  = discriminator,
                     mixed = discriminator,
                     ens   = discriminator_ens)[[arch]]

  gen = gen_to_use(mode           = mode,
                   arch           = arch,
                   input_channels = input_channels,
                   noise_channels = noise_channels,
                   filters_gen    = filters_gen,
                   ensemble_members = ensemble_members,
                   padding        = padding)
  disc = disc_to_use(arch         = arch,
                     input_channels = input_channels,
                     filters_disc   = filters_disc,
                     ensemble_members = ensemble_members,
                     padding        = padding)
  model = initialize_wgangp(gen, disc, mode, arch, lr_disc = lr_disc, lr_gen = lr_gen,
                            ensemble_size = ensemble_size,
                            content_loss_weight = content_loss_weight)

  gc()
  return(model)
}
