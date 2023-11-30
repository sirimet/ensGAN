## ADAPTED TO R FROM ORIGINAL PYTHON:
# # BSD 3-Clause License
#
# # Copyright (c) 2019, PySteps developers
# # All rights reserved.
#
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
#
# # * Redistributions of source code must retain the above copyright notice, this
# #   list of conditions and the following disclaimer.
#
# # * Redistributions in binary form must reproduce the above copyright notice,
# #   this list of conditions and the following disclaimer in the documentation
# #   and/or other materials provided with the distribution.
#
# # * Neither the name of the copyright holder nor the names of its
# #   contributors may be used to endorse or promote products derived from
# #   this software without specific prior written permission.
#
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#' Compute radially averaged power spectral density (RAPSD) from the given 2D input field
#'
#' @param field A 2d array of shape (m, n) containing the input field.
#' @param fft_method By default stats::fft
#' @param return_freq Whether to also return the Fourier frequencies. By default FALSE.
#' @param d Sample spacing (inverse of the sampling rate). Defaults to 1. Applicable if return_freq is TRUE.
#' @param normalize If TRUE, normalize the power spectrum so that it sums to one.
#' @param ...
#'
#' @return One-dimensional array containing the RAPSD. The length of the array is int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n). If return_freq, a one-dimensional array containing the Fourier frequencies is also returned.
#' @export
#'
#' @examples
rapsd <- function(field,
                  fft_method = NULL,
                  return_freq = FALSE,
                  d = 1.0,
                  normalize = FALSE,
                  ...){
  # """Compute radially averaged power spectral density (RAPSD) from the given
  #   2D input field.
  #   Parameters
  #   ----------
  #   field: array_like
  #       A 2d array of shape (m, n) containing the input field.
  #   fft_method: object
  #       A module or object implementing the same methods as numpy.fft and
  #       scipy.fftpack. If set to None, field is assumed to represent the
  #       shifted discrete Fourier transform of the input field, where the
  #       origin is at the center of the array
  #       (see numpy.fft.fftshift or scipy.fftpack.fftshift).
  #   return_freq: bool
  #       Whether to also return the Fourier frequencies.
  #   d: scalar
  #       Sample spacing (inverse of the sampling rate). Defaults to 1.
  #       Applicable if return_freq is 'True'.
  #   normalize: bool
  #       If True, normalize the power spectrum so that it sums to one.
  #   Returns
  #   -------
  #   out: ndarray
  #     One-dimensional array containing the RAPSD. The length of the array is
  #     int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
  #   freq: ndarray
  #     One-dimensional array containing the Fourier frequencies.
  #   References
  #   ----------
  #   :cite:`RC2011`
  #   """

  if(length(dim(field)) != 2){
    stop(paste0(length(dim(field)), "dimensions are found, but the number of dimensions should be 2"))
  }

  if(sum(is.nan(field)) > 0){ # np.sum(np.isnan(field)) > 0:
    stop("input field should not contain nans")
  }


  c(m, n) %<-% dim(field)

  c(yc, xc) %<-% compute_centred_coord_array(m, n)

  # r_grid = np.sqrt(xc * xc + yc * yc).round()
  aa <- matrix(xc * xc, nrow = ncol(xc), ncol = ncol(xc))
  bb <- matrix(yc * yc, nrow = nrow(yc), ncol = nrow(yc), byrow = T)
  r_grid <- round(sqrt(aa + bb))

  l = max(dim(field))

  if(l %% 2 == 1){
    # r_range = np.arange(0, int(l / 2) + 1)
    r_range <- 1:(round(l / 2) + 1)
  }else{
    # r_range = np.arange(0, int(l / 2))
    r_range <- 1:round(l / 2)
  }

  if(!is.null(fft_method)){
    psd <- fftshift(fft(field)) # , **fft_kwargs
    psd <- abs(psd) ** 2 / length(psd)
  }else{
    psd <- field
  }


  result <- list()
  for(r in r_range){
    mask <- r_grid == r
    psd_vals <- psd[mask]
    # result.append(np.mean(psd_vals))
    result <- c(result, mean(psd_vals))
  }

  result = array(unlist(result), dim = c(length(result), 1))

  if(normalize){
    result <- result / sum(result)
  }

  if(return_freq){
    freq = fftfreq(l, d = d)
    freq = freq[r_range]
    return(list(result = result, freq = freq))
  }else{
    return(result)
  }
}

#' Title
#'
#' @param fft_freq
#' @param fft_power
#' @param x_units
#' @param y_units
#' @param wavelength_ticks
#' @param color
#' @param lw
#' @param label
#' @param ax
#' @param ...
#'
#' @return
#' @export
#'
#' @examples
plot_spectrum1d <- function(
    fft_freq,
    fft_power,
    x_units = NULL,
    y_units = NULL,
    wavelength_ticks = NULL,
    color = "k",
    lw = 1.0,
    label = NULL,
    ax = NULL,
    # **kwargs,
    ...){
  # Well this is rather pythonesque...

  # """
  #   Function to plot in log-log a radially averaged Fourier spectrum.
  #   Parameters
  #   ----------
  #   fft_freq: array-like
  #       1d array containing the Fourier frequencies computed with the function
  #       :py:func:`pysteps.utils.spectral.rapsd`.
  #   fft_power: array-like
  #       1d array containing the radially averaged Fourier power spectrum
  #       computed with the function :py:func:`pysteps.utils.spectral.rapsd`.
  #   x_units: str, optional
  #       Units of the X variable (distance, e.g. "km").
  #   y_units: str, optional
  #       Units of the Y variable (amplitude, e.g. "dBR").
  #   wavelength_ticks: array-like, optional
  #       List of wavelengths where to show xticklabels.
  #   color: str, optional
  #       Line color.
  #   lw: float, optional
  #       Line width.
  #   label: str, optional
  #       Label (for legend).
  #   ax: Axes, optional
  #       Plot axes.
  #   Returns
  #   -------
  #   ax: Axes
  #       Plot axes
  #   """
  # Check input dimensions
  n_freq <- length(fft_freq)
  n_pow  <- length(fft_power)
  if(n_freq != n_pow){
    stop(paste0("Dimensions of the 1d input arrays must be equal. ", n_freq, " vs ", n_pow))
  }

  # alas, here our paths diverge, python.

  theData <- tibble(fft_freq = fft_freq, fft_power = fft_power) %>%
    filter(fft_freq > 0.0) %>%
    mutate(fft_freq  = 10 * log10(fft_freq),
           fft_power = 10 * log10(fft_power))

  g <- theData %>%
    ggplot(aes(fft_freq, fft_power)) +
    geom_line()


  # if(is.null(ax)){
  #   ax = plt.subplot(111)
  # }
  #
  # # Plot spectrum in log-log scale
  # ax.plot(
  #   10 * np.log10(fft_freq[np.where(fft_freq > 0.0)]),
  #   10 * np.log10(fft_power[np.where(fft_freq > 0.0)]),
  #   color=color,
  #   linewidth=lw,
  #   label=label
  # )

  # X-axis
  if(!is_null(wavelength_ticks)){
    # wavelength_ticks = np.array(wavelength_ticks)
    freq_ticks <- 1 / wavelength_ticks
    xticks <- 10 * log10(freq_ticks)
    xticklabels <- wavelength_ticks
    g <- g + scale_x_continuous(breaks = xticks, labels = xticklabels)
    if(!is.null(x_units)){
      xlabel <- paste0("Wavelength [", x_units, "]")
      g <- g + xlab(xlabel)
    }
  }else{
    if(!is.null(x_units)){
      xlabel <- paste0("Frequency [1/", x_units, "]")
      g <- g + xlab(xlabel)
    }
  }

  # Y-axis
  if(!is.null(y_units)){
    # { -> {{ with f-strings
    # power_units = fr"$10log_{{ 10 }}(\frac{{ {y_units}^2 }}{{ {x_units} }})$"
    power_units <- bquote("Power "~10*log[10] ~ frac(.(y_units)^2, .(x_units)))

    g <- g + ylab(power_units)

  }

  return(g)
}

