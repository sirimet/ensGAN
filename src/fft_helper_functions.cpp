#include <Rcpp.h>
using namespace Rcpp;

// Function to compute a 2D coordinate array, where the origin is at the center
// [[Rcpp::export]]
List compute_centred_coord_array(int M, int N) {
  IntegerVector s1, s2;

  if (M % 2 == 1) {
    s1 = seq(-floor(M / 2), floor(M / 2));
  } else {
    s1 = seq(1 - floor(M / 2), floor(M / 2));
  }

  if (N % 2 == 1) {
    s2 = seq(-floor(N / 2), floor(N / 2));
  } else {
    s2 = seq(1 - floor(N / 2), floor(N / 2));
  }

  NumericMatrix YC = NumericMatrix(s1.size(), 1, s1.begin());
  NumericMatrix XC = NumericMatrix(1, s2.size(), s2.begin());

  return List::create(YC, XC);
}

// Function to swap up and down for fft
// [[Rcpp::export]]
ComplexMatrix fft_swap_up_down(ComplexMatrix input_matrix) {
  int rows = input_matrix.nrow();
  int cols = input_matrix.ncol();
  int rows_half = std::floor(rows / 2);

  ComplexMatrix result(rows, cols);

  // Copying the lower half to the upper half
  for (int i = 0; i < rows_half; ++i) {
    for (int j = 0; j < cols; ++j) {
      result(i, j) = input_matrix(i + rows - rows_half, j);
    }
  }

  // Copying the upper half to the lower half
  for (int i = rows_half; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result(i, j) = input_matrix(i - rows_half, j);
    }
  }

  return result;
}

// Function to swap left and right for fft
// [[Rcpp::export]]
ComplexMatrix fft_swap_left_right(ComplexMatrix input_matrix) {
  int rows = input_matrix.nrow();
  int cols = input_matrix.ncol();
  int cols_half = std::ceil(cols / 2);

  ComplexMatrix result(rows, cols);

  // Copying the right half to the left half
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols_half; ++j) {
      result(i, j) = input_matrix(i, j + cols - cols_half);
    }
  }

  // Copying the left half to the right half
  for (int i = 0; i < rows; ++i) {
    for (int j = cols_half; j < cols; ++j) {
      result(i, j) = input_matrix(i, j - cols_half);
    }
  }

  return result;
}

// Main fftshift function
// [[Rcpp::export]]
ComplexMatrix fftshift(ComplexMatrix input_matrix, int dim = -1) {
  if (dim == -1) {
    input_matrix = fft_swap_up_down(input_matrix);
    return fft_swap_left_right(input_matrix);
  } else if (dim == 1) {
    return fft_swap_up_down(input_matrix);
  } else if (dim == 2) {
    return fft_swap_left_right(input_matrix);
  } else {
    stop("Invalid dimension parameter");
  }
}

// Function to swap up and down for ifft
// [[Rcpp::export]]
ComplexMatrix ifft_swap_up_down(ComplexMatrix input_matrix) {
  int rows = input_matrix.nrow();
  int cols = input_matrix.ncol();
  int rows_half = std::floor(rows / 2.0);
  int rows_other = rows - rows_half;

  ComplexMatrix result(rows, cols);

  // Copying the lower half to the upper half
  for (int i = 0; i < rows_other; ++i) {
    for (int j = 0; j < cols; ++j) {
      result(i, j) = input_matrix(i + rows_half, j);
    }
  }

  // Copying the upper half to the lower half
  for (int i = rows_other; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result(i, j) = input_matrix(i - rows_other, j);
    }
  }

  return result;
}

// Function to swap left and right for ifft
// [[Rcpp::export]]
ComplexMatrix ifft_swap_left_right(ComplexMatrix input_matrix) {
  int rows = input_matrix.nrow();
  int cols = input_matrix.ncol();
  int cols_half = std::ceil(cols / 2);
  int cols_other = cols - cols_half;

  ComplexMatrix result(rows, cols);

  // Copying the right half to the left half
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols_other; ++j) {
      result(i, j) = input_matrix(i, j + cols - cols_other);
    }
  }

  // Copying the left half to the right half
  for (int i = 0; i < rows; ++i) {
    for (int j = cols_other; j < cols; ++j) {
      result(i, j) = input_matrix(i, j - cols_other);
    }
  }

  return result;
}

// Main ifftshift function
// [[Rcpp::export]]
ComplexMatrix ifftshift(ComplexMatrix input_matrix, int dim = -1) {
  if (dim == -1) {
    input_matrix = ifft_swap_up_down(input_matrix);
    return ifft_swap_left_right(input_matrix);
  } else if (dim == 1) {
    return ifft_swap_up_down(input_matrix);
  } else if (dim == 2) {
    return ifft_swap_left_right(input_matrix);
  } else {
    stop("Invalid dimension parameter");
  }
}


// [[Rcpp::export]]
NumericVector fftfreq(int n, double d = 1.0) {
  if (n <= 0) {
    stop("n should be a positive integer");
  }

  double val = 1.0 / (n * d);
  NumericVector results(n);
  int N = (n - 1) / 2 + 1;

  // Fill the first half
  for (int i = 0; i < N; ++i) {
    results[i] = i;
  }

  // Fill the second half
  for (int i = N, j = -(n / 2); i < n; ++i, ++j) {
    results[i] = j;
  }

  return results * val;
}
