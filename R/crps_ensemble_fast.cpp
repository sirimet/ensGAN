#include <Rcpp.h>
using namespace Rcpp;
#include <math.h>


//  compute approximate crps for an ensemble
// [[Rcpp::export]]
NumericVector crps_ensemble_fast(NumericMatrix ens, NumericVector obs) {
  int            n = ens.nrow();
  int            m = ens.ncol();
  double         crps1, crps2;
  NumericVector  crps(n);

  for (int i=0; i<n; i++) {
    crps1 = 0;
    for (int j=0; j<m; j++)
      crps1 += fabs(ens(i,j) - obs(i));

      crps2 = 0;
      for (int j=0; j<m-1; j++)
        crps2 += fabs(ens(i,j) - ens(i,j+1));

        crps(i) = crps1/m - crps2/(2*m);
  }

  return(crps);
}
