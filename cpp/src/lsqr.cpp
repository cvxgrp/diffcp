#include "lsqr.h"

// Vector lsqr(const LinearOperator& A, const Vector& b, const double damp, const double atol, const double btol, const double conlim,
//             int iter_lim, const bool show) {
//   int m = A.m;
//   int n = A.n;

//   if (iter_lim == -1) {
//     iter_lim = 2 * A.n;
//   }

//   Vector var = Vector::Zero(n);

//   int itn = 0;
//   int istop = 0;
//   int nstop = 0;
//   double ctol = 0.0;
//   if (conlim > 0) {
//     ctol = 1.0 / conlim;
//   }
//   double anorm = 0.0;
//   double dampsq = damp * damp;
//   double ddnorm = 0.0;
//   double res2 = 0.0;
//   double xnorm = 0.0;
//   double xxnorm = 0.0;
//   double z = 0.0;
// }