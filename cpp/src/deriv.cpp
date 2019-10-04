#include "deriv.h"
#include "lsqr.h"

Vector _solve_derivative(SparseMatrix Q, ListOfCones cones, Vector z, Vector rhs) {
  int N = z.size();

  LinearOperator M = (Q - eye(N)) * dpi(z, cones) + eye(N);
  LsqrResult result = lsqr(M, rhs);
  return result.x;
}

Vector _solve_adjoint_derivative(SparseMatrix Q, ListOfCones cones, Vector z, Vector dz) {
  int N = z.size();

  LinearOperator MT = dpi(z, cones).transpose() * (Q.transpose() - eye(N)) + eye(N);
  LsqrResult result = lsqr(MT, dz);
  return result.x
}