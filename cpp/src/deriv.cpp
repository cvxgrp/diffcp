#include "deriv.h"
#include "lsqr.h"

inline double gt(double x, double t) {
  if (x >= t) {
    return 1.0;
  } else {
    return 0.0;
  }
}

LinearOperator dpi(const Vector& u, const Vector& v, double w, const std::vector<Cone>& cones) {
  LinearOperator eye = identity(u.size());
  LinearOperator D_proj = dprojection(v, cones, true);
  LinearOperator last = scalar(gt(w, 0.0));

  std::vector<LinearOperator> linops {eye, D_proj, last};

  return block_diag(linops);
}

Vector _solve_derivative(const SparseMatrix& Q, const std::vector<Cone>& cones,
    const Vector& u, const Vector& v, double w, const Vector& rhs) {
  int N = u.size() + v.size() + 1;

  LinearOperator M = (Q - identity(N)) * dpi(u, v, w, cones) + identity(N);
  LsqrResult result = lsqr(M, rhs);
  return result.x;
}

Vector _solve_adjoint_derivative(const SparseMatrix& Q, const std::vector<Cone>& cones,
    const Vector& u, const Vector& v, double w, const Vector& dz) {
  int N = u.size() + v.size() + 1;

  LinearOperator MT = dpi(u, v, w, cones).transpose() * (Q.transpose() - identity(N)) + identity(N);
  LsqrResult result = lsqr(MT, dz);
  return result.x
}