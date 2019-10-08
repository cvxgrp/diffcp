#include "deriv.h"
#include "lsqr.h"

inline double gt(double x, double t) {
  if (x >= t) {
    return 1.0;
  } else {
    return 0.0;
  }
}

LinearOperator dpi(const Vector &u, const Vector &v, double w,
                   const std::vector<Cone> &cones) {
  LinearOperator eye = identity(u.size());
  LinearOperator D_proj = dprojection(v, cones, true);
  LinearOperator last = scalar(gt(w, 0.0));

  std::vector<LinearOperator> linops{eye, D_proj, last};

  return block_diag(linops);
}

LinearOperator M_operator(const SparseMatrix &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w) {
  int N = u.size() + v.size() + 1;
  return (aslinearoperator(Q) - identity(N)) * dpi(u, v, w, cones) +
         identity(N);
}

Matrix dpi_dense(const Vector &u, const Vector &v, double w,
                 const std::vector<Cone> &cones) {
  int n = u.size();
  int m = v.size();
  int N = n + m + 1;
  Matrix D = Matrix::Zero(N, N);
  D.block(0, 0, n, n).diagonal().array() += 1.;
  // Could be optimized by having dprojection_dense modifying this in-place,
  // or by not explicitly adding the first and last blocks.
  D.block(n, n, m, m) = dprojection_dense(v, cones, true);
  D(N - 1, N - 1) = gt(w, 0.0);
  return D;
}

Matrix M_dense(const Matrix &Q, const std::vector<Cone> &cones, const Vector &u,
               const Vector &v, double w) {
  int n = u.size();
  int m = v.size();
  int N = n + m + 1;
  Matrix eye = Matrix::Identity(N, N);
  return (Q - eye) * dpi_dense(u, v, w, cones) + eye;
}

Vector _solve_derivative_dense(const Matrix &M, const Matrix &MT,
                               const Vector &rhs) {
  // TODO: Factorization could be cached to optimize multiple calls
  return (MT * M).ldlt().solve(MT * rhs);
}

Vector _solve_adjoint_derivative_dense(const Matrix &M, const Matrix &MT,
                                       const Vector &dz) {
  // TODO: Factorization could be cached to optimize multiple calls
  return (M * MT).ldlt().solve(M * dz);
}
