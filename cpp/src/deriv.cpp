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

Vector _solve_derivative(const SparseMatrix &Q, const std::vector<Cone> &cones,
                         const Vector &u, const Vector &v, double w,
                         const Vector &rhs) {
  int N = u.size() + v.size() + 1;

  LinearOperator M =
      (aslinearoperator(Q) - identity(N)) * dpi(u, v, w, cones) + identity(N);
  LsqrResult result = lsqr(M, rhs);
  return result.x;
}

Vector _solve_adjoint_derivative(const SparseMatrix &Q,
                                 const std::vector<Cone> &cones,
                                 const Vector &u, const Vector &v, double w,
                                 const Vector &dz) {
  int N = u.size() + v.size() + 1;

  LinearOperator MT = dpi(u, v, w, cones).transpose() *
                          (aslinearoperator(Q).transpose() - identity(N)) +
                      identity(N);
  LsqrResult result = lsqr(MT, dz);
  return result.x;
}

SparseMatrix M_sparse(const SparseMatrix& Q, const std::vector<Cone>& cones,
                      const Vector& u, const Vector& v, double w) {
  int n = 1;
  SparseMatrix D(n,n);
  return D;
}

Vector _solve_derivative_sparse(const SparseMatrix& M, const Vector& rhs) {
  // TODO: Fill in
  int n = 1;
  Vector x = Vector::Zero(n);
  return x;
}

Vector _solve_adjoint_derivative_sparse(const SparseMatrix& MT, const Vector& dz) {
  // TODO: Fill in
  int n = 1;
  Vector x = Vector::Zero(n);
  return x;
}

Matrix M_dense(const Matrix& Q, const std::vector<Cone>& cones,
               const Vector& u, const Vector& v, double w) {
  // TODO: Fill in
  int n = 1;
  Matrix D = Matrix::Zero(n, n);
  return D;
}

Vector _solve_derivative_dense(const Matrix& M, const Vector& rhs) {
  // TODO: Fill in
  int n = 1;
  Vector x = Vector::Zero(n);
  return x;
}

Vector _solve_adjoint_derivative_dense(const Matrix& MT, const Vector& dz) {
  // TODO: Fill in
  int n = 1;
  Vector x = Vector::Zero(n);
  return x;
}
