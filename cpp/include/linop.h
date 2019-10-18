#pragma once

#include "eigen_includes.h"
#include <functional>
#include <vector>

using VecFn = std::function<Vector(const Vector &)>;

class LinearOperator {
  /**
   * m x n linear operator
   */
public:
  const int m;
  const int n;
  const VecFn matvec;
  const VecFn rmatvec;

  explicit LinearOperator(int rows, int cols, const VecFn &matvec_in,
                          const VecFn &rmatvec_in)
      : m(rows), n(cols), matvec(matvec_in), rmatvec(rmatvec_in){};
  LinearOperator operator+(const LinearOperator &obj) const;
  LinearOperator operator-(const LinearOperator &obj) const;
  LinearOperator operator*(const LinearOperator &obj) const;
  LinearOperator transpose() const {
    return LinearOperator(n, m, rmatvec, matvec);
  }

  Vector apply_matvec(const Vector &x) const { return matvec(x); }
  Vector apply_rmatvec(const Vector &x) const { return rmatvec(x); }
};

LinearOperator block_diag(const std::vector<LinearOperator> &linear_operators);
LinearOperator aslinearoperator(const Matrix &A);
LinearOperator aslinearoperator(const SparseMatrix &A);
LinearOperator zero(int m, int n);
LinearOperator identity(int n);
LinearOperator diag(const Array &coefficients);
LinearOperator scalar(double x);
