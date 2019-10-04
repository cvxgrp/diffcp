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
                          const VecFn &rmatvec_in) : m(rows), n(cols), matvec(matvec_in), rmatvec(rmatvec) {};
  LinearOperator operator+(const LinearOperator &obj);
  LinearOperator operator-(const LinearOperator &obj);
  LinearOperator operator*(const LinearOperator &obj);
  LinearOperator transpose() { return LinearOperator(n, m, rmatvec, matvec); }
};

LinearOperator block_diag(const std::vector<LinearOperator> &linear_operators);
