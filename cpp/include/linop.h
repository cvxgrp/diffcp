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
  int m;
  int n;
  VecFn matvec;
  VecFn rmatvec;

  LinearOperator(){};
  explicit LinearOperator(int rows, int cols, const VecFn &matvec_in,
                          const VecFn &rmatvec_in);
  LinearOperator operator+(const LinearOperator &obj);
  LinearOperator operator-(const LinearOperator &obj);
  LinearOperator operator*(const LinearOperator &obj);
  void transpose() { return std::swap(matvec, rmatvec); }
};

LinearOperator block_diag(const std::vector<LinearOperator> &linear_operators);

