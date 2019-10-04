#include "linop.h"
#include <iostream>
#include <assert.h>

LinearOperator LinearOperator::operator+(const LinearOperator &obj) {
  assert(m == obj.m);
  assert(n == obj.n);

  const VecFn result_matvec = [this, obj](const Vector &x) -> Vector {
    return matvec(x) + obj.matvec(x);
  };
  const VecFn result_rmatvec = [this, obj](const Vector &x) -> Vector {
    return rmatvec(x) + obj.matvec(x);
  };
  return LinearOperator(m, n, result_matvec, result_rmatvec);
}

LinearOperator LinearOperator::operator-(const LinearOperator &obj) {
  assert(m == obj.m);
  assert(n == obj.n);

  const VecFn result_matvec = [this, obj](const Vector &x) -> Vector {
    return matvec(x) + obj.matvec(-x);
  };
  const VecFn result_rmatvec = [this, obj](const Vector &x) -> Vector {
    return rmatvec(x) + obj.rmatvec(-x);
  };
  return LinearOperator(m, n, result_matvec, result_rmatvec);
}

LinearOperator LinearOperator::operator*(const LinearOperator &obj) {
  assert(n == obj.m);

  const VecFn result_matvec = [this, obj](const Vector &x) -> Vector {
    return matvec(obj.matvec(x));
  };
  const VecFn result_rmatvec = [this, obj](const Vector &x) -> Vector {
    return obj.rmatvec(rmatvec(x));
  };
  return LinearOperator(m, obj.n, result_matvec, result_rmatvec);
}

LinearOperator block_diag(const std::vector<LinearOperator> &linear_operators) {
  assert(linear_operators.size() > 0);

  int rows = 0;
  int cols = 0;

  for (const LinearOperator &linop : linear_operators) {
    rows += linop.m;
    cols += linop.n;
  }

  std::cout << rows << ", " << cols << std::endl;

  const VecFn result_matvec = [linear_operators,
                               rows, cols](const Vector &x) -> Vector {
    assert(x.size() == cols);
    Vector out = Vector::Zero(rows);
    int i = 0;
    int j = 0;
    for (const LinearOperator &linop : linear_operators) {
      out.segment(i, linop.m) = linop.matvec(x.segment(j, linop.n));
      i += linop.m;
      j += linop.n;
    }
    return out;
  };
  const VecFn result_rmatvec = [linear_operators,
                                rows, cols](const Vector &x) -> Vector {
    assert(x.size() == rows);
    Vector out = Vector::Zero(cols);
    int i = 0;
    int j = 0;
    for (const LinearOperator &linop : linear_operators) {
      out.segment(i, linop.n) = linop.rmatvec(x.segment(j, linop.m));
      i += linop.n;
      j += linop.m;
    }
    return out;
  };

  return LinearOperator(rows, cols, result_matvec, result_rmatvec);
}
