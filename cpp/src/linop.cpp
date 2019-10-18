#include "linop.h"
#include <assert.h>

LinearOperator LinearOperator::operator+(const LinearOperator &obj) const {
  assert(m == obj.m);
  assert(n == obj.n);

  const LinearOperator this_op = *this;
  const VecFn result_matvec = [this_op, obj](const Vector &x) -> Vector {
    return this_op.matvec(x) + obj.matvec(x);
  };
  const VecFn result_rmatvec = [this_op, obj](const Vector &x) -> Vector {
    return this_op.rmatvec(x) + obj.rmatvec(x);
  };
  return LinearOperator(m, n, result_matvec, result_rmatvec);
}

LinearOperator LinearOperator::operator-(const LinearOperator &obj) const {
  assert(m == obj.m);
  assert(n == obj.n);

  const LinearOperator this_op = *this;
  const VecFn result_matvec = [this_op, obj](const Vector &x) -> Vector {
    return this_op.matvec(x) + obj.matvec(-x);
  };
  const VecFn result_rmatvec = [this_op, obj](const Vector &x) -> Vector {
    return this_op.rmatvec(x) + obj.rmatvec(-x);
  };
  return LinearOperator(m, n, result_matvec, result_rmatvec);
}

LinearOperator LinearOperator::operator*(const LinearOperator &obj) const {
  assert(n == obj.m);

  const LinearOperator this_op = *this;
  const VecFn result_matvec = [this_op, obj](const Vector &x) -> Vector {
    return this_op.matvec(obj.matvec(x));
  };
  const VecFn result_rmatvec = [this_op, obj](const Vector &x) -> Vector {
    return obj.rmatvec(this_op.rmatvec(x));
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

  const VecFn result_matvec = [linear_operators, rows,
                               cols](const Vector &x) -> Vector {
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
  const VecFn result_rmatvec = [linear_operators, rows,
                                cols](const Vector &x) -> Vector {
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

LinearOperator aslinearoperator(const Matrix &A) {
  const VecFn result_matvec = [A](const Vector &x) -> Vector { return A * x; };
  const VecFn result_rmatvec = [A](const Vector &x) -> Vector {
    return A.transpose() * x;
  };
  return LinearOperator(A.rows(), A.cols(), result_matvec, result_rmatvec);
}

LinearOperator aslinearoperator(const SparseMatrix &A) {
  const VecFn result_matvec = [A](const Vector &x) -> Vector { return A * x; };
  const VecFn result_rmatvec = [A](const Vector &x) -> Vector {
    return A.transpose() * x;
  };
  return LinearOperator(A.rows(), A.cols(), result_matvec, result_rmatvec);
}

LinearOperator zero(int m, int n) {
  const VecFn matvec = [](const Vector &x) -> Vector {
    return Vector::Zero(x.size());
  };
  return LinearOperator(m, n, matvec, matvec);
}

LinearOperator identity(int n) {
  const VecFn matvec = [](const Vector &x) -> Vector { return x; };
  return LinearOperator(n, n, matvec, matvec);
}

LinearOperator diag(const Array &coefficients) {
  const VecFn matvec = [coefficients](const Vector &x) -> Vector {
    return (coefficients * x.array()).matrix();
  };
  return LinearOperator(coefficients.size(), coefficients.size(), matvec,
                        matvec);
}

LinearOperator scalar(double num) {
  const VecFn matvec = [num](const Vector &x) -> Vector {
    Vector result = Vector::Zero(1);
    result[0] = num * x[0];
    return result;
  };
  return LinearOperator(1, 1, matvec, matvec);
}
