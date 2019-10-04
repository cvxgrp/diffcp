#include "linop.h"

LinearOperator::LinearOperator(
    int rows, int cols, const std::function<Vector(const Vector &)> &matvec_in,
    const std::function<Vector(const Vector &)> &rmatvec_in) {
  m = rows;
  n = cols;
  matvec = matvec_in;
  rmatvec = rmatvec_in;
}

LinearOperator LinearOperator::operator+(const LinearOperator &obj) {
  const VecFn result_matvec = [this, obj](const Vector &x) -> Vector {
    return matvec(x) + obj.matvec(x);
  };
  const VecFn result_rmatvec = [this, obj](const Vector &x) -> Vector {
    return rmatvec(x) + obj.matvec(x);
  };
  return LinearOperator(m, n, result_matvec, result_rmatvec);
}

LinearOperator LinearOperator::operator-(const LinearOperator &obj) {
  const VecFn result_matvec = [this, obj](const Vector &x) -> Vector {
    return matvec(x) + obj.matvec(-x);
  };
  const VecFn result_rmatvec = [this, obj](const Vector &x) -> Vector {
    return rmatvec(x) + obj.rmatvec(-x);
  };
  return LinearOperator(m, n, result_matvec, result_rmatvec);
}

LinearOperator LinearOperator::operator*(const LinearOperator &obj) {
  const VecFn result_matvec = [this, obj](const Vector &x) -> Vector {
    return matvec(obj.matvec(x));
  };
  const VecFn result_rmatvec = [this, obj](const Vector &x) -> Vector {
    return obj.rmatvec(rmatvec(x));
  };
  return LinearOperator(m, n, result_matvec, result_rmatvec);
}

LinearOperator block_diag(const std::vector<LinearOperator> &linear_operators) {
  int rows = 0;
  int cols = 0;

  for (const auto &linop : linear_operators) {
    rows += linop.m;
    cols += linop.n;
  }

  const VecFn result_matvec = [linear_operators,
                               rows](const Vector &x) -> Vector {
    Vector out = Vector::Zero(rows);
    int i = 0;
    int j = 0;
    for (const auto &linop : linear_operators) {
      out.segment(i, i + linop.m) = linop.matvec(x.segment(j, j + linop.n));
      i += linop.m;
      j += linop.n;
    }
    return out;
  };
  const VecFn result_rmatvec = [linear_operators,
                                cols](const Vector &x) -> Vector {
    Vector out = Vector::Zero(cols);
    int i = 0;
    int j = 0;
    for (const auto &linop : linear_operators) {
      out.segment(i, i + linop.n) = linop.rmatvec(x.segment(j, j + linop.m));
      i += linop.n;
      j += linop.m;
    }
    return out;
  };

  return LinearOperator(rows, cols, result_matvec, result_rmatvec);
}
