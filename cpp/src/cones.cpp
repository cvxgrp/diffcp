#include <functional>
#include <numeric>

#include "cones.h"
#include "exp.h"


Vector lower_triangular_from_matrix(const Matrix &matrix) {

}

Matrix full_matrix_from_lower_triangular(const Vector &lower_tri, int dim) {
  Matrix mat = Matrix::Zero(dim, dim);
  int offset = 0;
  for (int col = 0; col < dim; ++col) {
    for (int row = col; row < dim; ++row) {
      mat[row, col] = lower_tri[offset];  
      mat[col, row] = lower_tri[offset];
      ++offset;
    }
  }
}

LinearOperator _dprojection(Vector x, ConeType type, bool dual) {
  int m = x.size();
  int n = x.size();
  switch (type) {
  case ZERO: {
    return dual ? identity(n) : zero(n, n);
  }
  case POS: {
    const Array sign = x.cwiseSign();
    return diag(0.5 * (sign + 1));
  }
  case SOC: {
    const double t = x[0];
    const Vector &z = x.segment(1, n - 1);
    const double norm_z = z.norm();
    if (norm_z <= t) {
      return identity(n);
    } else if (norm_z <= -t) {
      return zero(n, n);
    } else {
      const Vector unit_z = z / norm_z;
      const VecFn matvec = [t, z, unit_z, norm_z](const Vector &y) -> Vector {
        double y_t = y[0];
        const Vector &y_z = y.segment(1, y.size() - 1);
        const double first_chunk = norm_z * y_t + z.dot(y_z);
        const Vector second_chunk =
            z * y_t + (t + norm_z) * y_z - t * unit_z * unit_z.dot(y_z);
        Vector output(1 + second_chunk.size());
        output << first_chunk, second_chunk;
        return (1.0 / (2 * norm_z)) * output;
      };
      return LinearOperator(m, n, matvec, matvec);
    }
  }
  case PSD: {
    // TODO
    return identity(n);
  }
  case EXP: {
    // TODO
    return identity(n);
  }
  }
}

LinearOperator dprojection(Vector x, std::vector<Cone> cones, bool dual) {
  std::vector<LinearOperator> lin_ops;

  int offset = 0;
  for (const Cone &cone : cones) {
    const ConeType &type = cone.type;
    const std::vector<int> &sizes = cone.sizes;
    if (std::accumulate(sizes.begin(), sizes.end(), 0) == 0) {
      continue;
    }
    for (int size : sizes) {
      if (type == PSD) {
        size = size * (size + 1) / 2;
      } else if (type == EXP) {
        size *= 3;
      }
      lin_ops.emplace_back(_dprojection(x.segment(offset, size), type, dual));
      offset += size;
    }
  }
  return block_diag(lin_ops);
}
