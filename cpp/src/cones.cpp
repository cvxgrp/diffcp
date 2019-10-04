#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

#include "cones.h"
#include "exp.h"

constexpr double CONE_THRESH = 1e-6;
const double EulerConstant = std::exp(1.0);

bool in_exp(const Eigen::Vector3d &x) {
  return (x[0] <= 0 && x[1] == 0 && x[2] >= 0) or
         (x[1] > 0 && x[1] * exp(x[0] / x[1]) - x[2] <= CONE_THRESH);
}

bool in_exp_dual(const Eigen::Vector3d &x) {
  return (x[0] == 0 && x[1] >= 0 && x[2] >= 0) or
         (x[0] < 0 &&
          -x[0] * exp(x[1] / x[0]) - EulerConstant * x[2] <= CONE_THRESH);
}

int vectorized_psd_size(int n) { return n * (n + 1) / 2; }

Vector lower_triangular_from_matrix(const Matrix &matrix) {
  int n = matrix.rows();
  Vector lower_tri = Vector::Zero(vectorized_psd_size(n));
  double sqrt_two = std::sqrt(2.0);
  int offset = 0;
  for (int col = 0; col < n; ++col) {
    for (int row = col; row < n; ++row) {
      if (row != col) {
        lower_tri[offset] = matrix(row, col) * sqrt_two;
      } else {
        lower_tri[offset] = matrix(row, col);
      }
      ++offset;
    }
  }
  return lower_tri;
}

Matrix matrix_from_lower_triangular(const Vector &lower_tri) {
  int n = static_cast<int>(std::sqrt(2 * lower_tri.size()));
  Matrix matrix = Matrix::Zero(n, n);
  double sqrt_two = sqrt(2.0);
  int offset = 0;
  for (int col = 0; col < n; ++col) {
    for (int row = col; row < n; ++row) {
      if (row != col) {
        matrix(row, col) = lower_tri[offset] / sqrt_two;
        matrix(col, row) = lower_tri[offset] / sqrt_two;
      } else {
        matrix(row, col) = lower_tri[offset];
      }
      ++offset;
    }
  }
  return matrix;
}

LinearOperator _dprojection_exp(const Vector &x, bool dual) {
  int num_cones = x.size() / 3;
  std::vector<LinearOperator> ops;
  ops.reserve(num_cones);
  int offset = 0;
  for (int i = 0; i < num_cones; ++i) {
    Eigen::Vector3d x_i;
    if (dual) {
      x_i = -1 * x.segment(offset, 3);
    } else {
      x_i = x.segment(offset, 3);
    }
    offset += 3;

    if (in_exp(x_i)) {
      ops.emplace_back(identity(3));
    } else if (in_exp_dual(x_i)) {
      ops.emplace_back(zero(3, 3));
    } else if (x_i[0] < 0 && x_i[1] < 0) {
      const VecFn matvec = [x_i](const Vector &y) -> Vector {
        Eigen::Vector3d out;
        double last_component = 0;
        if (x_i[2] >= 0) {
          last_component = y[2];
        }
        out << y[0], 0, last_component;
        return out;
      };
      ops.emplace_back(LinearOperator(3, 3, matvec, matvec));
    } else {
      double t;
      double rs[3] = {0.0, 0.0, 0.0};
      assert(_proj_exp_cone(rs, &t) == 0);
      double r = rs[0];
      double s = rs[1];
      if (s == 0) {
        // TODO(akshayka): print warning
        s = std::abs(r);
      }
      double l = t - x_i[2];
      double alpha = std::exp(r / s);
      double beta = l * r / (s * s) * alpha;

      Eigen::Matrix<double, 4, 4> J_inv;
      // clang-format off
      J_inv << alpha, (-r + s) / s * alpha, -1, 0,
               1 + l / s * alpha, -beta, 0, alpha,
               -beta, 1 + beta * r / s, 0, (1 - r / s) * alpha,
               0, 0, 1, -1;
      // clang-format on
      // extract a 3x3 subblock, with top-left corner at row 0, column 1
      const Matrix J = J_inv.inverse().block<3, 3>(0, 1);
      ops.emplace_back(aslinearoperator(J));
    }
  }

  const LinearOperator D = block_diag(ops);
  if (dual) {
    const VecFn matvec = [D](const Vector &y) -> Vector {
      return y - D.matvec(y);
    };
    const VecFn rmatvec = [D](const Vector &y) -> Vector {
      return y - D.rmatvec(y);
    };
    return LinearOperator(x.size(), x.size(), matvec, rmatvec);
  }
  return D;
}

LinearOperator _dprojection_psd(const Vector &x) {
  int n = x.size();
  const Matrix &X = matrix_from_lower_triangular(x);
  Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(X.rows());
  eigen_solver.compute(X);
  const Vector &eigenvalues = eigen_solver.eigenvalues();
  const Matrix &Q = eigen_solver.eigenvectors();
  if (eigenvalues[0] >= 0) {
    return identity(n);
  }
  // k is the index of the first nonnegative index
  int k = -1;
  for (int i = 0; i < eigenvalues.size(); ++i) {
    if (eigenvalues[i] >= 0) {
      k = i;
      break;
    }
  }
  assert(k > -1);
  const VecFn matvec = [eigenvalues, Q, k](const Vector &y) -> Vector {
    Matrix tmp = Q.transpose() * matrix_from_lower_triangular(y) * Q;
    // Componentwise multiplication by the matrix `B` from BMB'18.
    for (int i = 0; i < tmp.rows(); ++i) {
      for (int j = 0; j < tmp.cols(); ++j) {
        if (i <= k && j <= k) {
          tmp(i, j) = 0;
        } else if (i > k && j <= k) {
          double lambda_i_pos = std::max(eigenvalues[i], 0.0);
          double lambda_j_neg = std::min(eigenvalues[j], 0.0);
          tmp(i, j) *= lambda_i_pos / (lambda_j_neg + lambda_i_pos);
        } else if (i <= k && j > k) {
          double lambda_i_neg = std::min(eigenvalues[i], 0.0);
          double lambda_j_pos = std::max(eigenvalues[j], 0.0);
          tmp(i, j) *= lambda_j_pos / (lambda_i_neg + lambda_j_pos);
        }
      }
    }
    Matrix result = Q * tmp * Q.transpose();
    return lower_triangular_from_matrix(result);
  };

  return LinearOperator(n, n, matvec, matvec);
}

LinearOperator _dprojection_soc(const Vector &x) {
  int n = x.size();
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
    return LinearOperator(n, n, matvec, matvec);
  }
}

LinearOperator _dprojection_pos(const Vector &x) {
  const Array sign = x.cwiseSign();
  return diag(0.5 * (sign + 1));
}

LinearOperator _dprojection_zero(const Vector &x, bool dual) {
  int n = x.size();
  return dual ? identity(n) : zero(n, n);
}

LinearOperator _dprojection(const Vector &x, ConeType type, bool dual) {
  if (type == ZERO) {
    return _dprojection_zero(x, dual);
  } else if (type ==POS) {
    return _dprojection_pos(x);
  } else if (type == SOC) {
    return _dprojection_soc(x);
  } else if (type == PSD) {
    return _dprojection_psd(x);
  } else {
    assert(type == EXP);
    return _dprojection_exp(x, dual);
  }
}

LinearOperator dprojection(const Vector &x, const std::vector<Cone> &cones,
                           bool dual) {
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
        size = vectorized_psd_size(size);
      } else if (type == EXP) {
        size *= 3;
      }
      lin_ops.emplace_back(_dprojection(x.segment(offset, size), type, dual));
      offset += size;
    }
  }
  return block_diag(lin_ops);
}
